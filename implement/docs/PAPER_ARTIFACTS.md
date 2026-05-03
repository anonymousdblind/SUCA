# PAPER ARTIFACTS

这个目录下新增了两层代码：

- `scripts/extract_report_artifacts.py`：把当前的 `eval_results/evaluation_report.md` 转成标准化 JSON。
- `scripts/build_paper_artifacts.py`：读取一个 manifest，把 benchmark summary、训练诊断、耗时 profile、奖励曲线和定性图拼装成论文需要的表和图。
- `scripts/export_time_breakdown.py`：把 profiling 结果转成 `time_breakdown.json`。
- `scripts/export_reward_curves.py`：把 wandb/local metrics 导成论文 builder 需要的 reward curve CSV。
- `scripts/export_geneval_summary.py` / `scripts/eval_dpgbench_summary.py`：把正式评测结果整理成标准 summary JSON。
- `scripts/eval_dpgbench_formal.py`：把外部 mPLUG 打分、raw score CSV 和标准 summary JSON 串成单脚本工作流。
- `scripts/eval_formal.py`：现在原生支持 `--summary_json`，可直接写 Geneval2 标准 summary JSON。
- `tools/run_mplug_score.py`：DPG 外部打分适配器接口，后续只替换内部 mPLUG 调用即可。

`tools/run_mplug_score.py` 现在支持两种真实接法：

- `--transport local`：导入本地 Python scorer，适合直接加载模型权重。
- `--transport http`：请求远程评分服务，适合把 mPLUG 独立部署。

`scripts/eval_dpgbench_formal.py` 现在支持 `--backend-extra-json`，可以把设备号、batch size、prompt template、checkpoint tag 等额外参数统一透传给 scoring command template，避免把长参数串硬编码进模板本身。

## 1. 为什么要加这一层

论文里的结果展示不只是单个 benchmark 分数，还包括：

- 主文三张 benchmark 表
- partial results 表
- ablation 表
- appendix 的 atom / spatial / dpg 分桶表
- reward curves 图
- qualitative 拼图
- config/time breakdown 表

现有 `evaluate.py`、`eval_formal.py`、`eval_spatial_api.py` 只能分别产出单项结果，还不能自动回写到论文结构对应的表图。新的 artifact builder 负责这一步。

## 2. 标准输入格式

推荐把每个 benchmark 的结果都整理成一个 JSON 文件。

### 2.1 GenEval2

```json
{
  "benchmark": "geneval2",
  "overall": {"soft_tifa_gm": 21.93},
  "skills": {
    "Object": 87.12,
    "Attribute": 72.26,
    "Count": 47.49,
    "Position": 43.04,
    "Verb": 16.24
  },
  "atom_count": {
    "3": 52.62,
    "4": 46.86,
    "5": 25.95,
    "6": 20.16,
    "7": 9.03,
    "8": 9.18,
    "9": 9.20,
    "10": 2.47
  }
}
```

### 2.2 SpatialGenEval

```json
{
  "benchmark": "spatialgeneval",
  "overall": {"avg_acc": 66.40},
  "dimensions": {
    "Left/Right": 68.20,
    "Above/Below": 64.10,
    "Inside/Around": 59.30,
    "Front/Behind": 55.80
  }
}
```

### 2.3 DPG-Bench

```json
{
  "benchmark": "dpgbench",
  "overall": {"Overall": 73.80},
  "categories": {
    "Entity": 82.10,
    "Attribute": 74.90,
    "Relation": 64.40
  }
}
```

## 3. 先把当前 report 转成 JSON

在 `implement/` 下运行：

```bash
python scripts/extract_report_artifacts.py \
  --report eval_results/evaluation_report.md \
  --output-dir eval_results/normalized
```

这一步会生成：

- `eval_results/normalized/base_geneval2.json`
- `eval_results/normalized/sft_geneval2.json`
- `eval_results/normalized/base_spatialgeneval.json`
- `eval_results/normalized/sft_spatialgeneval.json`
- `eval_results/normalized/training_diagnostics.json`

## 3.1 训练分析资产导出

```bash
python scripts/export_time_breakdown.py \
  --input profiling/time_profile.csv \
  --output analysis/time/time_breakdown.json

python scripts/export_reward_curves.py \
  --input wandb_exports/suca_metrics.csv \
  --output analysis/curves/suca_reward_curves.csv \
  --label "\\method"
```

如果你启用了新的训练器结构化日志，这两步也可以直接吃训练产出的 `analysis/time/step_timing.jsonl` 和 `analysis/curves/per_unit_rewards.jsonl`，不再依赖 wandb 导出。

## 3.2 统一结果目录方案

固定目录方案见 [implement/docs/PAPER_RESULT_SCHEME.md](/Volumes/Happy/paper/基于强化学习的T2I增强/implement/docs/PAPER_RESULT_SCHEME.md)。

如果你愿意把所有 benchmark summary 都写到固定路径，直接使用 [implement/docs/paper_artifact_manifest.canonical.json](/Volumes/Happy/paper/基于强化学习的T2I增强/implement/docs/paper_artifact_manifest.canonical.json) 即可，后续不需要改 manifest。

## 4. 生成论文表图

使用示例 manifest：

```bash
python scripts/build_paper_artifacts.py \
  --manifest docs/paper_artifact_manifest.example.json \
  --output-dir paper_artifacts
```

输出目录默认会包含：

- `paper_artifacts/tables/*.csv`
- `paper_artifacts/tables/*_rows.tex`
- `paper_artifacts/figures/*.png`
- `paper_artifacts/reports/paper_artifact_report.md`

## 5. 当前脚本能直接覆盖的论文对象

- 主文 `tab:geneval_main`
- 主文 `tab:spatial_main`
- 主文 `tab:dpg_main`
- 主文 partial results 表
- 主文 `tab:ablations`
- 附录 `tab:config_v3`
- 附录 `tab:time_v3`
- 附录 `tab:geneval_atom_v3`
- 附录 `tab:spatial_dim_v3`
- 附录 `tab:dpg_cat_v3`
- 附录 `fig:reward_curves_v3`
- 定性拼图 `fig:qualitative`

## 6. 后续接入建议

- GenEval2 正式评测结束后，把每个方法的结果落成一个标准 JSON，再在 manifest 里补路径。
- SpatialGenEval 如果原始输出是更细粒度的 10 维结果，先在评测脚本里导出论文使用的 4 维聚合结果，再交给 artifact builder。
- DPG-Bench 评测完成后，至少导出 `Entity`、`Attribute`、`Relation`、`Overall` 四列。
- 训练曲线建议统一导出成 CSV，至少包含 `step` 和四类 unit reward。
- profiling 统一导出成 `analysis/time_breakdown.json`，避免手工填表。

## 7. 边界

- 这个 builder 不负责重新跑 benchmark，只负责把已经完成的训练和评测结果转成论文资产。
- 如果 manifest 里某些文件还不存在，脚本不会中断整个构建，而会在 `paper_artifact_report.md` 里记录缺失项。