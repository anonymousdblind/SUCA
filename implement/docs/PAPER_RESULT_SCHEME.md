# PAPER RESULT SCHEME

目标是把论文 builder 的输入路径固定下来，让你跑完训练和正式评测后，不需要再改 manifest，只需要把标准 summary JSON 和分析文件写到约定目录。

## 1. 固定目录约定

建议统一使用下面这棵树：

```text
implement/
├── analysis/
│   ├── summaries/
│   │   ├── geneval2/
│   │   │   ├── base.json
│   │   │   ├── sft_warmup.json
│   │   │   ├── flow_grpo_scalar.json
│   │   │   └── suca.json
│   │   ├── spatialgeneval/
│   │   │   ├── base.json
│   │   │   ├── sft_warmup.json
│   │   │   ├── flow_grpo_scalar.json
│   │   │   └── suca.json
│   │   └── dpgbench/
│   │       ├── base.json
│   │       ├── flow_grpo_scalar.json
│   │       └── suca.json
│   ├── curves/
│   │   ├── scalar_reward_curves.csv
│   │   └── suca_reward_curves.csv
│   └── time/
│       └── time_breakdown.json
└── docs/
    └── paper_artifact_manifest.canonical.json
```

这样 manifest 可以固定一次，不需要每次换 checkpoint 或换评测结果再改路径。

## 2. 三个 benchmark 的标准 summary JSON

### 2.1 GenEval2

```json
{
  "benchmark": "geneval2",
  "variant": "suca",
  "overall": {"soft_tifa_gm": 24.31},
  "skills": {
    "Object": 88.10,
    "Attribute": 75.40,
    "Count": 52.80,
    "Position": 46.20,
    "Verb": 18.50
  },
  "atom_count": {
    "3": 54.0,
    "4": 48.7,
    "5": 28.4,
    "6": 23.1,
    "7": 12.0,
    "8": 10.2,
    "9": 8.9,
    "10": 5.1
  }
}
```

### 2.2 SpatialGenEval

```json
{
  "benchmark": "spatialgeneval",
  "variant": "suca",
  "overall": {
    "avg_acc": 68.2,
    "basic_acc": 77.1,
    "spatial_acc": 65.9
  },
  "dimensions": {
    "Left/Right": 69.1,
    "Above/Below": 67.8,
    "Inside/Around": 63.4,
    "Front/Behind": 61.5
  }
}
```

### 2.3 DPG-Bench

```json
{
  "benchmark": "dpgbench",
  "variant": "suca",
  "overall": {"Overall": 74.8},
  "categories": {
    "Entity": 82.1,
    "Attribute": 75.3,
    "Relation": 66.4
  }
}
```

## 3. 推荐接法

### 3.1 GenEval2

`eval_formal.py` 现在已经支持直接输出标准 summary JSON，不再必须经过单独转换：

```bash
python scripts/eval_formal.py \
  --checkpoint logs/fullparam_restored/checkpoints/checkpoint-2800 \
  --comparison-json eval_results/formal_comparison.json \
  --summary-json analysis/summaries/geneval2/flow_grpo_scalar.json \
  --summary-model-key grpo \
  --summary-variant-name flow_grpo_scalar
```

SUCA checkpoint 同理导出到 `analysis/summaries/geneval2/suca.json`。

### 3.2 SpatialGenEval

直接改 `eval_spatial_api.py`，让它在保存原始 `output_json` 的同时，再写一个标准 summary JSON。

### 3.3 DPG-Bench

推荐直接走单脚本工作流：

```bash
python scripts/eval_dpgbench_formal.py \
  --image-dir eval_results/suca/dpgbench/images \
  --prompt-dir data/dpg-bench/dpg_bench/prompts \
  --raw-score-csv eval_results/dpgbench/suca_raw_scores.csv \
  --summary-json analysis/summaries/dpgbench/suca.json \
  --variant-name suca \
  --backend-extra-json '{"device":"cuda:0","batch_size":8}' \
  --score-command-template 'python tools/run_mplug_score.py --backend mplug --transport local --scorer-target tools.mplug_local_backend_example:score_image_prompt --model-path /path/to/mplug_weights --backend-extra-json {backend_extra_json_quoted} --image {image_quoted} --prompt {prompt_quoted} --output {output_quoted}'
```

其中外部命令只需要往 `{output}` 写一个 JSON，包含 `overall_score`、`score` 或 `dpg_score` 任一字段即可。
如果你是通过远程服务暴露 mPLUG 打分，则模板改成：

```bash
python tools/run_mplug_score.py --backend mplug --transport http --server-url http://127.0.0.1:9000/score --backend-extra-json {backend_extra_json_quoted} --image {image_quoted} --prompt {prompt_quoted} --output {output_quoted}
```

如果只是做链路 smoke test，可以临时换成 `--backend mock`，但这个分数不能用于论文结果。

仓库内的 [implement/tools/mplug_local_backend_example.py](/Volumes/Happy/paper/基于强化学习的T2I增强/implement/tools/mplug_local_backend_example.py) 给出了 `--scorer-target` 的标准签名、返回格式和异常约定。真实 mPLUG 接入时建议直接按这个示例实现。

### 3.4 Warmup 训练入口

`train_warmup.py` 现在也会默认把 `analysis_dir` 对齐到仓库内的 `implement/analysis`。如需覆盖，可以显式传：

```bash
python train_warmup.py --analysis_dir /custom/analysis/root
```

## 4. 分析资产

### 4.1 时间分解

把 profiling 输出喂给：

```bash
python scripts/export_time_breakdown.py \
  --input analysis/time/step_timing.jsonl \
  --output analysis/time/time_breakdown.json
```

### 4.2 Reward curves

从 wandb CSV 或本地 metrics CSV 导出：

```bash
python scripts/export_reward_curves.py \
  --input analysis/curves/per_unit_rewards.jsonl \
  --output analysis/curves/scalar_reward_curves.csv \
  --label "Flow-GRPO w/o \\method"

python scripts/export_reward_curves.py \
  --input analysis/curves/per_unit_rewards.jsonl \
  --output analysis/curves/suca_reward_curves.csv \
  --label "\\method"
```

## 5. 一次性固定 manifest

如果所有输出都遵守上面的路径，就直接使用 [implement/docs/paper_artifact_manifest.canonical.json](/Volumes/Happy/paper/基于强化学习的T2I增强/implement/docs/paper_artifact_manifest.canonical.json)。之后不需要再改 manifest，只要不断覆盖这些固定结果文件即可。