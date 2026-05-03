# SUCA 实验总结

## 基线

| 模型 | GenEval2 soft_tifa_gm | SpatialGenEval avg_acc |
|------|----------------------|----------------------|
| SD3.5 Base | 16.91 | 64.08 |
| SD3.5 + SFT warmup (7000 steps) | 21.93 | 66.40 |

所有 RL 实验都基于 SFT warmup checkpoint 开始。

---

## 实验一览

### Exp 1: LoRA + suca_vqa
- **配置**: LoRA, lr=1e-5, suca_vqa reward
- **结果**: clip=0%（PPO 完全无效），LoRA 参数太少，SDE log-prob 对小参数变化不敏感
- **结论**: LoRA + Flow-GRPO 不可行，放弃

### Exp 2: 全参 + suca_vqa
- **配置**: 全参 bf16, lr=1e-5/1e-6/3e-7, suca_vqa reward, Fine-T2I 训练数据
- **结果**: reward 从 0.58 持续降到 0.45（3900 steps），eval 退化
- **问题**: suca_vqa 信号太弱（分数集中在 0.45-0.55），训练数据（Fine-T2I 通用描述）与评测（GenEval2 组合生成）不对齐
- **结论**: 奖励信号不够 + 数据分布不匹配

### Exp 3: 全参 + ImageReward + Fine-T2I 数据
- **配置**: 全参 bf16, lr=1e-6, ImageReward reward, Fine-T2I 6000 条训练数据
- **结果**: 训练 reward 持续上升（0.77→1.17），Eval-OOD 从 1.10 涨到 1.11
- **问题**: Eval-ID（组合型）先涨后回落（0.99→1.08→0.99），attribute 技能持续恶化（0.86→0.58）
- **结论**: ImageReward 只优化整体质量，不关注组合准确性

### Exp 4: 全参 + ImageReward + T2I-CompBench 数据
- **改动**: 训练数据从 Fine-T2I 换成 T2I-CompBench（5559 条组合型 prompt）
- **配置**: 全参 bf16, lr=1e-6, ImageReward, T2I-CompBench
- **结果**: 跟 Exp 3 同样模式——step=12 涨（1.08），之后回落到 baseline
- **结论**: 换组合数据没有解决问题，ImageReward 不区分语义单元

### Exp 5: 全参 + ImageReward + suca_vqa + SUCA (保守配置)
- **改动**: 开启 SUCA credit assignment，suca_vqa weight=0.0（只提供 per-unit scores 给 SUCA）
- **配置**: lr=1e-6, clip=5e-4, beta=0.1, max_grad_norm=0.1
- **结果**: SUCA per-unit 打分在工作（192/192 样本都有 per-unit scores），但 advantage 太大（1.8-2.7），grad 被截断
- **问题**: SUCA advantage 没有归一化（只除了 std 没减 mean）
- **结论**: 需要对 SUCA advantage 做 zero-mean 归一化

### Exp 6: Exp 5 + advantage 归一化
- **改动**: SUCA advantage 加上 `(adv - mean) / std` 归一化
- **结果**: grad 从 0.15-0.23 降到 0.05-0.07，训练更稳定。但 eval 仍然波动（step=12 涨到 1.08，step=24 回落）
- **结论**: 归一化改善了训练稳定性，但没解决 eval 不持续涨的问题

### Exp 7: SUCA relaxed 配置
- **改动**: lr 10x (1e-5), clip 40x (0.02), grad_norm 10x (1.0), beta 1/10 (0.01)
- **结果**: 跑了 65 epoch，clip≈0%（PPO 约束完全失效），eval 全程横盘（0.99 附近）
- **问题**: 太宽松，模型自由漂移但没有向好的方向走
- **结论**: relaxed 配置不行

### Exp 8: suca_vqa 改为 log-margin reward
- **改动**: reward server 返回 log P(Yes) - log P(No) 作为 per-unit score，区分度从 0.04 提升到 40x
- **配置**: ImageReward weight=1.0 + suca_vqa margin weight=0.2, relaxed 训练参数
- **结果**: SUCA raw_std 从 0.07-0.13 提升到 1.2-1.5（区分度大幅增加），但 eval 仍波动
- **问题**: clip=0%，PPO 约束失效；且 GRPO 组内样本 reward 差异仍然小
- **结论**: margin reward 提升了 SUCA 内部区分度，但不解决 GRPO 组内比较弱的根本问题

### Exp 9: 稀疏过程奖励 (Sparse Process Reward) + group_size=8
- **改动**: 在 anchor timestep (t=3, t=7) 解码中间图像，用 VQA 打分语义单元，作为直接的 per-timestep 监督
  - t=3: 只打 entity/count（模糊图能识别大结构）
  - t=7: 打所有类型
  - 置信度门控: raw prob 在 [0.3, 0.7] 区间的不参与梯度
  - 与终端奖励组合: `w_t = w_t_terminal + 0.3 * w_t_process`
- **结果**:
  - Process reward 工作正常: t=3 mean=-5.3（语义单元未出现），t=7 mean=2.1（已可见）
  - advantage raw_std=1.85（之前所有实验 0.07-0.13），**区分度提升 15 倍**
  - step=12: ImageReward 1.089（所有实验最高），attribute 1.191（**+27%**，所有实验最高）
- **问题**: 只跑了 3 epoch，未看到长期趋势
- **结论**: process reward 显著提升了信号质量，短期效果最好

### Exp 10: 稀疏过程奖励 + group_size=16
- **改动**: num_image_per_prompt 从 8 增到 16，增大组内比较范围
- **结果**: 跑了 10 epoch 后 OOM 崩溃。attribute 先涨后跌（1.08→1.17→0.75），跟之前模式一样
- **问题**: OOM (exitcode -9)；group_size=16 没有比 8 更好
- **结论**: group_size 不是瓶颈

---

## 核心发现

### 1. 反复出现的模式
所有实验都呈现 **step=12 涨、step=24 回落** 的模式，不管配置怎么改。说明问题不在单一超参数，而在方法层面。

### 2. GRPO 组内比较信号太弱
同一 prompt 生成的多张图 reward 差异小（模型是接近确定性的），GRPO 的 group-relative advantage 接近 0。这是所有实验 eval 不持续涨的根本原因。

### 3. 稀疏过程奖励是最有效的改进
Process reward 绕过了 GRPO 弱信号问题，直接给 timestep 级别的监督。advantage 区分度提升 15 倍，短期 eval 效果最好（attribute +27%）。

### 4. suca_vqa 打分饱和
suca_vqa 的 per-unit 分数太高（0.8-0.9），区分度不够。改成 log-margin 后有所改善。

### 5. 训练数据必须是组合型
Fine-T2I（通用描述）训不出组合能力。T2I-CompBench（组合型 prompt）与评测分布对齐。

### 6. LoRA 不可行
LoRA 只改 1.28% 参数，SDE log-prob 对小变化不敏感，PPO 完全无效。必须全参训练。

---

## 当前最佳配置

```
模型: SD3.5-Medium 全参 bf16 (2.24B params)
SFT warmup: checkpoint-7000
训练数据: T2I-CompBench (5559 条组合型 prompt)
奖励: ImageReward (weight=1.0) + suca_vqa margin (weight=0.2)
SUCA: 开启, per-unit credit assignment + attention-based C[k,t]
过程奖励: 开启, anchor t=3 (entity/count) + t=7 (all), lambda=0.3
Group size: 8
lr: 1e-5, clip=0.02, beta=0.01, max_grad_norm=1.0
```

---

## 待解决问题

1. **eval 先涨后跌** — process reward 改善了信号质量但需要更长训练验证是否能持续
2. **clip=0%** — relaxed 配置下 PPO 约束失效，可能需要中间值（如 clip=0.005）
3. **OOM** — group_size=16 时 anchor decode 显存不足，需优化或保持 group=8
4. **责任矩阵校准** — C[k,t] 基于 attention proxy 未验证因果准确性（Feature 2 待实现）
