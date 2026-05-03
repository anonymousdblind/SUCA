# SUCA Evaluation Report

**Model**: Stable Diffusion 3.5 Medium (MMDiT, 2.24B params)
**Last Updated**: 2026-04-04

---

## 1. Base vs SFT Warmup (Formal Benchmark)

**SFT Data**: Fine-T2I (82K image-text pairs, 3 epochs, 7500 steps)
**Resolution**: 512 x 512
**Inference**: 28 steps, CFG = 7.0, seed = 42
**Date**: 2026-03-20

### 1.1 GenEval2 (800 prompts)

**Evaluation Model**: Qwen3-VL-8B-Instruct
**Metric**: Soft-TIFA (VQA soft probability)

#### Overall Scores

| Metric | Base | SFT | Delta |
|--------|------|-----|-------|
| **soft_tifa_gm** | 16.91 | **21.93** | **+5.02** |
| soft_tifa_am | 64.27 | **67.11** | +2.84 |

#### Per-Skill Breakdown

| Skill | Base | SFT | Delta |
|-------|------|-----|-------|
| Object | 87.39 | 87.12 | -0.27 |
| **Attribute** | 67.85 | **72.26** | **+4.41** |
| **Count** | 42.97 | **47.49** | **+4.52** |
| Position | 43.58 | 43.04 | -0.54 |
| Verb | 21.36 | 16.24 | -5.12 |

#### Per-Compositionality (atom_count)

| Atoms | Base | SFT | Delta |
|-------|------|-----|-------|
| 3 | 40.77 | **52.62** | **+11.85** |
| 4 | 33.37 | **46.86** | **+13.49** |
| 5 | 20.81 | **25.95** | +5.14 |
| 6 | 14.89 | **20.16** | +5.27 |
| 7 | 10.05 | 9.03 | -1.02 |
| 8 | 7.35 | **9.18** | +1.83 |
| 9 | 5.13 | **9.20** | +4.07 |
| 10 | 2.94 | 2.47 | -0.47 |

**Analysis**: SFT warmup significantly improves attribute binding (+4.41) and counting (+4.52). The gains are most prominent at lower compositionality levels (atoms 3-6), suggesting the model better handles simpler compositional prompts after warmup. Object detection remains stable (~87%), while verb understanding slightly degrades (-5.12), likely due to limited action-oriented data in Fine-T2I.

### 1.2 SpatialGenEval (50 samples, Qwen2.5-VL-72B API)

**Evaluation Model**: Qwen2.5-VL-72B-Instruct (DashScope API)
**Protocol**: 5 rollouts per image, majority vote (threshold=4)

#### Overall Scores

| Metric | Base (n=49) | SFT (n=50) | Delta |
|--------|-------------|------------|-------|
| **avg_acc** | 64.08 | **66.40** | **+2.32** |
| basic_acc | 70.41 | **77.00** | **+6.59** |
| spatial_acc | 62.50 | **63.75** | +1.25 |

#### Per-Dimension Breakdown

| Dimension | Base | SFT | Delta |
|-----------|------|-----|-------|
| **Object** | 65.3 | **74.0** | **+8.7** |
| **Attribute** | 75.5 | **80.0** | **+4.5** |
| Position | 57.1 | **62.0** | +4.9 |
| Orientation | 59.2 | 54.0 | -5.2 |
| Layout | 65.3 | **70.0** | +4.7 |
| Comparison | 63.3 | **64.0** | +0.7 |
| Proximity | 67.3 | **70.0** | +2.7 |
| **Occlusion** | 38.8 | **46.0** | **+7.2** |
| Motion | 69.4 | 66.0 | -3.4 |
| Causal | 79.6 | 78.0 | -1.6 |

**Analysis**: SFT improves basic understanding (Object +8.7, Attribute +4.5) and several spatial dimensions (Position +4.9, Layout +4.7, Occlusion +7.2). Orientation and Motion slightly degrade. The improvement in basic_acc (+6.59) is much stronger than spatial_acc (+1.25), indicating warmup primarily strengthens semantic accuracy rather than spatial reasoning.

### 1.3 DPG-Bench

**Status**: Images generated (1065 per model). Evaluation requires mPLUG VQA model (not yet configured).
**Images**: `eval_results/{base,sft}/dpgbench/images/`

---

## 2. RL Training Results (Training Metrics)

所有 RL 实验基于 SFT warmup checkpoint 开始。以下为训练过程中的 reward 指标（非正式 benchmark）。

### 2.1 RL 实验 Best-OOD 分数对比

| 实验 | 配置 | OOD Score | ID Score | Best Epoch |
|------|------|-----------|----------|------------|
| imagereward_grpo | ImageReward only (no SUCA) | 1.111 | 1.076 | 3 |
| suca_compbench | SUCA + ImageReward (保守) | 1.111 | 1.086 | 3 |
| **suca_compositional** | **SUCA + 过程奖励 (relaxed)** | **1.727** | **1.035** | **6** |
| fullparam_suca_restored | SUCA relaxed (早期) | 0.736 | 0.490 | 3 |

### 2.2 最佳实验详情 (Exp 9: 稀疏过程奖励)

**配置**: 全参 bf16, ImageReward + suca_vqa margin, process reward at t=3,7
**关键指标 (step=12)**:
- ImageReward: **1.089** (所有实验最高)
- Attribute skill: **1.191** (+27% vs baseline)
- Advantage raw_std: **1.85** (之前所有实验 0.07-0.13，**区分度提升 15 倍**)

### 2.3 RL 阶段已知问题

- 所有实验呈现 **step=12 涨、step=24+ 回落** 的模式
- GRPO 组内比较信号弱（同 prompt 生成的图 reward 差异小）
- **尚未对 RL checkpoint 进行正式 GenEval2/SpatialGenEval benchmark 评测**

---

## 3. Summary

### Base vs SFT (Formal Benchmark)

| Benchmark | Metric | Base | SFT | Delta |
|-----------|--------|------|-----|-------|
| **GenEval2** | soft_tifa_gm | 16.91 | **21.93** | **+5.02** |
| **GenEval2** | Attribute | 67.85 | **72.26** | **+4.41** |
| **GenEval2** | Count | 42.97 | **47.49** | **+4.52** |
| **SpatialGenEval** | avg_acc | 64.08 | **66.40** | **+2.32** |
| **SpatialGenEval** | basic_acc | 70.41 | **77.00** | **+6.59** |
| **SpatialGenEval** | spatial_acc | 62.50 | **63.75** | +1.25 |

### RL Training Metrics (Non-benchmark)

| Experiment | Best OOD | Best Attribute | Key Improvement |
|-----------|----------|----------------|-----------------|
| Baseline (no RL) | ~1.0 | ~0.94 | — |
| ImageReward GRPO | 1.11 | — | Global quality |
| **SUCA + Process Reward** | **1.73** | **1.19 (+27%)** | **Per-unit credit + timestep supervision** |

### Key Takeaways

1. **SFT warmup is effective**: Consistent improvements across GenEval2 (+5.02) and SpatialGenEval (+2.32)
2. **Strongest SFT gains in attribute and counting**: Data-driven improvements from Fine-T2I
3. **RL shows promise via training metrics**: Sparse process reward yields best signal quality (15x advantage discrimination)
4. **RL formal benchmark pending**: Need to run GenEval2/SpatialGenEval on best RL checkpoints
5. **No catastrophic forgetting**: Object detection remains stable (~87%)

### Next Steps

- **[Critical]** GenEval2/SpatialGenEval formal benchmark on `logs/suca_compositional/checkpoints/checkpoint-best-ood/`
- PPO hyperparameter sweep (Round 1: clip × beta, 6 configs ready)
- Round 2: lambda × lr sweep (after Round 1 best is known)
- DPG-Bench evaluation with mPLUG
- Full SpatialGenEval (1230 samples) after best RL config confirmed
