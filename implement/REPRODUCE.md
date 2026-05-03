# SUCA — Semantic Unit Credit Assignment for Compositional T2I Generation

## 项目概览

SUCA 在 Flow-GRPO 框架之上，引入语义单元级别的信用分配（credit assignment），让扩散模型的 RL 训练可以精确到每个语义成分（物体、属性、数量、关系）在每个去噪步骤上的贡献。

### 核心创新

```
传统 GRPO:  reward → 均匀分配到所有 timestep
SUCA:       reward → 解析语义单元 → attention-based 责任矩阵 C[k,t] → 加权分配到各 timestep
            + 稀疏过程奖励 → anchor timestep 直接解码打分 → per-timestep 监督
```

### 架构

```
GPU 0-5: SD3.5-M 全参训练 (accelerate 6-GPU data parallel)
GPU 6,7: Qwen3-VL-8B reward server (FastAPI)
```

---

## 1. 环境配置

### 硬件要求
- 8× NVIDIA A800/A100 80GB (SXM)
- 存储: ~50TB (模型 + 数据)
- 内存: ≥256GB

### 软件环境

```bash
python -m venv .venv
source .venv/bin/activate

# 核心包
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install diffusers==0.37.0
pip install transformers==4.57.6
pip install accelerate==1.13.0
pip install peft==0.18.1
pip install wandb==0.25.1

# Flow-GRPO 依赖
pip install ml_collections absl-py

# Reward server
pip install fastapi uvicorn qwen-vl-utils

# 评测
pip install scipy scikit-learn

# Flow-GRPO 本身
cd flow_grpo && pip install -e . && cd ..
```

### 关键包版本

| 包 | 版本 |
|---|------|
| torch | 2.6.0+cu124 |
| diffusers | 0.37.0 |
| transformers | 4.57.6 |
| accelerate | 1.13.0 |
| peft | 0.18.1 |

---

## 2. 模型路径

| 模型 | 路径 | 说明 |
|------|------|------|
| SD3.5-Medium | `models/stable-diffusion-3.5-medium/` | 46GB, HuggingFace 下载 |
| Qwen3-VL-8B | `models/Qwen3-VL-8B-Instruct/` | VLM reward 打分 |
| Qwen2.5-VL-7B | `models/Qwen2.5-VL-7B-Instruct/` | 备选 VLM |

### 下载模型

```bash
# 需要 HuggingFace token (SD3.5 需要同意许可协议)
huggingface-cli login --token YOUR_TOKEN

python -c "
from huggingface_hub import snapshot_download
snapshot_download('stabilityai/stable-diffusion-3.5-medium', local_dir='models/stable-diffusion-3.5-medium')
snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='models/Qwen3-VL-8B-Instruct')
"
```

---

## 3. 数据路径

### 训练数据

| 数据 | 路径 | 数量 | 说明 |
|------|------|------|------|
| T2I-CompBench prompts | `dataset/t2i_compbench/` | 5559 条 | RL 训练用（组合型 prompt） |
| GenEval 训练 prompts | `dataset/geneval/train.txt` | 6000 条 | 早期 RL 训练用 |
| GenEval 测试 prompts | `dataset/geneval/test.txt` | 50 条 | 训练中 eval 用 |
| Fine-T2I warmup | `data/fine-t2i/` | 82K 图文对, 11GB | SFT warmup 用 |

### 测试数据集 (Benchmark)

| 数据集 | 路径 | 数量 | 评测方式 |
|--------|------|------|----------|
| GenEval2 | `data/geneval2/` | 800 prompts | soft_tifa_gm (VQA + 几何平均) |
| SpatialGenEval | `data/spatialgeneval/` | 1230 prompts | 10 道多选题 × 5 次投票 |
| DPG-Bench | `data/dpg-bench/` | ~300 prompts | 依赖传播命题分解 |

---

## 4. 核心代码结构

```
SUGA0316/
├── suca/                          # SUCA 核心模块
│   ├── semantic_parser.py         # 语义单元解析 (规则 + LLM)
│   ├── attention_extractor.py     # Cross-attention 提取
│   ├── responsibility_matrix.py   # C[k,t] 责任矩阵构建
│   ├── unit_reward.py             # 单元级 VQA reward 计算
│   ├── diffusion_policy.py        # SD3.5 策略封装 + log_prob
│   ├── suca_trainer.py            # 早期独立 trainer
│   ├── parallel_trainer.py        # 多 GPU 并行 trainer
│   ├── pipeline_trainer.py        # 流水线 trainer
│   ├── fsdp_pipeline_trainer.py   # FSDP 流水线 trainer
│   └── warmup_dataset.py          # SFT warmup 数据集
│
├── flow_grpo/                     # Flow-GRPO 框架 (fork from yifan123/flow_grpo)
│   ├── scripts/
│   │   └── train_sd3_suca.py      # ★ 主训练脚本 (SUCA + Flow-GRPO 融合)
│   ├── config/
│   │   └── suca.py                # ★ 训练参数配置 (含 sweep 配置)
│   └── flow_grpo/
│       └── rewards.py             # reward 函数 (含 suca_vqa_score)
│
├── scripts/
│   ├── launch_suca_flowgrpo.sh    # ★ 一键启动脚本
│   ├── launch_ablation.sh         # 消融实验启动
│   ├── launch_sweep_r1.sh         # PPO 超参 sweep
│   ├── run_reward_server.py       # Qwen3-VL FastAPI reward server
│   ├── generate_images.py         # 批量图片生成
│   ├── eval_geneval2_shard.py     # GenEval2 分片评测
│   ├── eval_formal.py             # 正式 benchmark 评测
│   ├── eval_spatial_api.py        # SpatialGenEval API 评测
│   ├── eval_benchmark_standalone.py # 独立 benchmark 评测
│   ├── generate_prompts.py        # prompt 生成
│   ├── download_fine_t2i.py       # Fine-T2I 下载
│   └── download_fine_t2i_fast.py  # Fine-T2I 快速下载
│
├── docs/
│   └── experiment_summary.md      # RL 实验总结 (10 轮实验记录)
│
├── config/
│   └── default.yaml               # Hydra 配置 (早期 trainer 用)
│
├── train.py                       # 训练入口
├── train_warmup.py                # SFT warmup 训练脚本
└── evaluate.py                    # 评测脚本
```

---

## 5. 训练流程

### Phase 1: SFT Warmup

```bash
# 4 GPU, 3 epochs, ~3 小时
CUDA_VISIBLE_DEVICES=2,3,6,7 \
.venv/bin/python -m accelerate.commands.launch \
  --num_processes 4 --mixed_precision bf16 --multi_gpu \
  train_warmup.py \
  --model_name models/stable-diffusion-3.5-medium \
  --metadata_file data/fine-t2i/warmup_100k.json \
  --data_root data/fine-t2i \
  --output_dir outputs/warmup \
  --resolution 512 --train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 --max_train_steps 7500 \
  --gradient_checkpointing --mixed_precision bf16
```

**SFT warmup checkpoint**: `outputs/warmup/checkpoint-7000/`

### Phase 2: SUCA + Flow-GRPO RL 训练

```bash
# 一键启动 (8 GPU)
bash scripts/launch_suca_flowgrpo.sh
```

**启动脚本做了什么：**
1. GPU 6,7 启动 Qwen3-VL FastAPI reward server (port 8100, 8101)
2. 等待 health check 通过
3. GPU 0-5 启动 accelerate 6-GPU 全参训练
4. 日志输出到 `outputs/suca_flowgrpo_v2/`

### 当前最佳训练参数 (`flow_grpo/config/suca.py`)

```python
# 模型 — 全参训练 (LoRA 已证实不可行)
pretrained.model = "models/stable-diffusion-3.5-medium"
use_lora = False              # 全参 bf16 (2.24B params)

# 采样
sample.num_steps = 10            # 训练时 10 步去噪
sample.eval_num_steps = 40       # eval 时 40 步
sample.guidance_scale = 4.5
resolution = 512
sample.train_batch_size = 8      # 每 GPU 8 个 prompt
sample.num_image_per_prompt = 8  # 每 prompt 8 张图 (group_size)
sample.num_batches_per_epoch = 4

# 训练
train.learning_rate = 1e-5
train.beta = 0.03                # KL 系数 (sweep 中)
train.clip_range = 0.005         # PPO clip (sweep 中)
train.max_grad_norm = 1.0
train.ema = True
mixed_precision = "bf16"

# 奖励
reward_fn = {"imagereward": 1.0, "suca_vqa": 0.2}
dataset = "dataset/t2i_compbench"  # 组合型 prompt

# SUCA 特有
suca_enabled = True
suca_tau = 0.1                   # 注意力阈值
suca_top_m_spatial = 3           # 空间 attention top-M
suca_attention_layers = [5, 10, 15, 20]
suca_reward_ports = [8100, 8101]

# 稀疏过程奖励 (Exp 9 引入，效果最好)
process_reward = True
process_anchor_steps = [3, 7]    # t=3 (结构), t=7 (属性)
process_lambda = 0.3             # 过程奖励权重
process_confidence_gate = [0.3, 0.7]  # 置信度门控

# 日志
eval_freq = 3                    # 每 3 epoch eval 一次
save_freq = 30                   # 每 30 epoch 存 checkpoint
```

### 可用配置变体

| 配置名 | 说明 |
|--------|------|
| `ablation_no_suca` | 纯 ImageReward Flow-GRPO (无 SUCA) |
| `ablation_with_suca` | SUCA + ImageReward (保守参数) |
| `suca_compositional` | SUCA relaxed (已弃用，clip 太松) |
| `sweep_clip*_beta*` | Round 1: clip × beta 超参 sweep (6 组) |
| `sweep_r2_lam*_lr*` | Round 2: lambda × lr sweep (6 组) |

---

## 6. 实验记录

### 已完成的 RL 实验 (11 轮)

| 实验 | 目录 | 配置要点 | 最新 checkpoint | 结果 |
|------|------|---------|-----------------|------|
| Exp 1 | `logs/suca_sd3/` | LoRA + suca_vqa | checkpoint-480 | 失败：LoRA 参数太少，PPO 无效 |
| Exp 2 | `logs/suca_sd3_lr3e5/` | LoRA 调参 | checkpoint-2000 | 失败：同上 |
| Exp 2b | `logs/suca_sd3_lr3e5_v2/` | LoRA 继续调 | checkpoint-900 | 失败：同上 |
| Exp 3 | `logs/suca_sd3_fullparam/` | 全参 + ImageReward + Fine-T2I | checkpoint-3600 | reward 上升但 eval 先涨后跌 |
| Exp 4 | `logs/fullparam_restored/` | 全参 + ImageReward + CompBench | checkpoint-2800 | 同上 |
| Exp 5-6 | `logs/fullparam_clip_suca_lr3e7/` | SUCA credit + 归一化 | checkpoint-30 | 训练更稳但 eval 仍波动 |
| Exp 7 | `logs/fullparam_suca_restored/` | SUCA relaxed | checkpoint-120 | clip=0%，PPO 失效 |
| Exp 8 | `logs/imagereward_grpo/` | ImageReward baseline | checkpoint-120 | ood=1.11 |
| Exp 9 | `logs/suca_compbench/` | SUCA + 过程奖励 (保守) | checkpoint-best-ood | ood=1.11, id=1.09 |
| Exp 10 | `logs/suca_compositional/` | SUCA + 过程奖励 (relaxed) | checkpoint-240 | **ood=1.73 (最高)** |

### 核心发现

1. **LoRA 不可行** — LoRA 只改 1.28% 参数，SDE log-prob 对小变化不敏感，必须全参训练
2. **训练数据必须是组合型** — Fine-T2I（通用描述）训不出组合能力，必须用 T2I-CompBench
3. **GRPO 组内比较信号弱** — 同 prompt 生成的多张图 reward 差异小，是 eval 不持续涨的根本原因
4. **稀疏过程奖励最有效** — Process reward 绕过 GRPO 弱信号，advantage 区分度提升 15 倍
5. **反复出现 step=12 涨、step=24 回落** — 需要进一步调参或方法改进

详细实验分析见 `docs/experiment_summary.md`。

---

## 7. 评测

### 方法 1: 训练中自动 eval

训练脚本每 3 epoch 自动在 test set 上评测，结果 log 到 wandb：
- `eval_reward/imagereward` — ImageReward 分数
- `eval_reward/suca_vqa` — VQA 分数
- `eval_images` — 生成图片样例

### 方法 2: 独立 Benchmark 评测

```bash
# 生成图片
python scripts/generate_images.py \
  --model_path models/stable-diffusion-3.5-medium \
  --data_dir data \
  --output_dir eval_results/base \
  --resolution 512 --batch_size 4 \
  --benchmarks geneval2,spatialgeneval,dpgbench

# GenEval2 评测
python scripts/eval_geneval2_shard.py \
  --image_dir eval_results/base/geneval2/images \
  --benchmark_file data/geneval2/geneval2_data.jsonl \
  --vlm_model models/Qwen3-VL-8B-Instruct \
  --output_file eval_results/base/geneval2/scores.json

# SpatialGenEval (需要 API key)
python scripts/eval_spatial_api.py \
  --image_dir eval_results/base/spatialgeneval/images \
  --api_key YOUR_KEY --num_samples 50
```

### 已有评测结果 (Base vs SFT)

| 指标 | Base SD3.5 | SFT | 差异 |
|------|-----------|-----|------|
| GenEval2 (soft_tifa_gm) | 16.91 | 21.93 | +5.02 |
| Object | 87.39 | 87.12 | -0.27 |
| Attribute | 67.85 | 72.26 | +4.41 |
| Count | 42.97 | 47.49 | +4.52 |
| Position | 43.58 | 43.04 | -0.54 |
| SpatialGenEval avg_acc | 64.08 | 66.40 | +2.32 |

**注意**: RL checkpoint 尚未进行正式 GenEval2/SpatialGenEval benchmark 评测。训练过程中的 reward 指标显示短期有效（Exp 9 attribute +27%），但需要完整 benchmark 验证。

评测结果详见: `eval_results/evaluation_report.md`

---

## 8. Checkpoint 路径

| 阶段 | 路径 | 说明 |
|------|------|------|
| SFT warmup | `outputs/warmup/checkpoint-7000/` | 7000 步, 3 epoch |
| RL 最佳 (ood) | `logs/suca_compositional/checkpoints/checkpoint-best-ood/` | ood_score=1.73, epoch 6 |
| RL ablation | `logs/imagereward_grpo/checkpoints/checkpoint-best-ood/` | 无 SUCA baseline |
| RL SUCA 保守 | `logs/suca_compbench/checkpoints/checkpoint-best-ood/` | ood=1.11 |

---

## 9. WandB 监控

**项目地址**: `https://wandb.ai/YOUR_ENTITY/flow_grpo`

**关键指标**:

| 指标 | 含义 | 期望趋势 |
|------|------|----------|
| `train_reward/avg_mean` | 训练 reward | ↑ 上升 |
| `eval_reward/imagereward` | 测试 ImageReward | ↑ 上升 |
| `eval_reward/suca_vqa` | 测试 VQA 分数 | ↑ 上升 |
| `suca/advantages_mean` | 优势函数均值 | → 趋近 0 |
| `loss` | 策略 loss | ~ 稳定 |
| `approx_kl` | KL 散度 | ↑ 缓慢上升 (<0.1) |
| `clip_fraction` | PPO clip 比例 | > 0 (=0 说明约束失效) |

---

## 10. 快速复现步骤

```bash
# 1. 克隆项目
git clone YOUR_REPO SUGA0316
cd SUGA0316

# 2. 创建环境
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd flow_grpo && pip install -e . && cd ..

# 3. 下载模型 (需要 HF token)
# ... (见第 2 节)

# 4. 准备数据
# dataset/geneval/, dataset/t2i_compbench/ 已在 repo 中
# data/geneval2/, data/spatialgeneval/, data/dpg-bench/ 需要 git clone

# 5. SFT warmup
# ... (见第 5 节 Phase 1)
# 或直接使用已有 checkpoint: outputs/warmup/checkpoint-7000/

# 6. 启动 RL 训练
export WANDB_API_KEY="YOUR_KEY"
bash scripts/launch_suca_flowgrpo.sh

# 7. 监控
tail -f outputs/suca_flowgrpo_v2/launch.log
# 看 wandb dashboard
```

---

## 11. 常见问题

### Q: 僵尸 GPU 显存怎么清理？
```bash
for f in /proc/*/fd/*; do
  t=$(readlink "$f" 2>/dev/null)
  [[ "$t" == /dev/nvidia* ]] && echo "PID $(echo $f | cut -d/ -f3)"
done 2>/dev/null | sort -u
# 然后 kill -9 PID
```

### Q: NCCL 超时怎么办？
检查是否有 rank 在做非同步操作（如 HTTP 请求）。SUCA 的 credit assignment 只在 rank 0 运行，但用 `dist.broadcast` 同步结果。启动脚本设置了 NCCL 超时 1800 秒。

### Q: 训练 reward 不上升？
1. 检查 `clip_fraction` 是否 > 0（=0 说明 PPO clip 完全失效）
2. 检查 `approx_kl` 是否过大（>0.1 说明偏离太远）
3. 确认训练数据是组合型（T2I-CompBench），不要用 Fine-T2I

### Q: LoRA 可以用吗？
不行。实验证明 LoRA 只改 1.28% 参数，SDE log-prob 对小变化不敏感，PPO 完全无效。必须全参训练。

### Q: eval 先涨后跌怎么办？
这是所有实验的共性问题。最有效的缓解方案是稀疏过程奖励（process reward）。也可以尝试调整 clip/beta（sweep 配置已准备好）。
