# SUCA Project — Download Links

All models, datasets, and code repositories needed to reproduce this project.

## Models

| Model | HuggingFace Link | Size | Usage |
|-------|-----------------|------|-------|
| SD3.5-Medium | https://huggingface.co/stabilityai/stable-diffusion-3.5-medium | ~46 GB | Base diffusion model (全参训练) |
| Qwen3-VL-8B-Instruct | https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct | ~17 GB | VLM reward server |
| Qwen2.5-VL-7B-Instruct | https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct | ~15 GB | 备选 VLM |

### Download Commands

```bash
# SD3.5-Medium (requires HF token + license agreement)
huggingface-cli download stabilityai/stable-diffusion-3.5-medium --local-dir models/stable-diffusion-3.5-medium

# Qwen3-VL-8B
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct --local-dir models/Qwen3-VL-8B-Instruct

# Qwen2.5-VL-7B (optional backup)
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct
```

## Training Data

| Dataset | Source | Size | Usage | Local Path |
|---------|--------|------|-------|------------|
| Fine-T2I | https://huggingface.co/datasets/ma-xu/fine-t2i | ~135 GB (82K subset) | SFT warmup (image-text pairs) | `data/fine-t2i/` |
| T2I-CompBench | Derived from T2I-CompBench | 5559 prompts | RL 训练（组合型 prompt） | `dataset/t2i_compbench/` |
| GenEval prompts | Derived from Fine-T2I + custom | 6000 prompts | 早期 RL 训练 | `dataset/geneval/` |

### Download Commands

```bash
# Fine-T2I (streaming download of 100K subset)
python scripts/download_fine_t2i.py
# 或快速下载
python scripts/download_fine_t2i_fast.py

# T2I-CompBench 和 GenEval prompts 已在 repo 的 dataset/ 中
```

## Test Datasets (Benchmarks)

| Dataset | GitHub / Source | Size | Metrics | Local Path |
|---------|----------------|------|---------|------------|
| GenEval2 | https://github.com/djghosh13/geneval | 1.9 MB | soft_tifa_gm (object, attribute, count, position, verb) | `data/geneval2/` |
| SpatialGenEval | https://github.com/SpatialGenEval/SpatialGenEval | 24 MB | avg_acc, spatial_acc (10 spatial dimensions) | `data/spatialgeneval/` |
| DPG-Bench | https://github.com/TencentQQGYLab/ELLA | 33 MB | DPG Score (entity, attribute, relation) | `data/dpg-bench/` |

### Download Commands

```bash
# GenEval2
git clone https://github.com/djghosh13/geneval.git data/geneval2

# SpatialGenEval
git clone https://github.com/SpatialGenEval/SpatialGenEval.git data/spatialgeneval

# DPG-Bench
git clone https://github.com/TencentQQGYLab/ELLA.git data/dpg-bench
```

## Code Repositories

| Repository | Link | Usage |
|------------|------|-------|
| Flow-GRPO | https://github.com/yifan123/flow_grpo | Base RL training framework (SD3 + GRPO) |
| SUCA (this project) | Local | Semantic Unit Credit Assignment (our contribution) |

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install Flow-GRPO
cd flow_grpo && pip install -e . && cd ..
```

## Directory Structure After Download

```
SUGA0316/
  models/
    stable-diffusion-3.5-medium/     # 46 GB
    Qwen3-VL-8B-Instruct/           # 17 GB
    Qwen2.5-VL-7B-Instruct/         # 15 GB (optional)
  data/
    fine-t2i/                        # 135 GB (SFT warmup)
      images/                        # 82K images
      warmup_100k.json              # metadata
    geneval2/                        # 1.9 MB (test)
      geneval2_data.jsonl
    spatialgeneval/                   # 24 MB (test)
    dpg-bench/                       # 33 MB (test)
  dataset/
    t2i_compbench/                   # 5559 条 (RL 训练, 组合型)
    geneval/                         # 6000 条 (早期 RL 训练)
      train_metadata.jsonl
      test_metadata.jsonl
  flow_grpo/                         # Flow-GRPO framework
  suca/                              # SUCA module (our code)
  outputs/
    warmup/checkpoint-7000/          # SFT warmup checkpoint
  logs/                              # RL training checkpoints (11 experiments)
```

## Total Storage Required

| Component | Size |
|-----------|------|
| Models | ~78 GB |
| Training data (Fine-T2I subset) | ~135 GB |
| Test datasets | ~60 MB |
| Code + checkpoints | ~10 GB |
| **Total** | **~225 GB** |
