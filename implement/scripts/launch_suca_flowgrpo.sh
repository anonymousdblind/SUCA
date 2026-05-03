#!/bin/bash
# =================================================================
# SUCA + Flow-GRPO Training Launcher
#
# GPU 0,1,2,3,4,5 — accelerate 6-GPU data-parallel SD3.5 training
# GPU 6,7         — 2× Qwen3-VL FastAPI reward servers
# =================================================================

set -e
cd /tcci_mnt/liguoyi/project/SUGA0316
source .venv/bin/activate

export WANDB_API_KEY="2b9b4e9f586c76970ab77b0aded7fc04c909d288"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800  # 30 min timeout for SUCA credit assignment
export PYTHONPATH=/tcci_mnt/liguoyi/project/SUGA0316:/tcci_mnt/liguoyi/project/SUGA0316/flow_grpo:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

OUTDIR=outputs/suca_flowgrpo_v2
mkdir -p $OUTDIR

echo "============================================"
echo " SUCA + Flow-GRPO Training"
echo " Train: GPU 0-5 (6 cards)"
echo " Reward: GPU 6,7 (Qwen3-VL)"
echo "============================================"

# Kill leftovers
pkill -9 -f "run_reward_server\|train_sd3_suca\|vllm" 2>/dev/null || true
sleep 3

# ----- Step 1: Start 2× Qwen3-VL FastAPI Reward Servers (GPU 6, 7) -----
echo "[1/3] Starting Qwen3-VL reward servers on GPU 6,7..."
VLM=models/Qwen3-VL-8B-Instruct

for gpu_port in "6 8100" "7 8101"; do
  set -- $gpu_port
  gpu=$1; port=$2
  nohup .venv/bin/python -u scripts/run_reward_server.py \
    --model $VLM --gpu $gpu --port $port \
    > $OUTDIR/reward_gpu${gpu}.log 2>&1 &
  echo "  GPU $gpu → port $port (PID $!)"
done

echo "  Waiting for reward servers..."
for port in 8100 8101; do
  for i in $(seq 1 60); do
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
      echo "  Port $port: ready"
      break
    fi
    sleep 5
  done
done

# ----- Step 2: Prepare dataset (6000 training prompts) -----
echo "[2/3] Preparing dataset..."
mkdir -p dataset/geneval
python -c "
import json

# Training data: 6000 RL prompts as jsonl with metadata
rl_prompts = json.load(open('data/prompts.json'))[:6000]
with open('dataset/geneval/train_metadata.jsonl', 'w') as f:
    for p in rl_prompts:
        json.dump({'prompt': p}, f, ensure_ascii=False)
        f.write('\n')
with open('dataset/geneval/train.txt', 'w') as f:
    for p in rl_prompts:
        f.write(p + '\n')

# Test data: first 50 geneval2 prompts
data = [json.loads(l) for l in open('data/geneval2/geneval2_data.jsonl')]
with open('dataset/geneval/test_metadata.jsonl', 'w') as f:
    for d in data[:50]:
        json.dump({'prompt': d['prompt']}, f, ensure_ascii=False)
        f.write('\n')
with open('dataset/geneval/test.txt', 'w') as f:
    for d in data[:50]:
        f.write(d['prompt'] + '\n')

print(f'train: {len(rl_prompts)} prompts, test: 50 prompts')
"

# ----- Step 3: Launch Training (GPU 0-5, 6 cards) -----
echo "[3/3] Launching SUCA + Flow-GRPO training on GPU 0-5..."

nohup bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 .venv/bin/python -m accelerate.commands.launch \
  --num_processes 6 \
  --mixed_precision bf16 \
  --multi_gpu \
  flow_grpo/scripts/train_sd3_suca.py \
  --config flow_grpo/config/suca.py:suca_sd3_8gpu" \
  > $OUTDIR/train.log 2>&1 &

TRAIN_PID=$!
echo "  Training PID: $TRAIN_PID"
echo $TRAIN_PID > $OUTDIR/train.pid
echo "Done. Training running in background."
echo "  Monitor: tail -f $OUTDIR/train.log"
