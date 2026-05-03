#!/bin/bash
# =================================================================
# SUCA Pipeline Training Launcher v2
#
# GPU 0       — Trainer (LoRA backward, single process)
# GPU 1,2,3,4 — 4× Rollout workers (SD3.5 inference HTTP servers)
# GPU 5       — (spare / ref)
# GPU 6,7     — 2× vLLM reward servers (Qwen2.5-VL)
# =================================================================

set -e
cd /tcci_mnt/liguoyi/project/SUGA0316
source .venv/bin/activate

export WANDB_API_KEY="2b9b4e9f586c76970ab77b0aded7fc04c909d288"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/tcci_mnt/liguoyi/project/SUGA0316:$PYTHONPATH

OUTDIR=outputs/rl
mkdir -p $OUTDIR

echo "============================================"
echo " SUCA Pipeline Training v2"
echo " Trainer:GPU0 | Rollout:GPU1-4 | vLLM:GPU6-7"
echo "============================================"

# Kill any leftover processes
pkill -9 -f "run_rollout_worker\|run_reward_server\|vllm.*api_server\|train_single" 2>/dev/null || true
sleep 3

# ----- Step 1: Start 2× vLLM Reward Servers (GPU 6, 7) -----
echo "[1/3] Starting vLLM reward servers..."
VLM=models/Qwen2.5-VL-7B-Instruct

for gpu_port in "6 8100" "7 8101"; do
  set -- $gpu_port
  gpu=$1; port=$2
  CUDA_VISIBLE_DEVICES=$gpu VLLM_USE_V1=0 nohup .venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model $VLM --tensor-parallel-size 1 --port $port \
    --trust-remote-code --max-model-len 4096 --gpu-memory-utilization 0.85 \
    --dtype bfloat16 --disable-log-requests \
    > $OUTDIR/vllm_gpu${gpu}.log 2>&1 &
  echo "  GPU $gpu → port $port (PID $!)"
done

echo "  Waiting for vLLM..."
for port in 8100 8101; do
  for i in $(seq 1 90); do
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
      echo "  Port $port: ready"
      break
    fi
    sleep 5
  done
done

# ----- Step 2: Start 4× Rollout Workers (GPU 1,2,3,4) -----
echo "[2/3] Starting rollout workers..."

base_port=8200
for gpu in 1 2 3 4; do
  port=$((base_port + gpu - 1))
  nohup .venv/bin/python -u scripts/run_rollout_worker.py \
    --gpu $gpu --port $port \
    > $OUTDIR/rollout_gpu${gpu}.log 2>&1 &
  echo "  GPU $gpu → port $port (PID $!)"
done

echo "  Waiting for rollout workers..."
for port in 8200 8201 8202 8203; do
  for i in $(seq 1 60); do
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
      echo "  Port $port: ready"
      break
    fi
    sleep 5
  done
done

# ----- Step 3: Launch Trainer (GPU 0) -----
echo "[3/3] Launching trainer on GPU 0..."

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/train_single.py \
  2>&1 | tee $OUTDIR/rl_train.log

echo "Done."
