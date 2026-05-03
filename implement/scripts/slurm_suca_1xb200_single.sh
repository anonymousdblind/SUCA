#!/bin/bash
#SBATCH --job-name=suca_1xb200
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --constraint=b200
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=48:00:00
#SBATCH --output=/users/k2693154/SUCA/implement/slurm-%j.out

set -euo pipefail

ROOT=/users/k2693154/SUCA/implement
VENV_ACTIVATE="$ROOT/.venv/bin/activate"
cd "$ROOT"

if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "[ERROR] Missing virtual environment: $VENV_ACTIVATE"
  exit 1
fi

source "$VENV_ACTIVATE"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800
export PYTHONPATH="$ROOT:$ROOT/flow_grpo:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$ROOT/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH:-}"

# Single-GPU fallback: reward server and trainer share GPU 0.
export SUCA_REWARD_PORTS=8101

OUTDIR="$ROOT/outputs/suca_b200_1gpu"
mkdir -p "$OUTDIR"

cleanup() {
  pkill -f "scripts/run_reward_server.py --model" 2>/dev/null || true
}
trap cleanup EXIT

echo "============================================"
echo " SUCA single-B200 Slurm launch"
echo " Train: GPU 0 (single process)"
echo " Reward: GPU 0, port 8101 (colocated)"
echo " Root: $ROOT"
echo "============================================"

if [[ ! -d models/Qwen3-VL-8B-Instruct ]]; then
  echo "[ERROR] Missing reward model: models/Qwen3-VL-8B-Instruct"
  exit 1
fi

if [[ ! -d models/stable-diffusion-3.5-medium ]]; then
  echo "[ERROR] Missing base model: models/stable-diffusion-3.5-medium"
  exit 1
fi

echo "[1/2] Starting reward server on GPU 0..."
python -u scripts/run_reward_server.py \
  --model models/Qwen3-VL-8B-Instruct \
  --gpu 0 \
  --port 8101 \
  > "$OUTDIR/reward_gpu0.log" 2>&1 &

REWARD_PID=$!
echo "$REWARD_PID" > "$OUTDIR/reward.pid"
echo "  Reward PID: $REWARD_PID"

for i in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8101/health >/dev/null 2>&1; then
    echo "  Reward server ready on port 8101"
    break
  fi
  if ! kill -0 "$REWARD_PID" 2>/dev/null; then
    echo "[ERROR] Reward server exited early. Check $OUTDIR/reward_gpu0.log"
    exit 1
  fi
  sleep 5
done

if ! curl -fsS http://127.0.0.1:8101/health >/dev/null 2>&1; then
  echo "[ERROR] Reward server health check timed out. Check $OUTDIR/reward_gpu0.log"
  exit 1
fi

echo "[2/2] Launching single-GPU training on GPU 0..."
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  flow_grpo/scripts/train_sd3_suca.py \
  --config flow_grpo/config/suca.py:suca_sd3_4gpu \
  2>&1 | tee "$OUTDIR/train.log"