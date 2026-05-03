#!/bin/bash
# ============================================
# SUCA Ablation: Two 3-GPU experiments, each with own reward server
#   Exp A (no SUCA):  GPU 0,1,2 train + GPU 6 reward (port 8100)
#   Exp B (SUCA):     GPU 3,4,5 train + GPU 7 reward (port 8101)
# ============================================
set -e
cd /tcci_mnt/liguoyi/project/SUGA0316

OUTDIR=outputs/ablation
mkdir -p $OUTDIR

export WANDB_API_KEY="2b9b4e9f586c76970ab77b0aded7fc04c909d288"
export PYTHONPATH="$PWD:$PWD/flow_grpo:$PYTHONPATH"
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

echo "============================================"
echo " SUCA Ablation — Each experiment owns its reward server"
echo " Exp A: GPU 0,1,2 + GPU 6 (port 8100)"
echo " Exp B: GPU 3,4,5 + GPU 7 (port 8101)"
echo "============================================"

# ----- Reward server for Exp A (GPU 6, port 8100) -----
nohup .venv/bin/python -u scripts/run_reward_server.py \
  --model models/Qwen3-VL-8B-Instruct --gpu 6 --port 8100 \
  > $OUTDIR/reward_a.log 2>&1 &
echo "Reward A: GPU 6, port 8100 (PID $!)"

# ----- Reward server for Exp B (GPU 7, port 8101) -----
nohup .venv/bin/python -u scripts/run_reward_server.py \
  --model models/Qwen3-VL-8B-Instruct --gpu 7 --port 8101 \
  > $OUTDIR/reward_b.log 2>&1 &
echo "Reward B: GPU 7, port 8101 (PID $!)"

echo "Waiting for reward servers..."
for port in 8100 8101; do
  for i in $(seq 1 60); do
    if curl -s http://localhost:$port/health > /dev/null 2>&1; then
      echo "  Port $port: ready"
      break
    fi
    sleep 5
  done
done

# ----- Exp A (no SUCA) — GPU 0,1,2, reward on port 8100 only -----
echo "Launching Exp A (no SUCA) on GPU 0,1,2..."
CUDA_VISIBLE_DEVICES=0,1,2 SUCA_REWARD_PORTS=8100 \
nohup .venv/bin/python -m accelerate.commands.launch \
  --num_processes 3 --mixed_precision bf16 --multi_gpu --main_process_port 29500 \
  flow_grpo/scripts/train_sd3_suca.py \
  --config flow_grpo/config/suca.py:ablation_no_suca \
  > $OUTDIR/exp_a_no_suca.log 2>&1 &
echo "  Exp A PID: $!"

sleep 3

# ----- Exp B (SUCA) — GPU 3,4,5, reward on port 8101 only -----
echo "Launching Exp B (SUCA) on GPU 3,4,5..."
CUDA_VISIBLE_DEVICES=3,4,5 SUCA_REWARD_PORTS=8101 \
nohup .venv/bin/python -m accelerate.commands.launch \
  --num_processes 3 --mixed_precision bf16 --multi_gpu --main_process_port 29501 \
  flow_grpo/scripts/train_sd3_suca.py \
  --config flow_grpo/config/suca.py:ablation_with_suca \
  > $OUTDIR/exp_b_with_suca.log 2>&1 &
echo "  Exp B PID: $!"

echo ""
echo "Both running! Each has its own reward server — no contention."
echo "  Exp A: tail -f $OUTDIR/exp_a_no_suca.log"
echo "  Exp B: tail -f $OUTDIR/exp_b_with_suca.log"
echo "  Compare: grep '\[Eval\]' $OUTDIR/exp_a_no_suca.log $OUTDIR/exp_b_with_suca.log"
