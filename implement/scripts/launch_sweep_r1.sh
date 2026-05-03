#!/bin/bash
# =================================================================
# Exp 11: PPO Tightening Sweep — Round 1
#
# Sweep: clip ∈ {0.003, 0.005, 0.008} × beta ∈ {0.03, 0.05}
# Fixed: group_size=8, process_reward on, lambda=0.3, lr=1e-5
#
# Usage:
#   bash scripts/launch_sweep_r1.sh <config_name>
#   e.g. bash scripts/launch_sweep_r1.sh sweep_clip005_beta003
#
# Or run all 6 sequentially (one at a time on same GPUs):
#   bash scripts/launch_sweep_r1.sh ALL
# =================================================================

set -e
cd /tcci_mnt/liguoyi/project/SUGA0316
source .venv/bin/activate

export WANDB_API_KEY="2b9b4e9f586c76970ab77b0aded7fc04c909d288"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_TIMEOUT=1800
export PYTHONPATH=/tcci_mnt/liguoyi/project/SUGA0316:/tcci_mnt/liguoyi/project/SUGA0316/flow_grpo:$PYTHONPATH
export LD_LIBRARY_PATH=$PWD/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib:$LD_LIBRARY_PATH

ALL_CONFIGS=(
    sweep_clip003_beta003
    sweep_clip003_beta005
    sweep_clip005_beta003
    sweep_clip005_beta005
    sweep_clip008_beta003
    sweep_clip008_beta005
)

run_one() {
    local cfg=$1
    local OUTDIR=outputs/sweep_r1/${cfg}
    mkdir -p $OUTDIR

    echo "============================================"
    echo " Sweep R1: $cfg"
    echo " Train: GPU 0-5 | Reward: GPU 6,7"
    echo "============================================"

    # Ensure reward servers are running
    if ! curl -s http://localhost:8100/health > /dev/null 2>&1; then
        echo "[!] Reward servers not running. Start them first:"
        echo "    bash scripts/launch_reward_servers.sh"
        echo "    Or use launch_suca_flowgrpo.sh to start everything."
        exit 1
    fi

    echo "Starting training: $cfg ..."
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 .venv/bin/python -m accelerate.commands.launch \
        --num_processes 6 \
        --mixed_precision bf16 \
        --multi_gpu \
        flow_grpo/scripts/train_sd3_suca.py \
        --config "flow_grpo/config/suca.py:${cfg}" \
        2>&1 | tee $OUTDIR/train.log

    echo "Done: $cfg"
    echo ""
}

CONFIG=${1:-""}

if [ "$CONFIG" = "ALL" ]; then
    echo "Running all 6 sweep configs sequentially..."
    for cfg in "${ALL_CONFIGS[@]}"; do
        run_one "$cfg"
    done
    echo "All sweep runs complete."
elif [ -n "$CONFIG" ]; then
    run_one "$CONFIG"
else
    echo "Usage:"
    echo "  bash scripts/launch_sweep_r1.sh <config_name>"
    echo "  bash scripts/launch_sweep_r1.sh ALL"
    echo ""
    echo "Available configs:"
    for cfg in "${ALL_CONFIGS[@]}"; do
        echo "  $cfg"
    done
fi
