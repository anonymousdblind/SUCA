#!/bin/bash
# ============================================
# Auto Monitor: Check training every 20 min for 30 hours
# If problems detected, log them for review
# ============================================
cd /tcci_mnt/liguoyi/project/SUGA0316

LOG=outputs/ablation/monitor.log
TRAIN_LOG=outputs/ablation/fullparam_lr3e7.log
CHECKS=90  # 30 hours / 20 min = 90 checks
INTERVAL=1200  # 20 min in seconds

echo "============================================" >> $LOG
echo "Auto Monitor Started: $(date)" >> $LOG
echo "Training log: $TRAIN_LOG" >> $LOG
echo "Checks: $CHECKS × ${INTERVAL}s" >> $LOG
echo "============================================" >> $LOG

for i in $(seq 1 $CHECKS); do
    sleep $INTERVAL

    echo "" >> $LOG
    echo "--- Check $i/90 at $(date) ---" >> $LOG

    # 1. Check if training process is alive
    PROCS=$(ps aux | grep train_sd3 | grep -v grep | wc -l)
    echo "Processes: $PROCS" >> $LOG

    if [ "$PROCS" -eq 0 ]; then
        echo "WARNING: Training process DEAD!" >> $LOG
        # Check for errors
        tail -5 $TRAIN_LOG >> $LOG 2>/dev/null
        echo "ACTION: Training crashed. Needs manual restart." >> $LOG
        continue
    fi

    # 2. Get GPU status
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $LOG 2>/dev/null

    # 3. Get latest training metrics
    LATEST=$(grep -E "\[Epoch|\[Train\]" $TRAIN_LOG 2>/dev/null | tail -5)
    echo "$LATEST" >> $LOG

    # 4. Get latest eval
    EVAL=$(grep "\[Epoch.*reward=" $TRAIN_LOG 2>/dev/null | tail -1)
    if [ -n "$EVAL" ]; then
        # Extract reward value
        REWARD=$(echo "$EVAL" | grep -oP 'reward=\K[0-9.]+')
        echo "Latest reward: $REWARD" >> $LOG
    fi

    # 5. Check for errors
    ERRORS=$(grep -c "Error\|OOM\|NCCL" $TRAIN_LOG 2>/dev/null)
    echo "Error count: $ERRORS" >> $LOG

    # 6. Check grad and clip from latest Train line
    TRAIN_LINE=$(grep "\[Train\]" $TRAIN_LOG 2>/dev/null | tail -1)
    if [ -n "$TRAIN_LINE" ]; then
        GRAD=$(echo "$TRAIN_LINE" | grep -oP 'grad=\K[0-9.]+')
        CLIP=$(echo "$TRAIN_LINE" | grep -oP 'clip=\K[0-9.]+')
        LOSS=$(echo "$TRAIN_LINE" | grep -oP 'loss=\K[-0-9.]+')
        echo "grad=$GRAD clip=$CLIP loss=$LOSS" >> $LOG
    fi

    echo "Status: OK (process alive, check $i/$CHECKS)" >> $LOG
done

echo "" >> $LOG
echo "============================================" >> $LOG
echo "Monitor completed: $(date)" >> $LOG
echo "============================================" >> $LOG
