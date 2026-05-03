#!/bin/bash
# SUCA Training Script
# Usage: bash scripts/run_train.sh

set -e

cd "$(dirname "$0")/.."

# Activate virtual environment
source .venv/bin/activate

# Install diffusers if not present
pip install diffusers accelerate -q 2>/dev/null || true

# Generate prompts if not exist
if [ ! -f data/prompts.json ]; then
    echo "Generating training prompts..."
    python scripts/generate_prompts.py
fi

# Run training
echo "Starting SUCA training..."
python train.py \
    model.pretrained_model_name="models/stable-diffusion-2-1" \
    model.vlm_model_name="models/Qwen3-VL-8B-Instruct" \
    diffusion.num_inference_steps=50 \
    training.num_epochs=100 \
    training.learning_rate=1e-6 \
    training.samples_per_prompt=4 \
    training.num_uncertainty_samples=4 \
    suca.tau=0.1 \
    suca.lambda_u=0.5 \
    logging.output_dir=outputs \
    "$@"
