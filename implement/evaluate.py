"""
SUCA Evaluation Script.

Evaluates a trained diffusion model on compositional prompt benchmarks.
Computes per-unit and overall semantic alignment metrics.

Usage:
    python evaluate.py --checkpoint outputs/checkpoint_step1000 --prompts data/prompts.json
"""

import argparse
import json
import logging
import os
import sys

import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate(args):
    from diffusers import StableDiffusion3Pipeline

    from suca.semantic_parser import SemanticUnitParser
    from suca.unit_reward import UnitRewardComputer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    logger.info(f"Loading model from {args.base_model}")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
    ).to(device)

    # Load fine-tuned transformer if checkpoint provided
    transformer_path = os.path.join(args.checkpoint, "transformer")
    if os.path.exists(transformer_path):
        from diffusers import SD3Transformer2DModel
        pipeline.transformer = SD3Transformer2DModel.from_pretrained(
            transformer_path, torch_dtype=torch.float16
        ).to(device)
        logger.info("Loaded fine-tuned transformer.")

    # Load prompts
    with open(args.prompts, "r") as f:
        prompts = json.load(f)
    prompts = prompts[:args.num_prompts]
    logger.info(f"Evaluating on {len(prompts)} prompts.")

    # Setup components
    parser = SemanticUnitParser(use_rules_only=True)
    reward_computer = UnitRewardComputer(
        vlm_model_name=args.vlm_model,
        device=device,
    )

    # Evaluation
    results = []
    all_unit_rewards = []
    type_rewards = {"entity": [], "attribute": [], "count": [], "relation": []}

    out_dir = os.path.join(args.output_dir, "eval_results")
    os.makedirs(out_dir, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        logger.info(f"[{idx+1}/{len(prompts)}] {prompt[:60]}...")

        units = parser.parse(prompt)
        vqa_questions = [u.vqa_question for u in units]

        # Generate image
        with torch.no_grad():
            output = pipeline(
                prompt,
                num_inference_steps=50,
                guidance_scale=7.0,
            )
            image = output.images[0]

        # Compute rewards
        rewards = reward_computer.compute_unit_rewards(image, vqa_questions)

        # Per-unit results
        prompt_result = {
            "prompt": prompt,
            "overall_score": rewards.mean().item(),
            "units": [],
        }
        for i, unit in enumerate(units):
            r = rewards[i].item()
            all_unit_rewards.append(r)
            type_rewards[unit.unit_type.value].append(r)
            prompt_result["units"].append({
                "type": unit.unit_type.value,
                "description": unit.description,
                "question": unit.vqa_question,
                "reward": r,
            })
        results.append(prompt_result)

        # Save image
        if idx < args.num_save_images:
            image.save(os.path.join(out_dir, f"img_{idx:04d}.png"))

    # Aggregate metrics
    overall = sum(all_unit_rewards) / max(len(all_unit_rewards), 1)
    type_means = {
        k: sum(v) / max(len(v), 1) for k, v in type_rewards.items() if v
    }

    summary = {
        "num_prompts": len(prompts),
        "num_units_total": len(all_unit_rewards),
        "overall_unit_score": overall,
        "scores_by_type": type_means,
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Overall Unit Score: {overall:.4f}")
    for k, v in type_means.items():
        logger.info(f"  {k:12s}: {v:.4f}")
    logger.info(f"{'='*50}")

    with open(os.path.join(out_dir, "eval_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "eval_details.json"), "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUCA Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="models/stable-diffusion-3.5-medium")
    parser.add_argument("--vlm_model", type=str, default="models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--prompts", type=str, default="data/prompts.json")
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--num_save_images", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args(sys.argv[1:])
    evaluate(args)
