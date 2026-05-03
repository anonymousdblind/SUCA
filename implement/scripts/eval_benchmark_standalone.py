"""
Standalone benchmark eval — loads its own VLM on GPU 7.
Does NOT use training reward servers. Zero impact on training.

Usage:
  CUDA_VISIBLE_DEVICES=7 python scripts/eval_benchmark_standalone.py
"""
import os, sys, json, torch, time
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.stats import gmean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = f"{PROJ}/models/stable-diffusion-3.5-medium"
VLM_MODEL = f"{PROJ}/models/Qwen3-VL-8B-Instruct"
GENEVAL2_FILE = f"{PROJ}/data/geneval2/geneval2_data.jsonl"
NUM_EVAL = 100
RESOLUTION = 512
DEVICE = "cuda:0"  # mapped via CUDA_VISIBLE_DEVICES


class StandaloneVLMScorer:
    """Self-contained VLM scorer, no external server needed."""

    def __init__(self, model_path, device="cuda:0"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        print(f"Loading VLM: {model_path} on {device}...")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        ).to(device).eval()
        self.device = device
        print("VLM ready.")

    @torch.no_grad()
    def score_vqa(self, image: Image.Image, question: str) -> float:
        """Ask a yes/no VQA question, return P(yes)."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": f"{question} Answer in one word."},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        ).to(self.device)

        outputs = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        answer = self.processor.decode(new_tokens, skip_special_tokens=True).strip().lower()

        # Convert to probability-like score
        if answer in ["yes", "yeah", "correct", "true", "right"]:
            return 0.9
        elif answer in ["no", "nope", "false", "wrong", "not"]:
            return 0.1
        else:
            return 0.5


def load_pipeline(model_path, lora_path=None, device="cuda:0"):
    from diffusers import StableDiffusion3Pipeline
    print(f"Loading SD3.5 from {model_path}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    if lora_path and os.path.exists(lora_path):
        from peft import PeftModel
        # Check for adapter_config.json
        if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            pipe.transformer = PeftModel.from_pretrained(pipe.transformer, lora_path).to(device)
            pipe.transformer.merge_and_unload()
            print(f"  LoRA merged from {lora_path}")
        else:
            print(f"  WARNING: No adapter_config.json in {lora_path}, using base model")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_and_score(pipe, scorer, prompts, benchmark_data, tag, outdir, device):
    """Generate images and score them."""
    img_dir = os.path.join(outdir, tag, "images")
    os.makedirs(img_dir, exist_ok=True)

    all_scores = []
    t0 = time.time()

    for idx, (prompt, data) in enumerate(zip(prompts, benchmark_data)):
        # Generate
        with torch.no_grad():
            result = pipe(
                prompt=prompt, num_inference_steps=28, guidance_scale=4.5,
                height=RESOLUTION, width=RESOLUTION,
            )
        img = result.images[0]
        img.save(os.path.join(img_dir, f"{idx:04d}.png"))

        # Score each VQA question
        vqa_list = data["vqa_list"]
        scores = []
        for question, expected_answer in vqa_list:
            score = scorer.score_vqa(img, question)
            scores.append(score)
        all_scores.append(scores)

        if (idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (idx + 1) * (len(prompts) - idx - 1)
            print(f"  [{tag}] {idx+1}/{len(prompts)} | {elapsed:.0f}s elapsed | ETA {eta:.0f}s")

    # Save scores
    json.dump(all_scores, open(os.path.join(outdir, tag, "scores.json"), "w"))

    # Compute metrics
    per_prompt_gm = [gmean(s) if min(s) > 0 else 0.0 for s in all_scores]
    total_gm = 100 * np.mean(per_prompt_gm)

    skill_scores = {}
    for d, sl in zip(benchmark_data[:len(all_scores)], all_scores):
        for skill, score in zip(d.get("skills", []), sl):
            skill_scores.setdefault(skill, []).append(score)

    metrics = {"soft_tifa_gm": total_gm}
    for s, v in skill_scores.items():
        metrics[s] = 100 * np.mean(v)

    return metrics


def find_lora_path(ckpt_dir):
    """Find LoRA adapter in checkpoint directory."""
    candidates = [
        ckpt_dir,
        os.path.join(ckpt_dir, "transformer"),
        os.path.join(ckpt_dir, "lora"),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "adapter_config.json")):
            return c
    return None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="SUCA checkpoint path")
    parser.add_argument("--num_eval", type=int, default=100)
    parser.add_argument("--outdir", default=None)
    args = parser.parse_args()

    # Find latest checkpoint if not specified
    ckpt = args.checkpoint
    if ckpt is None:
        ckpt_dirs = sorted(Path(f"{PROJ}/logs/suca_sd3/checkpoints").glob("checkpoint-*"),
                           key=lambda p: int(p.name.split("-")[1]))
        if ckpt_dirs:
            ckpt = str(ckpt_dirs[-1])
        else:
            print("No checkpoint found!")
            return

    outdir = args.outdir or f"{PROJ}/eval_results/benchmark_{Path(ckpt).name}"
    num_eval = args.num_eval

    print("=" * 60)
    print("SUCA Standalone Benchmark Evaluation")
    print(f"  Base:       {BASE_MODEL}")
    print(f"  Checkpoint: {ckpt}")
    print(f"  VLM:        {VLM_MODEL}")
    print(f"  Device:     {DEVICE}")
    print(f"  Prompts:    {num_eval}")
    print(f"  Output:     {outdir}")
    print("=" * 60)

    # Load benchmark
    benchmark_data = [json.loads(l) for l in open(GENEVAL2_FILE)][:num_eval]
    prompts = [d["prompt"] for d in benchmark_data]

    # Load VLM scorer (stays in memory the whole time)
    scorer = StandaloneVLMScorer(VLM_MODEL, device=DEVICE)

    results = {}

    # --- Base model ---
    print("\n" + "=" * 40)
    print("[1/2] BASE MODEL")
    print("=" * 40)
    base_pipe = load_pipeline(BASE_MODEL, device=DEVICE)
    results["base"] = generate_and_score(base_pipe, scorer, prompts, benchmark_data, "base", outdir, DEVICE)
    del base_pipe
    torch.cuda.empty_cache()

    # --- Trained model ---
    print("\n" + "=" * 40)
    print("[2/2] TRAINED MODEL (SUCA)")
    print("=" * 40)
    lora_path = find_lora_path(ckpt)
    trained_pipe = load_pipeline(BASE_MODEL, lora_path=lora_path, device=DEVICE)
    results["trained"] = generate_and_score(trained_pipe, scorer, prompts, benchmark_data, "trained", outdir, DEVICE)
    del trained_pipe
    torch.cuda.empty_cache()

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'Base':>10} {'Trained':>10} {'Diff':>10}")
    print("-" * 55)
    all_keys = sorted(set(list(results["base"].keys()) + list(results["trained"].keys())))
    for key in all_keys:
        b = results["base"].get(key, 0)
        t = results["trained"].get(key, 0)
        d = t - b
        marker = " ↑" if d > 0.5 else " ↓" if d < -0.5 else ""
        print(f"{key:<20} {b:>10.2f} {t:>10.2f} {d:>+10.2f}{marker}")

    # Save
    report = {
        "base_model": BASE_MODEL,
        "checkpoint": ckpt,
        "lora_path": lora_path,
        "num_eval": num_eval,
        "results": results,
    }
    report_path = os.path.join(outdir, "comparison_report.json")
    json.dump(report, open(report_path, "w"), indent=2)
    print(f"\nReport: {report_path}")
    print("Images: {outdir}/base/images/ vs {outdir}/trained/images/")


if __name__ == "__main__":
    main()
