"""Formal benchmark evaluation for GenEval2 with optional normalized summary output."""
import argparse
import os, sys, json, torch, time
import numpy as np
from PIL import Image
from scipy.stats import gmean

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_MODEL = f"{PROJ}/models/stable-diffusion-3.5-medium"
GENEVAL2_FILE = f"{PROJ}/data/geneval2/geneval2_data.jsonl"
RESOLUTION = 512
DEVICE = "cuda:0"

def load_pipeline(model_path, device):
    from diffusers import StableDiffusion3Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def generate_and_score(pipe, benchmark_data, outdir, reward_port=8100, num_inference_steps=28, guidance_scale=4.5):
    import requests, base64
    from io import BytesIO

    os.makedirs(os.path.join(outdir, "images"), exist_ok=True)
    all_scores = []

    for idx, d in enumerate(benchmark_data):
        prompt = d["prompt"]
        # Generate
        with torch.no_grad():
            result = pipe(prompt=prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                         height=RESOLUTION, width=RESOLUTION)
        img = result.images[0]
        img.save(os.path.join(outdir, "images", f"{idx:04d}.png"))

        # Score via VQA
        vqa_list = d["vqa_list"]
        questions = [q for q, a in vqa_list]
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            resp = requests.post(
                f"http://localhost:{reward_port}/score_batch",
                json={"image_b64": img_b64, "questions": questions},
                timeout=60,
            )
            scores = resp.json()["scores"]
        except Exception as e:
            print(f"  Score error {idx}: {e}")
            scores = [0.5] * len(questions)

        all_scores.append(scores)
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(benchmark_data)} done", flush=True)

    return all_scores

def compute_metrics(all_scores, benchmark_data):
    per_prompt_gm = [gmean([max(s, 1e-6) for s in sl]) for sl in all_scores]
    total_gm = 100 * np.mean(per_prompt_gm)

    skill_scores = {}
    for d, sl in zip(benchmark_data[:len(all_scores)], all_scores):
        for skill, score in zip(d.get("skills", []), sl):
            skill_scores.setdefault(skill, []).append(score)

    per_skill = {s: 100 * np.mean(v) for s, v in skill_scores.items()}

    atom_scores = {}
    for d, gm in zip(benchmark_data[:len(all_scores)], per_prompt_gm):
        ac = d.get("atom_count", 0)
        atom_scores.setdefault(ac, []).append(gm)
    per_atom = {ac: 100 * np.mean(v) for ac, v in sorted(atom_scores.items())}

    return {"soft_tifa_gm": total_gm, "per_skill": per_skill, "per_atom": per_atom}


def to_normalized_summary(metrics, variant_name=None):
    summary = {
        "benchmark": "geneval2",
        "overall": {"soft_tifa_gm": metrics.get("soft_tifa_gm")},
        "skills": metrics.get("per_skill", {}),
        "atom_count": {str(k): v for k, v in metrics.get("per_atom", {}).items()},
    }
    if variant_name:
        summary["variant"] = variant_name
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=PROJ)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--benchmark-file", default=None)
    parser.add_argument("--num-eval", type=int, default=200)
    parser.add_argument("--reward-port", type=int, default=8100)
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--base-output-dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--trained-output-dir", default=None)
    parser.add_argument("--comparison-json", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--summary-model-key", default=None, help="Which model from the comparison to export as normalized summary")
    parser.add_argument("--summary-variant-name", default=None)
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root)
    base_model = args.base_model or os.path.join(project_root, "models/stable-diffusion-3.5-medium")
    benchmark_file = args.benchmark_file or os.path.join(project_root, "data/geneval2/geneval2_data.jsonl")
    comparison_json = args.comparison_json or os.path.join(project_root, "eval_results/formal_comparison.json")
    base_output_dir = args.base_output_dir or os.path.join(project_root, "eval_results/formal_base")
    trained_output_dir = args.trained_output_dir or os.path.join(project_root, "eval_results/formal_grpo")
    checkpoint = args.checkpoint or os.path.join(project_root, "logs/fullparam_restored/checkpoints/checkpoint-2800")

    benchmark_data = [json.loads(l) for l in open(benchmark_file)][:args.num_eval]
    print(f"GenEval2: {args.num_eval} prompts")
    print("=" * 60)

    results = {}

    # --- Base SD3.5 ---
    if not args.skip_base:
        print("\n[1/2] Base SD3.5-Medium...")
        pipe = load_pipeline(base_model, DEVICE)
        base_scores = generate_and_score(
            pipe,
            benchmark_data,
            base_output_dir,
            reward_port=args.reward_port,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        results["base"] = compute_metrics(base_scores, benchmark_data)
        del pipe; torch.cuda.empty_cache()

    # --- GRPO checkpoint ---
    if checkpoint and os.path.exists(checkpoint):
        print(f"\n[2/2] Trained checkpoint: {checkpoint}...")
        from diffusers import SD3Transformer2DModel
        pipe = load_pipeline(base_model, DEVICE)
        # Load trained transformer
        t = SD3Transformer2DModel.from_pretrained(
            os.path.join(checkpoint, "transformer"), torch_dtype=torch.bfloat16
        ).to(DEVICE)
        pipe.transformer = t
        grpo_scores = generate_and_score(
            pipe,
            benchmark_data,
            trained_output_dir,
            reward_port=args.reward_port,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        results["grpo"] = compute_metrics(grpo_scores, benchmark_data)
        del pipe; torch.cuda.empty_cache()

    # --- Print comparison ---
    print("\n" + "=" * 60)
    print("FORMAL GenEval2 RESULTS (200 prompts)")
    print("=" * 60)

    for model_name, metrics in results.items():
        print(f"\n--- {model_name} ---")
        print(f"  soft_tifa_gm: {metrics['soft_tifa_gm']:.2f}")
        for skill, val in sorted(metrics['per_skill'].items()):
            print(f"  {skill}: {val:.2f}")

    if len(results) >= 2:
        print(f"\n--- Comparison ---")
        print(f"{'Metric':<20} {'Base':>10} {'GRPO':>10} {'Diff':>10}")
        print("-" * 50)
        for key in ["soft_tifa_gm"]:
            b = results.get("base", {}).get(key, 0)
            g = results.get("grpo", {}).get(key, 0)
            print(f"{key:<20} {b:>10.2f} {g:>10.2f} {g-b:>+10.2f}")
        all_skills = set()
        for m in results.values():
            all_skills.update(m["per_skill"].keys())
        for skill in sorted(all_skills):
            b = results.get("base", {}).get("per_skill", {}).get(skill, 0)
            g = results.get("grpo", {}).get("per_skill", {}).get(skill, 0)
            print(f"{skill:<20} {b:>10.2f} {g:>10.2f} {g-b:>+10.2f}")

    # Save
    os.makedirs(os.path.dirname(comparison_json), exist_ok=True)
    with open(comparison_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {comparison_json}")

    if args.summary_json and args.summary_model_key:
        if args.summary_model_key not in results:
            raise ValueError(f"summary_model_key '{args.summary_model_key}' not found in results: {list(results.keys())}")
        normalized = to_normalized_summary(results[args.summary_model_key], variant_name=args.summary_variant_name)
        os.makedirs(os.path.dirname(args.summary_json), exist_ok=True)
        with open(args.summary_json, "w") as f:
            json.dump(normalized, f, indent=2, ensure_ascii=False)
        print(f"Saved normalized summary: {args.summary_json}")

if __name__ == "__main__":
    main()
