"""Generate images for all three benchmarks using SD3.5 pipeline."""
import os
import json
import argparse
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusion3Pipeline


def load_geneval2_prompts(data_dir):
    path = os.path.join(data_dir, "geneval2", "geneval2_data.jsonl")
    prompts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            prompts.append({"id": d["prompt"], "prompt": d["prompt"], "benchmark": "geneval2"})
    return prompts


def load_spatialgeneval_prompts(data_dir):
    path = os.path.join(data_dir, "SpatialGenEval", "eval", "SpatialGenEval_T2I_Prompts.jsonl")
    prompts = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            prompts.append({"id": d["id"], "prompt": d["prompt"], "benchmark": "spatialgeneval"})
    return prompts


def load_dpgbench_prompts(data_dir):
    prompt_dir = os.path.join(data_dir, "dpg-bench", "dpg_bench", "prompts")
    prompts = []
    for fname in sorted(os.listdir(prompt_dir)):
        if fname.endswith(".txt"):
            pid = fname.replace(".txt", "")
            with open(os.path.join(prompt_dir, fname)) as f:
                text = f.read().strip()
            prompts.append({"id": pid, "prompt": text, "benchmark": "dpgbench"})
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--transformer_path", type=str, default=None,
                        help="Path to fine-tuned transformer weights (for SFT model)")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--benchmarks", type=str, default="geneval2,spatialgeneval,dpgbench")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pipeline
    print(f"Loading SD3.5 pipeline from {args.model_path}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
    )

    # Load fine-tuned transformer if specified
    if args.transformer_path:
        print(f"Loading fine-tuned transformer from {args.transformer_path}...")
        from diffusers import SD3Transformer2DModel
        transformer = SD3Transformer2DModel.from_pretrained(
            args.transformer_path,
            torch_dtype=torch.float16,
        )
        pipe.transformer = transformer

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    benchmarks = [bench.strip() for bench in args.benchmarks.split(",") if bench.strip()]

    for bench in benchmarks:
        bench_key = bench.lower()
        print(f"\n=== Generating for {bench} ===")
        if bench_key == "geneval2":
            all_prompts = load_geneval2_prompts(args.data_dir)
        elif bench_key == "spatialgeneval":
            all_prompts = load_spatialgeneval_prompts(args.data_dir)
        elif bench_key == "dpgbench":
            all_prompts = load_dpgbench_prompts(args.data_dir)
        else:
            print(f"Unknown benchmark: {bench}")
            continue

        out_dir = os.path.join(args.output_dir, bench, "images")
        os.makedirs(out_dir, exist_ok=True)

        # For geneval2, we also need a prompt->filepath mapping
        geneval2_mapping = {}

        # Check which images already exist and skip them
        remaining = []
        for idx, item in enumerate(all_prompts):
            if bench_key == "geneval2":
                fname = f"{idx:06d}.png"
            else:
                fname = f"{item['id']}.png"
            save_path = os.path.join(out_dir, fname)
            if os.path.exists(save_path):
                if bench_key == "geneval2":
                    geneval2_mapping[item["prompt"]] = save_path
            else:
                remaining.append((idx, item))

        print(f"  Total prompts: {len(all_prompts)}, already done: {len(all_prompts) - len(remaining)}, remaining: {len(remaining)}")

        # Process remaining in batches
        for bi in tqdm(range(0, len(remaining), args.batch_size), desc=bench):
            batch_items = remaining[bi:bi + args.batch_size]
            batch_prompts = [item["prompt"] for _, item in batch_items]
            batch_indices = [idx for idx, _ in batch_items]

            gen = torch.Generator(device=device).manual_seed(args.seed + batch_indices[0])

            images = pipe(
                prompt=batch_prompts,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
                generator=gen,
            ).images

            for j, (img, (idx, item)) in enumerate(zip(images, batch_items)):
                if bench_key == "geneval2":
                    fname = f"{idx:06d}.png"
                    save_path = os.path.join(out_dir, fname)
                    img.save(save_path)
                    geneval2_mapping[item["prompt"]] = save_path
                elif bench_key == "spatialgeneval":
                    fname = f"{item['id']}.png"
                    save_path = os.path.join(out_dir, fname)
                    img.save(save_path)
                elif bench_key == "dpgbench":
                    fname = f"{item['id']}.png"
                    save_path = os.path.join(out_dir, fname)
                    img.save(save_path)

        # Save geneval2 mapping
        if bench_key == "geneval2":
            mapping_path = os.path.join(args.output_dir, bench, "image_paths.json")
            with open(mapping_path, "w") as f:
                json.dump(geneval2_mapping, f, indent=2)
            print(f"  Saved mapping to {mapping_path}")

        print(f"  Generated {len(all_prompts)} images -> {out_dir}")

    print("\nAll generation done!")


if __name__ == "__main__":
    main()
