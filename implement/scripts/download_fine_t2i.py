"""
High-concurrency downloader for Fine-T2I dataset.
Uses multiprocessing to parallelize image saving.
"""

import json
import os
import queue
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from datasets import load_dataset
from PIL import Image

OUT_DIR = "/tcci_mnt/liguoyi/project/SUGA0316/data/fine-t2i"
IMG_DIR = os.path.join(OUT_DIR, "images")
TARGET = 100_000
NUM_WORKERS = 32  # concurrent image save threads

os.makedirs(IMG_DIR, exist_ok=True)


def save_image(args):
    """Save and resize a single image."""
    idx, img, meta, txt = args
    img_path = os.path.join(IMG_DIR, f"{idx:06d}.jpg")
    if os.path.exists(img_path):
        # Skip already downloaded
        return {
            "id": idx,
            "image": f"images/{idx:06d}.jpg",
            "prompt": txt,
            "enhanced_prompt": meta.get("enhanced_prompt", ""),
            "category": meta.get("prompt_category", None),
            "style": meta.get("style", None),
        }
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.LANCZOS)
    img.save(img_path, quality=85)
    return {
        "id": idx,
        "image": f"images/{idx:06d}.jpg",
        "prompt": txt,
        "enhanced_prompt": meta.get("enhanced_prompt", ""),
        "category": meta.get("prompt_category", None),
        "style": meta.get("style", None),
    }


def main():
    # Check how many already downloaded
    existing = len([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])
    print(f"Already have {existing} images, targeting {TARGET} total.")

    print("Loading Fine-T2I in streaming mode...")
    ds = load_dataset("ma-xu/fine-t2i", streaming=True, split="train")

    metadata = [None] * TARGET
    completed = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {}
        batch = []

        for i, sample in enumerate(ds):
            if i >= TARGET:
                break

            img = sample["jpg"]
            meta = sample.get("json", {})
            txt = sample.get("txt", "")

            future = executor.submit(save_image, (i, img, meta, txt))
            futures[future] = i

            # Process completed futures periodically
            if len(futures) >= NUM_WORKERS * 2:
                done_futures = [f for f in futures if f.done()]
                for f in done_futures:
                    idx = futures.pop(f)
                    result = f.result()
                    metadata[idx] = result
                    completed += 1

                if completed % 5000 == 0 and completed > 0:
                    print(f"  Progress: {completed}/{TARGET} ({completed*100//TARGET}%)")

        # Wait for remaining futures
        for f in as_completed(futures):
            idx = futures[f]
            result = f.result()
            metadata[idx] = result
            completed += 1

            if completed % 5000 == 0:
                print(f"  Progress: {completed}/{TARGET} ({completed*100//TARGET}%)")

    # Remove None entries (shouldn't happen, but safety)
    metadata = [m for m in metadata if m is not None]
    print(f"\nTotal downloaded: {len(metadata)}")

    # Save metadata
    with open(os.path.join(OUT_DIR, "warmup_100k.json"), "w") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print("Saved warmup_100k.json")

    # RL subset
    import random
    random.seed(42)
    rl_subset = random.sample(metadata, min(12000, len(metadata)))
    rl_prompts = [p["prompt"] for p in rl_subset]

    with open(os.path.join(OUT_DIR, "rl_train_12k.json"), "w") as f:
        json.dump(rl_subset, f, ensure_ascii=False)
    with open("/tcci_mnt/liguoyi/project/SUGA0316/data/prompts.json", "w") as f:
        json.dump(rl_prompts, f, ensure_ascii=False)
    print("Saved rl_train_12k.json and data/prompts.json")
    print("Done!")


if __name__ == "__main__":
    main()
