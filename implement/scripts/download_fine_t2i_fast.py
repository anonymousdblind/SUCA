"""
Fast downloader for Fine-T2I: download tar shards in parallel, then extract locally.

Each shard has ~1024 samples. For 100K samples we need ~98 shards.
Shards are named: curated/train-000000.tar, curated/train-000001.tar, ...
"""

import json
import os
import random
import tarfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import hf_hub_download

REPO_ID = "ma-xu/fine-t2i"
OUT_DIR = "/tcci_mnt/liguoyi/project/SUGA0316/data/fine-t2i"
IMG_DIR = os.path.join(OUT_DIR, "images")
SHARD_DIR = os.path.join(OUT_DIR, "shards")
NUM_SHARDS = 98  # ~100K samples (1024 per shard)
NUM_DOWNLOAD_WORKERS = 8
NUM_EXTRACT_WORKERS = 16

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(SHARD_DIR, exist_ok=True)


def download_shard(shard_idx):
    """Download a single tar shard from HuggingFace."""
    filename = f"curated/train-{shard_idx:06d}.tar"
    local_path = os.path.join(SHARD_DIR, f"train-{shard_idx:06d}.tar")
    if os.path.exists(local_path):
        return local_path, shard_idx
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=SHARD_DIR,
        )
        return path, shard_idx
    except Exception as e:
        print(f"  ERROR downloading shard {shard_idx}: {e}")
        return None, shard_idx


def extract_shard(tar_path, shard_idx):
    """Extract images and metadata from a tar shard."""
    if tar_path is None:
        return []

    results = []
    try:
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            # Group by sample key (files share the same base name)
            samples = {}
            for m in members:
                base = os.path.splitext(m.name)[0]
                ext = os.path.splitext(m.name)[1]
                if base not in samples:
                    samples[base] = {}
                samples[base][ext] = m

            for sample_key, files in samples.items():
                if ".jpg" not in files:
                    continue

                # Read image
                jpg_member = files[".jpg"]
                img_data = tar.extractfile(jpg_member).read()

                # Read text
                txt = ""
                if ".txt" in files:
                    txt = tar.extractfile(files[".txt"]).read().decode("utf-8").strip()

                # Read metadata
                meta = {}
                if ".json" in files:
                    meta = json.loads(tar.extractfile(files[".json"]).read().decode("utf-8"))

                # Save image (resize to 1024)
                from PIL import Image
                img = Image.open(io.BytesIO(img_data))
                # Use sample key as filename
                clean_key = sample_key.replace("/", "_")
                img_filename = f"{clean_key}.jpg"
                img_path = os.path.join(IMG_DIR, img_filename)

                if not os.path.exists(img_path):
                    if max(img.size) > 1024:
                        img.thumbnail((1024, 1024), Image.LANCZOS)
                    img.save(img_path, quality=85)

                results.append({
                    "image": f"images/{img_filename}",
                    "prompt": txt,
                    "enhanced_prompt": meta.get("enhanced_prompt", ""),
                    "category": meta.get("prompt_category", None),
                    "style": meta.get("style", None),
                })
    except Exception as e:
        print(f"  ERROR extracting shard {shard_idx}: {e}")

    return results


def main():
    print(f"=== Downloading {NUM_SHARDS} shards ({NUM_SHARDS * 1024} samples) ===")

    # Phase 1: Download shards in parallel
    print(f"\nPhase 1: Downloading shards ({NUM_DOWNLOAD_WORKERS} workers)...")
    shard_paths = {}
    with ThreadPoolExecutor(max_workers=NUM_DOWNLOAD_WORKERS) as executor:
        futures = {executor.submit(download_shard, i): i for i in range(NUM_SHARDS)}
        done = 0
        for f in as_completed(futures):
            path, idx = f.result()
            shard_paths[idx] = path
            done += 1
            if done % 10 == 0:
                print(f"  Downloaded {done}/{NUM_SHARDS} shards")

    successful = sum(1 for v in shard_paths.values() if v is not None)
    print(f"  Downloaded {successful}/{NUM_SHARDS} shards successfully")

    # Phase 2: Extract images in parallel
    print(f"\nPhase 2: Extracting images ({NUM_EXTRACT_WORKERS} workers)...")
    all_metadata = []
    with ThreadPoolExecutor(max_workers=NUM_EXTRACT_WORKERS) as executor:
        futures = {
            executor.submit(extract_shard, shard_paths[i], i): i
            for i in range(NUM_SHARDS)
            if i in shard_paths and shard_paths[i] is not None
        }
        done = 0
        for f in as_completed(futures):
            results = f.result()
            all_metadata.extend(results)
            done += 1
            if done % 10 == 0:
                print(f"  Extracted {done}/{successful} shards ({len(all_metadata)} images)")

    # Add IDs
    for i, m in enumerate(all_metadata):
        m["id"] = i

    print(f"\nTotal images: {len(all_metadata)}")

    # Save metadata
    with open(os.path.join(OUT_DIR, "warmup_100k.json"), "w") as f:
        json.dump(all_metadata, f, ensure_ascii=False)
    print("Saved warmup_100k.json")

    # RL subset
    random.seed(42)
    rl_size = min(12000, len(all_metadata))
    rl_subset = random.sample(all_metadata, rl_size)
    rl_prompts = [p["prompt"] for p in rl_subset]

    with open(os.path.join(OUT_DIR, "rl_train_12k.json"), "w") as f:
        json.dump(rl_subset, f, ensure_ascii=False)
    with open("/tcci_mnt/liguoyi/project/SUGA0316/data/prompts.json", "w") as f:
        json.dump(rl_prompts, f, ensure_ascii=False)
    print("Saved rl_train_12k.json and data/prompts.json")

    # Cleanup: optionally remove tar shards to save space
    # for i in range(NUM_SHARDS):
    #     p = shard_paths.get(i)
    #     if p and os.path.exists(p):
    #         os.remove(p)
    # print("Cleaned up tar shards")

    print("Done!")


if __name__ == "__main__":
    main()
