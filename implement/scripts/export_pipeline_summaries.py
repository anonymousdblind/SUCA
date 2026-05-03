from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def geometric_mean(values: List[float]) -> float:
    safe = [max(float(v), 1e-6) for v in values if v is not None]
    if not safe:
        return 0.0
    return math.exp(sum(math.log(v) for v in safe) / len(safe))


def build_geneval_summary(score_path: Path, benchmark_path: Path, variant_name: str) -> Dict[str, object]:
    scores = load_json(score_path)
    with benchmark_path.open("r", encoding="utf-8") as handle:
        benchmark_rows = [json.loads(line) for line in handle if line.strip()]

    paired = list(zip(benchmark_rows, scores))
    overall = 100.0 * sum(geometric_mean(row_scores) for _, row_scores in paired) / len(paired)

    skill_buckets: Dict[str, List[float]] = {}
    atom_buckets: Dict[str, List[float]] = {}
    for record, row_scores in paired:
        skills = record.get("skills") or []
        if len(skills) == len(row_scores):
            for skill, score in zip(skills, row_scores):
                skill_buckets.setdefault(str(skill), []).append(float(score) * 100.0)
        atom_count = record.get("atom_count")
        if atom_count is not None:
            atom_buckets.setdefault(str(atom_count), []).append(geometric_mean(row_scores) * 100.0)

    return {
        "benchmark": "geneval2",
        "variant": variant_name,
        "overall": {"soft_tifa_gm": overall},
        "skills": {key: sum(vals) / len(vals) for key, vals in skill_buckets.items() if vals},
        "atom_count": {key: sum(vals) / len(vals) for key, vals in atom_buckets.items() if vals},
    }


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pipeline outputs to canonical summary paths")
    parser.add_argument("--benchmark-data", required=True)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--warmup-dir", required=True)
    parser.add_argument("--rl-dir", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark_data).resolve()
    output_root = Path(args.output_root).resolve()
    variants = {
        "base": Path(args.base_dir).resolve(),
        "sft_warmup": Path(args.warmup_dir).resolve(),
        "suca": Path(args.rl_dir).resolve(),
    }

    geneval_targets = {
        "base": output_root / "geneval2" / "base.json",
        "sft_warmup": output_root / "geneval2" / "sft_warmup.json",
        "suca": output_root / "geneval2" / "suca.json",
    }
    spatial_targets = {
        "base": output_root / "spatialgeneval" / "base.json",
        "sft_warmup": output_root / "spatialgeneval" / "sft_warmup.json",
        "suca": output_root / "spatialgeneval" / "suca.json",
    }
    dpg_targets = {
        "base": output_root / "dpgbench" / "base.json",
        "suca": output_root / "dpgbench" / "suca.json",
    }

    for variant_key, variant_dir in variants.items():
        score_path = variant_dir / "geneval2" / "scores.json"
        if score_path.exists():
            payload = build_geneval_summary(score_path, benchmark_path, variant_key)
            write_json(geneval_targets[variant_key], payload)

        spatial_summary = variant_dir / "spatialgeneval" / "summary.json"
        if spatial_summary.exists():
            copy_if_exists(spatial_summary, spatial_targets[variant_key])

        if variant_key in dpg_targets:
            dpg_summary = variant_dir / "dpgbench" / "summary.json"
            if dpg_summary.exists():
                copy_if_exists(dpg_summary, dpg_targets[variant_key])

    print(f"Exported canonical summaries under {output_root}")


if __name__ == "__main__":
    main()