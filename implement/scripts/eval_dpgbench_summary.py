"""Aggregate DPG-Bench raw scores into the normalized summary JSON used by the paper builder.

This script is intentionally evaluator-agnostic: any future DPG scorer only needs to
produce one overall score per prompt id. The script then categorizes prompts into
Entity / Attribute / Relation buckets and writes the canonical summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ATTRIBUTE_KEYWORDS = {
    "red", "blue", "green", "yellow", "black", "white", "brown", "orange", "purple", "pink",
    "small", "large", "big", "tiny", "wooden", "metal", "metallic", "glass", "gold", "silver",
    "bright", "dark", "round", "rectangular", "fluffy", "striped", "shiny",
}
RELATION_PATTERNS = [
    "left of", "right of", "above", "below", "inside", "around", "behind", "in front of",
    "on top of", "under", "next to", "between", "beside",
]


def load_scores(path: Path) -> Dict[str, float]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
    elif path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("records", [])

    scores = {}
    for row in rows:
        sample_id = row.get("id") or row.get("prompt_id") or row.get("image_id")
        score = row.get("overall_score")
        if score is None:
            score = row.get("score")
        if score is None:
            score = row.get("dpg_score")
        if sample_id is None or score is None:
            continue
        scores[str(sample_id)] = float(score)
    return scores


def load_prompts(prompt_dir: Path) -> Dict[str, str]:
    prompts = {}
    for file_path in sorted(prompt_dir.glob("*.txt")):
        if file_path.name.startswith("._"):
            continue
        prompts[file_path.stem] = file_path.read_text(encoding="utf-8").strip()
    return prompts


def categorize_prompt(prompt: str) -> str:
    text = prompt.lower()
    if any(pattern in text for pattern in RELATION_PATTERNS):
        return "Relation"
    tokens = set(re.findall(r"[a-z]+", text))
    if tokens & ATTRIBUTE_KEYWORDS:
        return "Attribute"
    return "Entity"


def summarize(scores: Dict[str, float], prompts: Dict[str, str]) -> Dict[str, object]:
    category_scores: Dict[str, List[float]] = defaultdict(list)
    used_scores: List[float] = []
    for sample_id, prompt in prompts.items():
        if sample_id not in scores:
            continue
        score = scores[sample_id]
        used_scores.append(score)
        category_scores[categorize_prompt(prompt)].append(score)

    categories = {
        category: round(sum(values) / len(values), 4)
        for category, values in sorted(category_scores.items()) if values
    }
    overall = round(sum(used_scores) / len(used_scores), 4) if used_scores else None

    return {
        "benchmark": "dpgbench",
        "overall": {"Overall": overall},
        "categories": categories,
        "num_samples": len(used_scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export normalized DPG-Bench summary JSON")
    parser.add_argument("--score-file", required=True, help="CSV/JSON/JSONL with per-sample DPG scores")
    parser.add_argument("--prompt-dir", required=True, help="Directory containing DPG prompt .txt files")
    parser.add_argument("--summary-json", required=True, help="Output normalized summary JSON path")
    parser.add_argument("--variant-name", default=None, help="Optional variant name to store in the summary")
    args = parser.parse_args()

    scores = load_scores(Path(args.score_file).resolve())
    prompts = load_prompts(Path(args.prompt_dir).resolve())
    summary = summarize(scores, prompts)
    if args.variant_name:
        summary["variant"] = args.variant_name

    output_path = Path(args.summary_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote DPG-Bench summary to {output_path}")


if __name__ == "__main__":
    main()