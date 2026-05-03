"""Single-workflow DPG-Bench evaluation shell.

This script orchestrates:
1. iterate over prompt/image pairs
2. run a backend scoring command per prompt
3. save per-prompt raw scores
4. aggregate to canonical summary JSON for the paper builder

The actual mPLUG scorer is intentionally injected via a command template so the
workflow remains usable before the final evaluator environment is installed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List


def load_prompts(prompt_dir: Path) -> Dict[str, str]:
    prompts = {}
    for file_path in sorted(prompt_dir.glob("*.txt")):
        if file_path.name.startswith("._"):
            continue
        prompts[file_path.stem] = file_path.read_text(encoding="utf-8").strip()
    return prompts


def run_external_score(
    command_template: str,
    sample_id: str,
    prompt: str,
    image_path: Path,
    backend_extra_json: str | None = None,
) -> float:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_json = Path(tmp_dir) / f"{sample_id}.json"
        prompt_escaped = prompt.replace("{", "{{").replace("}", "}}")
        backend_extra_json = backend_extra_json or "{}"
        command = command_template.format(
            id=sample_id,
            prompt=prompt_escaped,
            image=str(image_path),
            output=str(tmp_json),
            prompt_quoted=shlex.quote(prompt),
            image_quoted=shlex.quote(str(image_path)),
            output_quoted=shlex.quote(str(tmp_json)),
            backend_extra_json=backend_extra_json,
            backend_extra_json_quoted=shlex.quote(backend_extra_json),
        )
        subprocess.run(command, shell=True, check=True)
        payload = json.loads(tmp_json.read_text(encoding="utf-8"))
        for key in ["overall_score", "score", "dpg_score"]:
            if key in payload:
                return float(payload[key])
        raise ValueError(f"Scorer output missing score field: {tmp_json}")


def write_raw_scores(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "score", "prompt", "image_path"])
        writer.writeheader()
        writer.writerows(rows)


def summarize_dpg(rows: List[Dict[str, object]], prompt_dir: Path, summary_path: Path, variant_name: str | None) -> None:
    from eval_dpgbench_summary import summarize, load_prompts

    prompts = load_prompts(prompt_dir)
    scores = {str(row["id"]): float(row["score"]) for row in rows}
    summary = summarize(scores, prompts)
    if variant_name:
        summary["variant"] = variant_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Formal DPG-Bench workflow")
    parser.add_argument("--image-dir", required=True, help="Directory with generated DPG-Bench images")
    parser.add_argument("--prompt-dir", required=True, help="Directory with DPG-Bench prompt txt files")
    parser.add_argument("--raw-score-csv", required=True, help="Output CSV for per-prompt scores")
    parser.add_argument("--summary-json", required=True, help="Output normalized summary JSON")
    parser.add_argument("--variant-name", default=None, help="Optional variant name written into the summary JSON")
    parser.add_argument(
        "--score-command-template",
        default=None,
        help="External scorer command template. Available placeholders: {id}, {image}, {output}. The command must write a JSON file with overall_score/score/dpg_score.",
    )
    parser.add_argument(
        "--reuse-raw-score-csv",
        action="store_true",
        help="Skip scoring and only regenerate summary JSON from an existing raw-score CSV",
    )
    parser.add_argument(
        "--backend-extra-json",
        default=None,
        help="Optional JSON object string passed through to the scoring command template via {backend_extra_json} or {backend_extra_json_quoted}.",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir).resolve()
    prompt_dir = Path(args.prompt_dir).resolve()
    raw_score_csv = Path(args.raw_score_csv).resolve()
    summary_json = Path(args.summary_json).resolve()

    rows: List[Dict[str, object]] = []
    if args.reuse_raw_score_csv:
        with raw_score_csv.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        if not args.score_command_template:
            raise ValueError("score-command-template is required unless --reuse-raw-score-csv is set")
        prompts = load_prompts(prompt_dir)
        for sample_id, prompt in prompts.items():
            image_path = image_dir / f"{sample_id}.png"
            if not image_path.exists():
                continue
            score = run_external_score(
                args.score_command_template,
                sample_id,
                prompt,
                image_path,
                backend_extra_json=args.backend_extra_json,
            )
            rows.append({
                "id": sample_id,
                "score": score,
                "prompt": prompt,
                "image_path": str(image_path),
            })
        write_raw_scores(rows, raw_score_csv)

    summarize_dpg(rows, prompt_dir, summary_json, args.variant_name)
    print(f"Wrote raw scores to {raw_score_csv}")
    print(f"Wrote summary JSON to {summary_json}")


if __name__ == "__main__":
    main()