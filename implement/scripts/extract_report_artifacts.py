"""Extract normalized benchmark artifacts from the markdown evaluation report.

This script converts the human-readable report in eval_results/evaluation_report.md
into JSON summaries that can be consumed by build_paper_artifacts.py.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def clean_cell(text: str) -> str:
    text = text.strip()
    text = text.replace("**", "")
    text = text.replace("`", "")
    return text.strip()


def parse_float(text: str) -> float:
    cleaned = clean_cell(text).replace("+", "")
    cleaned = cleaned.replace("%", "")
    return float(cleaned)


def section_lines(lines: List[str], section_heading: str) -> List[str]:
    start = None
    for index, line in enumerate(lines):
        if line.strip() == section_heading:
            start = index + 1
            break
    if start is None:
        raise ValueError(f"Section not found: {section_heading}")

    end = len(lines)
    for index in range(start, len(lines)):
        if lines[index].startswith("### "):
            end = index
            break
    return lines[start:end]


def extract_table(lines: List[str], heading: str) -> List[Dict[str, str]]:
    start = None
    for index, line in enumerate(lines):
        if line.strip() == heading:
            start = index + 1
            break
    if start is None:
        raise ValueError(f"Heading not found: {heading}")

    table_lines: List[str] = []
    in_table = False
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
            in_table = True
            continue
        if in_table:
            break

    if len(table_lines) < 2:
        raise ValueError(f"No markdown table found after heading: {heading}")

    headers = [clean_cell(cell) for cell in table_lines[0].strip("|").split("|")]
    rows: List[Dict[str, str]] = []
    for row_line in table_lines[2:]:
        values = [clean_cell(cell) for cell in row_line.strip("|").split("|")]
        if len(values) != len(headers):
            continue
        rows.append(dict(zip(headers, values)))
    return rows


def build_geneval_summary(report_lines: List[str], variant_key: str) -> Dict[str, object]:
    lines = section_lines(report_lines, "### 1.1 GenEval2 (800 prompts)")
    overall_rows = extract_table(lines, "#### Overall Scores")
    skill_rows = extract_table(lines, "#### Per-Skill Breakdown")
    atom_rows = extract_table(lines, "#### Per-Compositionality (atom_count)")

    metric_column = "Base" if variant_key == "base" else "SFT"
    overall = {row["Metric"]: parse_float(row[metric_column]) for row in overall_rows}
    skills = {row["Skill"]: parse_float(row[metric_column]) for row in skill_rows}
    atom_count = {row["Atoms"]: parse_float(row[metric_column]) for row in atom_rows}

    return {
        "benchmark": "geneval2",
        "variant": variant_key,
        "overall": overall,
        "skills": skills,
        "atom_count": atom_count,
    }


def build_spatial_summary(report_lines: List[str], variant_key: str) -> Dict[str, object]:
    lines = section_lines(report_lines, "### 1.2 SpatialGenEval (50 samples, Qwen2.5-VL-72B API)")
    overall_rows = extract_table(lines, "#### Overall Scores")
    dimension_rows = extract_table(lines, "#### Per-Dimension Breakdown")

    overall_metric_column = "Base (n=49)" if variant_key == "base" else "SFT (n=50)"
    dimension_metric_column = "Base" if variant_key == "base" else "SFT"
    overall = {row["Metric"]: parse_float(row[overall_metric_column]) for row in overall_rows}
    dimensions = {row["Dimension"]: parse_float(row[dimension_metric_column]) for row in dimension_rows}

    return {
        "benchmark": "spatialgeneval",
        "variant": variant_key,
        "overall": overall,
        "dimensions": dimensions,
    }


def extract_rl_diagnostics(report_text: str) -> Dict[str, object]:
    imagereward_match = re.search(r"ImageReward:\s*\*\*(\d+\.\d+)\*\*", report_text)
    attribute_match = re.search(r"Attribute skill:\s*\*\*(\d+\.\d+)\*\*\s*\(\+(\d+)%", report_text)
    advantage_match = re.search(r"Advantage raw_std:\s*\*\*(\d+\.\d+)\*\*.*?提升\s*(\d+)\s*倍", report_text, re.S)

    if not imagereward_match or not attribute_match or not advantage_match:
        raise ValueError("Failed to parse RL diagnostic metrics from evaluation report")

    return {
        "best_rl_run": {
            "label": "SUCA + sparse process reward",
            "step": 12,
            "imagereward": float(imagereward_match.group(1)),
            "attribute_skill": float(attribute_match.group(1)),
            "attribute_reward_change_pct": float(attribute_match.group(2)),
            "advantage_raw_std": float(advantage_match.group(1)),
            "advantage_separability_x": float(advantage_match.group(2)),
        }
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract normalized artifacts from evaluation_report.md")
    parser.add_argument(
        "--report",
        default="eval_results/evaluation_report.md",
        help="Path to the markdown evaluation report",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results/normalized",
        help="Directory to store normalized JSON outputs",
    )
    args = parser.parse_args()

    report_path = Path(args.report).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report_text = report_path.read_text(encoding="utf-8")
    lines = report_text.splitlines()

    geneval_base = build_geneval_summary(lines, "base")
    geneval_sft = build_geneval_summary(lines, "sft")
    spatial_base = build_spatial_summary(lines, "base")
    spatial_sft = build_spatial_summary(lines, "sft")
    rl_diagnostics = extract_rl_diagnostics(report_text)

    outputs = {
        "base_geneval2.json": geneval_base,
        "sft_geneval2.json": geneval_sft,
        "base_spatialgeneval.json": spatial_base,
        "sft_spatialgeneval.json": spatial_sft,
        "training_diagnostics.json": rl_diagnostics,
    }

    for file_name, payload in outputs.items():
        (output_dir / file_name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {len(outputs)} normalized artifact files to {output_dir}")


if __name__ == "__main__":
    main()