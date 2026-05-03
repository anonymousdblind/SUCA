from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def compute_geneval_score(score_json_path: Path) -> Optional[float]:
    if not score_json_path.exists():
        return None
    payload = load_json(score_json_path)
    if not payload:
        return None

    per_prompt = []
    for row in payload:
        vals = [max(float(v), 1e-6) for v in row if v is not None]
        if not vals:
            continue
        gm = math.exp(sum(math.log(v) for v in vals) / len(vals))
        per_prompt.append(gm)
    if not per_prompt:
        return None
    return 100.0 * sum(per_prompt) / len(per_prompt)


def load_spatial_score(summary_json_path: Path) -> Optional[float]:
    if not summary_json_path.exists():
        return None
    payload = load_json(summary_json_path)
    overall = payload.get("overall", {})
    value = overall.get("avg_acc")
    return None if value is None else float(value)


def load_dpg_score(summary_json_path: Path) -> Optional[float]:
    if not summary_json_path.exists():
        return None
    payload = load_json(summary_json_path)
    overall = payload.get("overall", {})
    value = overall.get("Overall")
    return None if value is None else float(value)


def fmt(value: Optional[float]) -> str:
    return "--" if value is None else f"{value:.2f}"


def delta_text(value: Optional[float], ref: Optional[float]) -> str:
    if value is None or ref is None:
        return "--"
    return f"{value - ref:+.2f}"


def variant_metrics(variant_dir: Path):
    spatial_summary = variant_dir / "SpatialGenEval" / "summary.json"
    if not spatial_summary.exists():
        spatial_summary = variant_dir / "spatialgeneval" / "summary.json"
    return {
        "geneval2": compute_geneval_score(variant_dir / "geneval2" / "scores.json"),
        "spatialgeneval": load_spatial_score(spatial_summary),
        "dpgbench": load_dpg_score(variant_dir / "dpgbench" / "summary.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write baseline comparison markdown from pipeline outputs")
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--warmup-dir", required=True)
    parser.add_argument("--rl-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    warmup_dir = Path(args.warmup_dir).resolve()
    rl_dir = Path(args.rl_dir).resolve()
    output_path = Path(args.output).resolve()

    base = variant_metrics(base_dir)
    warmup = variant_metrics(warmup_dir)
    rl = variant_metrics(rl_dir)

    lines = [
        "# Automatic Baseline Comparison",
        "",
        "数据来源：full_pipeline_h1004_a1002.sh 自动生成。",
        "",
        "| Variant | GenEval2 soft_tifa_gm | Delta vs Base | SpatialGenEval avg_acc | Delta vs Base | DPG-Bench Overall | Delta vs Base |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Base | {fmt(base['geneval2'])} | -- | {fmt(base['spatialgeneval'])} | -- | {fmt(base['dpgbench'])} | -- |",
        f"| Warmup | {fmt(warmup['geneval2'])} | {delta_text(warmup['geneval2'], base['geneval2'])} | {fmt(warmup['spatialgeneval'])} | {delta_text(warmup['spatialgeneval'], base['spatialgeneval'])} | {fmt(warmup['dpgbench'])} | {delta_text(warmup['dpgbench'], base['dpgbench'])} |",
        f"| RL | {fmt(rl['geneval2'])} | {delta_text(rl['geneval2'], base['geneval2'])} | {fmt(rl['spatialgeneval'])} | {delta_text(rl['spatialgeneval'], base['spatialgeneval'])} | {fmt(rl['dpgbench'])} | {delta_text(rl['dpgbench'], base['dpgbench'])} |",
        "",
        "## Notes",
        "",
        "- GenEval2 直接从 `scores.json` 计算 soft_tifa_gm。",
        "- SpatialGenEval 读取 `summary.json` 的 `overall.avg_acc`。",
        "- DPG-Bench 只有在 full pipeline 运行时提供了 `DPG_SCORE_COMMAND_TEMPLATE` 才会有正式分数，否则显示为 `--`。",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote comparison markdown to {output_path}")


if __name__ == "__main__":
    main()