"""Export canonical time_breakdown.json for the paper artifact builder.

Supported inputs:
1. long-format CSV/JSONL with stage + duration columns
2. wide-format CSV/JSON with one row/object containing per-stage durations
3. JSON with a nested "stages" object
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CANONICAL_STAGES = [
    "Image sampling",
    "Semantic parsing",
    "VLM verification",
    "Routing aggregation",
    "Backward step",
]

STAGE_ALIASES = {
    "Image sampling": [
        "image sampling", "sampling", "rollout", "generation", "decode", "image_generation", "sampling_time",
    ],
    "Semantic parsing": [
        "semantic parsing", "parsing", "unit parsing", "parser", "parse_time",
    ],
    "VLM verification": [
        "vlm verification", "verification", "reward", "reward model", "vqa", "judge", "reward_time",
    ],
    "Routing aggregation": [
        "routing aggregation", "routing", "aggregation", "responsibility", "c[k,t]", "credit assignment",
    ],
    "Backward step": [
        "backward step", "backward", "optimization", "optimizer", "train step", "update",
    ],
}


def normalize_text(text: str) -> str:
    return text.strip().lower().replace("_", " ").replace("-", " ")


def canonical_stage(name: str) -> str | None:
    normalized = normalize_text(name)
    for stage, aliases in STAGE_ALIASES.items():
        if normalized == normalize_text(stage):
            return stage
        if any(alias in normalized for alias in aliases):
            return stage
    return None


def load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_json_family(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def parse_number(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("%", "")
    return float(text)


def extract_duration(record: Dict[str, object], duration_field: str) -> float | None:
    if duration_field in record:
        return parse_number(record[duration_field])
    if "duration_sec" in record:
        return parse_number(record["duration_sec"])
    if "duration_ms" in record:
        value = parse_number(record["duration_ms"])
        return None if value is None else value / 1000.0
    if "seconds" in record:
        return parse_number(record["seconds"])
    if "start" in record and "end" in record:
        start = parse_number(record["start"])
        end = parse_number(record["end"])
        if start is not None and end is not None:
            return end - start
    return None


def accumulate_long_format(records: Iterable[Dict[str, object]], stage_field: str, duration_field: str) -> Dict[str, float]:
    durations: Dict[str, float] = defaultdict(float)
    for record in records:
        raw_stage = record.get(stage_field) or record.get("stage") or record.get("name") or record.get("event")
        if raw_stage is None:
            continue
        stage = canonical_stage(str(raw_stage))
        if stage is None:
            continue
        duration = extract_duration(record, duration_field)
        if duration is None:
            continue
        durations[stage] += duration
    return dict(durations)


def accumulate_wide_format(record: Dict[str, object]) -> Dict[str, float]:
    durations: Dict[str, float] = defaultdict(float)
    for key, value in record.items():
        stage = canonical_stage(key)
        if stage is None:
            continue
        duration = parse_number(value)
        if duration is None:
            continue
        durations[stage] += duration
    return dict(durations)


def extract_durations(payload, stage_field: str, duration_field: str) -> Dict[str, float]:
    if isinstance(payload, dict):
        if "stages" in payload:
            stages = payload["stages"]
            if isinstance(stages, dict):
                return accumulate_wide_format(stages)
            if isinstance(stages, list):
                return accumulate_long_format(stages, stage_field, duration_field)
        return accumulate_wide_format(payload)
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and any(stage_field in row or "stage" in row or "name" in row for row in payload):
            return accumulate_long_format(payload, stage_field, duration_field)
        if payload and isinstance(payload[0], dict):
            merged: Dict[str, float] = defaultdict(float)
            for row in payload:
                for stage, value in accumulate_wide_format(row).items():
                    merged[stage] += value
            return dict(merged)
    raise ValueError("Unsupported input structure for timing export")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical time_breakdown.json")
    parser.add_argument("--input", required=True, help="CSV/JSON/JSONL file with timing data")
    parser.add_argument("--output", default="analysis/time_breakdown.json", help="Output JSON path")
    parser.add_argument("--stage-field", default="stage", help="Field name for stage/event in long-format input")
    parser.add_argument("--duration-field", default="duration_sec", help="Field name for duration in long-format input")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    payload = load_csv(input_path) if input_path.suffix.lower() == ".csv" else load_json_family(input_path)
    durations = extract_durations(payload, args.stage_field, args.duration_field)
    total = sum(durations.values())
    if total <= 0:
        raise ValueError("No stage durations could be extracted from the input")

    summary = {stage: round(100.0 * durations.get(stage, 0.0) / total, 2) for stage in CANONICAL_STAGES}
    summary["_meta"] = {
        "total_duration_sec": round(total, 4),
        "source": str(input_path),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote timing summary to {output_path}")


if __name__ == "__main__":
    main()