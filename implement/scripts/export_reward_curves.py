"""Export canonical reward curve CSV for paper_artifact builder.

Supported inputs:
1. wide-format CSV/JSONL with step + per-unit columns
2. long-format CSV/JSONL with step + unit_type + value
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


UNIT_TYPES = ["Attribute", "Count", "Entity", "Relation"]
UNIT_ALIASES = {
    "Attribute": ["attribute", "attr", "attribute_reward", "attr_reward"],
    "Count": ["count", "count_reward", "counting"],
    "Entity": ["entity", "object", "entity_reward", "object_reward"],
    "Relation": ["relation", "rel", "relation_reward", "spatial"],
}


def normalize_text(text: str) -> str:
    return text.strip().lower().replace("_", "").replace("-", "").replace("/", "").replace(" ", "")


def canonical_unit(name: str) -> str | None:
    normalized = normalize_text(name)
    for unit, aliases in UNIT_ALIASES.items():
        if normalized == normalize_text(unit):
            return unit
        if normalized in {normalize_text(alias) for alias in aliases}:
            return unit
    return None


def load_records(path: Path):
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "records" in payload:
        return payload["records"]
    raise ValueError("Reward curve input must be CSV, JSONL, or JSON list")


def parse_float(value) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value).strip())


def extract_rows(records, step_field: str, value_field: str, unit_field: str) -> Dict[float, Dict[str, float]]:
    rows: Dict[float, Dict[str, float]] = defaultdict(dict)
    if not records:
        return rows

    sample = records[0]
    lower_keys = {normalize_text(key): key for key in sample.keys()}
    long_format = normalize_text(unit_field) in lower_keys and normalize_text(value_field) in lower_keys

    if long_format:
        step_key = lower_keys.get(normalize_text(step_field), step_field)
        unit_key = lower_keys[normalize_text(unit_field)]
        value_key = lower_keys[normalize_text(value_field)]
        for record in records:
            step = parse_float(record.get(step_key))
            unit = canonical_unit(str(record.get(unit_key, "")))
            value = parse_float(record.get(value_key))
            if step is None or unit is None or value is None:
                continue
            rows[step][unit] = value
        return rows

    step_key = lower_keys.get(normalize_text(step_field), step_field)
    wide_unit_keys = {}
    for raw_key in sample.keys():
        unit = canonical_unit(raw_key)
        if unit is not None:
            wide_unit_keys[unit] = raw_key

    for record in records:
        step = parse_float(record.get(step_key))
        if step is None:
            continue
        for unit, raw_key in wide_unit_keys.items():
            value = parse_float(record.get(raw_key))
            if value is not None:
                rows[step][unit] = value
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export canonical reward curve CSV")
    parser.add_argument("--input", required=True, help="CSV/JSON/JSONL file with reward curve data")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--label", required=True, help="Method label to attach in the output CSV")
    parser.add_argument("--step-field", default="step", help="Step column name")
    parser.add_argument("--unit-field", default="unit_type", help="Unit type field for long-format input")
    parser.add_argument("--value-field", default="value", help="Value field for long-format input")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    records = load_records(input_path)
    rows = extract_rows(records, args.step_field, args.value_field, args.unit_field)
    if not rows:
        raise ValueError("No reward curve points could be extracted from the input")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", *UNIT_TYPES, "method"])
        for step in sorted(rows.keys()):
            writer.writerow([
                int(step) if float(step).is_integer() else step,
                *("" if unit not in rows[step] else f"{rows[step][unit]:.6f}" for unit in UNIT_TYPES),
                args.label,
            ])
    print(f"Wrote reward curves to {output_path}")


if __name__ == "__main__":
    main()