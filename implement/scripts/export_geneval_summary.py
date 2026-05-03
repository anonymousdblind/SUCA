"""Convert formal_comparison.json into normalized benchmark summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export normalized GenEval2 summary from formal_comparison.json")
    parser.add_argument("--input", required=True, help="Path to formal_comparison.json")
    parser.add_argument("--model-key", required=True, help="Model key inside the input JSON, e.g. base or grpo")
    parser.add_argument("--output", required=True, help="Output normalized summary JSON path")
    parser.add_argument("--variant-name", default=None, help="Optional variant name to store in the JSON")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).resolve().read_text(encoding="utf-8"))
    model_payload = payload[args.model_key]
    summary = {
        "benchmark": "geneval2",
        "variant": args.variant_name or args.model_key,
        "overall": {
            "soft_tifa_gm": model_payload.get("soft_tifa_gm"),
        },
        "skills": model_payload.get("per_skill", {}),
        "atom_count": {str(key): value for key, value in model_payload.get("per_atom", {}).items()},
    }

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote normalized GenEval2 summary to {output_path}")


if __name__ == "__main__":
    main()