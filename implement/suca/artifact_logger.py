from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List


class ArtifactLogger:
    """Write structured training artifacts for later paper aggregation."""

    def __init__(self, root_dir: str, analysis_subdir: str = "analysis"):
        self.root_dir = Path(root_dir)
        if self.root_dir.name == analysis_subdir:
            self.analysis_dir = self.root_dir
        else:
            self.analysis_dir = self.root_dir / analysis_subdir
        self.time_dir = self.analysis_dir / "time"
        self.curves_dir = self.analysis_dir / "curves"
        self.logs_dir = self.analysis_dir / "logs"
        for directory in [self.analysis_dir, self.time_dir, self.curves_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def append_step_timing(self, record: Dict[str, object]) -> None:
        self._append_jsonl(self.time_dir / "step_timing.jsonl", record)

    def append_step_metrics(self, record: Dict[str, object]) -> None:
        self._append_jsonl(self.logs_dir / "step_metrics.jsonl", record)

    def append_unit_rewards(
        self,
        *,
        step: int,
        prompt: str,
        rewards_by_type: Dict[str, List[float]],
        method_label: str | None = None,
    ) -> None:
        for unit_type, values in rewards_by_type.items():
            if not values:
                continue
            record = {
                "step": step,
                "prompt": prompt,
                "unit_type": unit_type,
                "value": sum(values) / len(values),
            }
            if method_label:
                record["method"] = method_label
            self._append_jsonl(self.curves_dir / "per_unit_rewards.jsonl", record)

    @staticmethod
    def rewards_by_unit_type(units: Iterable[object], rewards: Iterable[object]) -> Dict[str, List[float]]:
        grouped: Dict[str, List[float]] = defaultdict(list)
        for unit, reward in zip(units, rewards):
            unit_type = getattr(getattr(unit, "unit_type", None), "value", None)
            if unit_type is None:
                continue
            grouped[unit_type.capitalize()].append(float(reward))
        return dict(grouped)

    @staticmethod
    def _append_jsonl(path: Path, record: Dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")