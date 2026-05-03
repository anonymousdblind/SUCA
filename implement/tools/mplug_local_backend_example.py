"""Example local backend for tools/run_mplug_score.py.

This file documents the stable callable contract expected by
--scorer-target module:function.

Required callable signature:

    def score_image_prompt(
        image_path: str,
        prompt: str,
        model_path: str | None = None,
        **kwargs,
    ) -> float | dict:

Accepted return values:
1. float
2. dict containing one of: overall_score / score / dpg_score

Error handling:
- Raise ValueError for invalid input or decoding failures.
- Raise RuntimeError for backend/model initialization failures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def score_image_prompt(
    image_path: str,
    prompt: str,
    model_path: str | None = None,
    **kwargs: Any,
):
    """Example scorer implementation.

    This placeholder validates inputs and returns a deterministic dummy score.
    Replace the body with real mPLUG inference while keeping the signature stable.
    """

    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"image_path does not exist: {image_path}")
    if not prompt.strip():
        raise ValueError("prompt must be a non-empty string")

    backend_name = kwargs.get("backend_name", "mplug_local_example")
    batch_size = int(kwargs.get("batch_size", 1))
    device = kwargs.get("device", "cuda:0")

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    prompt_len = len(prompt.split())
    base_score = min(0.95, 0.35 + 0.03 * prompt_len)

    return {
        "overall_score": round(base_score, 4),
        "backend_name": backend_name,
        "device": device,
        "batch_size": batch_size,
        "model_path": model_path,
        "note": "Example local scorer only. Replace with real mPLUG inference.",
    }