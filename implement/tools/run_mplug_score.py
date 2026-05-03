"""Standard adapter interface for DPG-Bench scoring backends.

This tool defines a stable CLI contract for scripts/eval_dpgbench_formal.py.
It supports three execution modes:

1. mock: deterministic fake score for smoke tests.
2. mplug + local transport: import a local Python scorer that can load real model weights.
3. mplug + http transport: call a remote scoring service.

The caller only depends on the output JSON containing overall_score.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import json
import sys
from io import BytesIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def resolve_callable(target: str):
    if ":" not in target:
        raise ValueError(f"Callable target must use module:function format, got: {target}")
    module_name, func_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    return func


def mock_score(image_path: Path, prompt: str) -> float:
    seed_text = f"{image_path.name}::{prompt}".encode("utf-8")
    digest = hashlib.sha256(seed_text).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return round(0.45 + 0.30 * bucket, 4)


def score_with_http_backend(image_path: Path, prompt: str, server_url: str, timeout: float, model_path: str | None = None) -> float:
    import requests

    payload = {
        "image_b64": load_image_base64(image_path),
        "prompt": prompt,
    }
    if model_path:
        payload["model_path"] = model_path
    response = requests.post(server_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    for key in ["overall_score", "score", "dpg_score"]:
        if key in data:
            return float(data[key])
    raise ValueError(f"HTTP backend did not return a known score field: {data}")


def score_with_local_backend(
    image_path: Path,
    prompt: str,
    scorer_target: str,
    model_path: str | None = None,
    extra_kwargs: dict | None = None,
) -> float:
    scorer_fn = resolve_callable(scorer_target)
    kwargs = dict(extra_kwargs or {})
    kwargs.update({
        "image_path": str(image_path),
        "prompt": prompt,
        "model_path": model_path,
    })
    result = scorer_fn(**kwargs)
    if isinstance(result, dict):
        for key in ["overall_score", "score", "dpg_score"]:
            if key in result:
                return float(result[key])
    return float(result)


def score_with_mplug_backend(
    image_path: Path,
    prompt: str,
    *,
    transport: str,
    server_url: str | None,
    scorer_target: str | None,
    model_path: str | None,
    timeout: float,
    extra_kwargs: dict | None,
) -> float:
    if transport == "http":
        if not server_url:
            raise ValueError("server_url is required when transport=http")
        return score_with_http_backend(image_path, prompt, server_url, timeout, model_path=model_path)

    if transport == "local":
        if not scorer_target:
            raise ValueError("local backend requires --scorer-target module:function")
        return score_with_local_backend(
            image_path,
            prompt,
            scorer_target=scorer_target,
            model_path=model_path,
            extra_kwargs=extra_kwargs,
        )

    raise ValueError(f"Unsupported transport: {transport}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DPG-Bench mPLUG score adapter")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--prompt", default="", help="Prompt text associated with the image")
    parser.add_argument(
        "--backend",
        choices=["stub", "mock", "mplug"],
        default="stub",
        help="Scoring backend. Use mock only for pipeline smoke tests, not for paper results.",
    )
    parser.add_argument(
        "--transport",
        choices=["local", "http"],
        default="local",
        help="Transport used by the mplug backend: local Python scorer or remote HTTP service.",
    )
    parser.add_argument("--server-url", default=None, help="HTTP endpoint returning overall_score/score/dpg_score JSON")
    parser.add_argument("--scorer-target", default=None, help="Local Python callable in module:function format")
    parser.add_argument("--model-path", default=None, help="Optional local model/checkpoint path passed through to the backend")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout in seconds for remote scoring")
    parser.add_argument(
        "--backend-extra-json",
        default=None,
        help="Optional JSON object string passed through to local backend kwargs",
    )
    args = parser.parse_args()

    image_path = Path(args.image).resolve()
    output_path = Path(args.output).resolve()

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if args.backend == "stub":
        raise RuntimeError(
            "Placeholder adapter invoked with --backend stub. "
            "Use --backend mock for smoke tests or implement the real mPLUG backend."
        )
    if args.backend == "mock":
        score = mock_score(image_path, args.prompt)
        payload = {
            "backend": "mock",
            "overall_score": score,
            "is_placeholder": True,
            "image": str(image_path),
            "prompt": args.prompt,
        }
    else:
        extra_kwargs = json.loads(args.backend_extra_json) if args.backend_extra_json else None
        score = score_with_mplug_backend(
            image_path,
            args.prompt,
            transport=args.transport,
            server_url=args.server_url,
            scorer_target=args.scorer_target,
            model_path=args.model_path,
            timeout=args.timeout,
            extra_kwargs=extra_kwargs,
        )
        payload = {
            "backend": "mplug",
            "transport": args.transport,
            "overall_score": float(score),
            "is_placeholder": False,
            "image": str(image_path),
            "prompt": args.prompt,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote score JSON to {output_path}")


if __name__ == "__main__":
    main()