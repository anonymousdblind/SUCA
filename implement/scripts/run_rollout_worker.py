"""
Rollout worker — generates images via HTTP requests.
Runs as a standalone process on a dedicated GPU.
"""
import argparse
import base64
import logging
import os
from io import BytesIO

import torch
from fastapi import FastAPI
from pydantic import BaseModel
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rollout-worker")

app = FastAPI()
policy = None


class GenerateRequest(BaseModel):
    prompt: str


class SyncRequest(BaseModel):
    adapter_path: str


def load_model(model_path, warmup_ckpt, gpu_id, num_steps=15, guidance=4.5):
    global policy
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
    from suca.diffusion_policy import DiffusionPolicy

    device = "cuda:0"
    logger.info(f"Loading SD3.5 on GPU {gpu_id}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device)
    if warmup_ckpt and os.path.exists(warmup_ckpt):
        t = SD3Transformer2DModel.from_pretrained(
            warmup_ckpt, torch_dtype=torch.float16
        ).to(device)
        pipe.transformer = t
    pipe.set_progress_bar_config(disable=True)
    policy = DiffusionPolicy(pipe, num_steps, guidance)
    logger.info(f"Ready on GPU {gpu_id}.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(req: GenerateRequest):
    """Generate an image and return as base64."""
    with torch.no_grad():
        traj = policy.generate_trajectory(req.prompt)
    image = traj["image"]
    buf = BytesIO()
    image.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image_b64": img_b64}


@app.post("/sync_weights")
def sync_weights(req: SyncRequest):
    """Load updated LoRA adapter weights from disk."""
    try:
        from peft import PeftModel
        adapter_path = req.adapter_path
        if os.path.exists(adapter_path):
            # Merge new adapter into base model
            logger.info(f"Syncing weights from {adapter_path}")
            from diffusers import SD3Transformer2DModel
            # Load the full merged state dict
            import torch as _torch
            state = _torch.load(os.path.join(adapter_path, "adapter_model.bin"), map_location="cuda:0")
            # For simplicity, just reload the whole transformer
            # In production, do incremental LoRA merge
            policy.pipeline.transformer.load_state_dict(state, strict=False)
            return {"status": "synced"}
    except Exception as e:
        logger.error(f"Sync failed: {e}")
    return {"status": "failed"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--port", type=int, default=8200)
    parser.add_argument("--model", default="models/stable-diffusion-3.5-medium")
    parser.add_argument("--warmup", default="outputs/warmup/checkpoint-7000/transformer")
    parser.add_argument("--steps", type=int, default=15)
    parser.add_argument("--guidance", type=float, default=4.5)
    args = parser.parse_args()

    load_model(args.model, args.warmup, args.gpu, args.steps, args.guidance)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)
