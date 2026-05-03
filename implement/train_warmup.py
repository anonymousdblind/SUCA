"""
SFT Warmup Training Script for SD3.5 Medium.

Supervised fine-tuning (denoising loss) on Fine-T2I image-text pairs
before RL training. Uses Accelerate for multi-GPU DDP.

Usage:
    accelerate launch --config_file config/accelerate_warmup.yaml train_warmup.py
"""

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from suca.warmup_dataset import WarmupDataset

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SD3.5 SFT Warmup")
    parser.add_argument("--model_name", type=str, default="models/stable-diffusion-3.5-medium")
    parser.add_argument("--metadata_file", type=str, default="data/fine-t2i/warmup_100k.json")
    parser.add_argument("--data_root", type=str, default="data/fine-t2i")
    parser.add_argument("--output_dir", type=str, default="outputs/warmup")
    parser.add_argument("--analysis_dir", type=str, default=None,
                        help="Structured analysis root. Defaults to <repo>/analysis for manifest alignment.")
    parser.add_argument("--resolution", type=int, default=512,
                        help="Training resolution (512 for warmup to save memory, 1024 for RL)")
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=10,
                        help="Max epochs (training stops at max_train_steps regardless)")
    parser.add_argument("--max_train_steps", type=int, default=5000)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--report_to", type=str, default="none", choices=["none", "wandb"])
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint dir to resume from (e.g. outputs/warmup/checkpoint-3000)")
    return parser.parse_args()


def encode_prompts(pipeline, prompts, device, weight_dtype):
    """Encode a batch of prompts using SD3's triple text encoders."""
    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=None,
            prompt_3=None,
            device=device,
            num_images_per_prompt=1,
        )
    return prompt_embeds.to(dtype=weight_dtype), pooled_prompt_embeds.to(dtype=weight_dtype)


@record
def main():
    args = parse_args()

    for path in [args.model_name, args.metadata_file, args.data_root]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required warmup path not found: {path}")

    repo_root = Path(__file__).resolve().parent
    if args.analysis_dir is None:
        args.analysis_dir = str((repo_root / "analysis").resolve())
    os.makedirs(args.analysis_dir, exist_ok=True)

    log_with = None if args.report_to == "none" else args.report_to

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=os.path.join(args.analysis_dir, "warmup_logs"),
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process and args.report_to != "none":
        accelerator.init_trackers(
            project_name="suca-warmup-sft",
            config=vars(args),
        )

    set_seed(args.seed)

    # Load pipeline
    logger.info(f"Loading SD3.5 pipeline from {args.model_name}")
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
    )

    # Extract components
    transformer = pipeline.transformer
    vae = pipeline.vae
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)

    # Freeze VAE and text encoders
    vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    if pipeline.text_encoder_3 is not None:
        pipeline.text_encoder_3.requires_grad_(False)

    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Transformer must be in float32 for gradient computation (AMP handles mixed precision)
    weight_dtype = torch.float16 if args.mixed_precision == "fp16" else (
        torch.bfloat16 if args.mixed_precision == "bf16" else torch.float32
    )
    transformer.to(torch.float32)
    vae.to(accelerator.device, dtype=weight_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=weight_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    if pipeline.text_encoder_3 is not None:
        pipeline.text_encoder_3.to(accelerator.device, dtype=weight_dtype)

    # Optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-2,
    )

    # Dataset and dataloader
    dataset = WarmupDataset(
        metadata_file=args.metadata_file,
        data_root=args.data_root,
        resolution=args.resolution,
    )
    logger.info(f"Dataset: {len(dataset)} samples")

    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        prompts = [b["prompt"] for b in batch]
        return {"pixel_values": pixel_values, "prompts": prompts}

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # LR scheduler
    from torch.optim.lr_scheduler import CosineAnnealingLR
    num_update_steps = min(
        args.max_train_steps,
        math.ceil(len(dataloader) / args.gradient_accumulation_steps) * args.num_train_epochs,
    )
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_update_steps, eta_min=1e-7)

    # Prepare with accelerator
    transformer, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, dataloader, lr_scheduler
    )

    # Training
    total_batch_size = (
        args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    )
    logger.info("***** Running SFT Warmup *****")
    logger.info(f"  Num samples = {len(dataset)}")
    logger.info(f"  Num epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total batch size = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Max train steps = {args.max_train_steps}")
    logger.info(f"  Resolution = {args.resolution}")

    global_step = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint is not None:
        ckpt_transformer_path = os.path.join(args.resume_from_checkpoint, "transformer")
        if os.path.exists(ckpt_transformer_path):
            logger.info(f"Resuming from {args.resume_from_checkpoint}")
            from diffusers.models import SD3Transformer2DModel
            ckpt_state = SD3Transformer2DModel.from_pretrained(ckpt_transformer_path)
            unwrapped = accelerator.unwrap_model(transformer)
            unwrapped.load_state_dict(ckpt_state.state_dict())
            del ckpt_state
            # Extract step number from checkpoint name
            ckpt_name = os.path.basename(args.resume_from_checkpoint)
            if ckpt_name.startswith("checkpoint-"):
                global_step = int(ckpt_name.split("-")[1])
                # Advance lr_scheduler
                for _ in range(global_step):
                    lr_scheduler.step()
                logger.info(f"Resumed at step {global_step}")
        else:
            logger.warning(f"Checkpoint not found at {ckpt_transformer_path}, training from scratch")

    progress_bar = tqdm(
        range(args.max_train_steps),
        initial=global_step,
        desc="Training",
        disable=not accelerator.is_local_main_process,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Calculate how many dataloader steps to skip for resume
    steps_to_skip = global_step * args.gradient_accumulation_steps

    for epoch in range(args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(dataloader):
            if global_step >= args.max_train_steps:
                break
            # Skip already-trained steps when resuming
            if steps_to_skip > 0:
                steps_to_skip -= 1
                continue

            with accelerator.accumulate(transformer):
                # 1. Encode images to latents
                pixel_values = batch["pixel_values"].to(dtype=weight_dtype)
                with torch.no_grad():
                    latents = vae.encode(pixel_values).latent_dist.sample()
                    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

                # 2. Encode text
                prompt_embeds, pooled_prompt_embeds = encode_prompts(
                    pipeline, batch["prompts"], accelerator.device, weight_dtype
                )

                # 3. Sample noise and timesteps (flow matching)
                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]

                # Uniform timestep sampling for flow matching [0, 1]
                # Using logit-normal distribution (SD3 training recipe)
                u = torch.randn(batch_size, device=latents.device)
                t = torch.sigmoid(u)  # logit-normal -> (0, 1)

                # Flow matching: x_t = (1 - t) * x_0 + t * noise
                t_expand = t.view(batch_size, 1, 1, 1)
                noisy_latents = (1 - t_expand) * latents + t_expand * noise

                # Target: velocity = noise - x_0
                target = noise - latents

                # 4. Predict with transformer
                # SD3 scheduler expects timesteps in [0, 1000] range
                timesteps = (t * 1000).long()

                model_pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

                # 5. Compute loss (MSE on velocity prediction)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(
                    loss=f"{loss.detach().item():.4f}",
                    lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
                )

                if global_step % args.log_steps == 0:
                    accelerator.log(
                        {"train_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]},
                        step=global_step,
                    )
                    logger.info(
                        f"Step {global_step}/{args.max_train_steps} | "
                        f"Loss: {loss.detach().item():.4f} | "
                        f"LR: {lr_scheduler.get_last_lr()[0]:.2e}"
                    )

                if global_step % args.save_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(transformer)
                        unwrapped.save_pretrained(os.path.join(save_path, "transformer"))
                        logger.info(f"Checkpoint saved to {save_path}")

    # Final save
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "checkpoint-final")
        os.makedirs(save_path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(transformer)
        unwrapped.save_pretrained(os.path.join(save_path, "transformer"))
        logger.info(f"Final checkpoint saved to {save_path}")

    accelerator.end_training()
    logger.info("SFT Warmup complete!")


if __name__ == "__main__":
    main()
