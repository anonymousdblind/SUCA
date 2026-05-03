"""
Diffusion Policy Wrapper for SD3.5 (MMDiT): Treats the denoising process as an RL policy.

The denoising trajectory z_T -> z_{T-1} -> ... -> z_0 is modeled as a
sequence of actions, where:
  - state s_t = (z_t, t, c)  [latent, timestep, text conditioning]
  - action a_t = v_theta(z_t, t, c)  [predicted velocity / flow]
  - pi_theta(a_t | s_t) = N(a_t; v_theta(z_t, t, c), sigma^2 I)

SD3.5 uses Flow Matching with an MMDiT (SD3Transformer2DModel) instead of
a UNet, and has 3 text encoders (CLIP-L, CLIP-G, T5-XXL).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image


class DiffusionPolicy:
    """
    Wraps a StableDiffusion3Pipeline as an RL policy for SUCA training.
    Handles:
      - Forward denoising with noise/latent tracking
      - Log-probability computation for policy gradient
      - KL divergence against a reference policy
    """

    def __init__(
        self,
        pipeline,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
    ):
        self.pipeline = pipeline
        self.transformer = pipeline.transformer
        self.scheduler = pipeline.scheduler
        self.vae = pipeline.vae
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = pipeline.device

    @torch.no_grad()
    def encode_prompt(self, prompt: str) -> Dict[str, torch.Tensor]:
        """Encode text prompt using SD3's triple text encoders.

        Returns a dict with:
          - prompt_embeds: combined text embeddings for conditional
          - negative_prompt_embeds: combined text embeddings for unconditional
          - pooled_prompt_embeds: pooled embeddings for conditional
          - negative_pooled_prompt_embeds: pooled embeddings for unconditional
        """
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.pipeline.encode_prompt(
            prompt=prompt,
            prompt_2=None,
            prompt_3=None,
            negative_prompt="",
            device=self.device,
            num_images_per_prompt=1,
        )
        return {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
        }

    def generate_trajectory(
        self,
        prompt: str,
        text_cond: Optional[Dict[str, torch.Tensor]] = None,
        latents: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        attention_extractor=None,
    ) -> Dict:
        """
        Run the full denoising trajectory, recording all intermediate states.

        Returns dict with:
          - "image": decoded PIL image
          - "latents_trajectory": list of latent tensors at each step
          - "model_outputs": list of transformer outputs at each step
          - "timesteps": list of timestep values
          - "initial_latents": the starting noise
          - "text_cond": text conditioning dict
        """
        if text_cond is None:
            text_cond = self.encode_prompt(prompt)

        prompt_embeds = text_cond["prompt_embeds"]
        negative_prompt_embeds = text_cond["negative_prompt_embeds"]
        pooled_prompt_embeds = text_cond["pooled_prompt_embeds"]
        negative_pooled_prompt_embeds = text_cond["negative_pooled_prompt_embeds"]

        # Initialize scheduler
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Initialize latents — SD3.5 has 16 channels, patch_size=2
        num_channels = self.transformer.config.in_channels
        height = width = self.pipeline.default_sample_size * self.pipeline.vae_scale_factor
        shape = (
            1,
            num_channels,
            height // self.pipeline.vae_scale_factor,
            width // self.pipeline.vae_scale_factor,
        )
        if latents is None:
            latents = torch.randn(
                shape, generator=generator, device=self.device, dtype=prompt_embeds.dtype
            )
        initial_latents = latents.clone()

        latents_trajectory = [latents.clone()]
        model_outputs_list = []

        # Classifier-free guidance: concat negative + positive
        do_cfg = self.guidance_scale > 1.0
        if do_cfg:
            prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled_embeds_combined = torch.cat(
                [negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0
            )

        for i, t in enumerate(timesteps):
            if attention_extractor is not None:
                attention_extractor.set_timestep(i)

            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            # SD3 scheduler scaling
            # Flow matching scheduler: no input scaling needed
            if hasattr(self.scheduler, 'scale_model_input'):
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # else: flow matching doesn't require scaling

            embeds = prompt_embeds_combined if do_cfg else prompt_embeds
            pooled = pooled_embeds_combined if do_cfg else pooled_prompt_embeds

            # MMDiT forward
            model_output = self.transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(latent_model_input.shape[0]),
                encoder_hidden_states=embeds,
                pooled_projections=pooled,
                return_dict=False,
            )[0]

            # Classifier-free guidance
            if do_cfg:
                output_uncond, output_text = model_output.chunk(2)
                model_output = output_uncond + self.guidance_scale * (output_text - output_uncond)

            model_outputs_list.append(model_output.clone())

            # Scheduler step
            latents = self.scheduler.step(model_output, t, latents).prev_sample
            latents_trajectory.append(latents.clone())

        # Decode to image
        image = self._decode_latents(latents)

        return {
            "image": image,
            "latents_trajectory": latents_trajectory,
            "model_outputs": model_outputs_list,
            "timesteps": timesteps.tolist(),
            "initial_latents": initial_latents,
            "text_cond": text_cond,
        }

    def compute_log_probs(
        self,
        trajectory: Dict,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute log pi_theta(a_t | s_t) for each timestep.

        Under Gaussian policy assumption:
          log pi(a_t|s_t) = -||a_t - mu_theta(s_t)||^2 / (2*sigma^2) - C
        """
        text_cond = trajectory["text_cond"]
        latents_traj = trajectory["latents_trajectory"]
        outputs_old = trajectory["model_outputs"]
        timesteps = trajectory["timesteps"]

        prompt_embeds = text_cond["prompt_embeds"]
        negative_prompt_embeds = text_cond["negative_prompt_embeds"]
        pooled_prompt_embeds = text_cond["pooled_prompt_embeds"]
        negative_pooled_prompt_embeds = text_cond["negative_pooled_prompt_embeds"]

        do_cfg = self.guidance_scale > 1.0
        if do_cfg:
            embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            embeds = prompt_embeds
            pooled = pooled_prompt_embeds

        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        log_probs = []
        for i, t_val in enumerate(timesteps):
            t = torch.tensor([t_val], device=self.device)
            latents = latents_traj[i]

            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            # Flow matching scheduler: no input scaling needed
            if hasattr(self.scheduler, 'scale_model_input'):
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # else: flow matching doesn't require scaling

            model_output = self.transformer(
                hidden_states=latent_model_input,
                timestep=t.expand(latent_model_input.shape[0]),
                encoder_hidden_states=embeds,
                pooled_projections=pooled,
                return_dict=False,
            )[0]

            if do_cfg:
                out_uncond, out_text = model_output.chunk(2)
                mu = out_uncond + self.guidance_scale * (out_text - out_uncond)
            else:
                mu = model_output

            a = outputs_old[i]
            diff = (a - mu).flatten()
            # Normalize by sqrt(num_elements) for stable gradients
            log_prob = -0.5 * (diff ** 2).sum() / (diff.numel() ** 0.5) / (sigma ** 2)
            log_probs.append(log_prob)

        return torch.stack(log_probs)

    def compute_kl_divergence(
        self,
        trajectory: Dict,
        ref_transformer: nn.Module,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute per-timestep KL divergence between current and reference policy.

        KL(pi_theta || pi_ref) at timestep t ~ ||mu_theta - mu_ref||^2 / (2*sigma^2)
        """
        text_cond = trajectory["text_cond"]
        latents_traj = trajectory["latents_trajectory"]
        timesteps = trajectory["timesteps"]

        prompt_embeds = text_cond["prompt_embeds"]
        negative_prompt_embeds = text_cond["negative_prompt_embeds"]
        pooled_prompt_embeds = text_cond["pooled_prompt_embeds"]
        negative_pooled_prompt_embeds = text_cond["negative_pooled_prompt_embeds"]

        do_cfg = self.guidance_scale > 1.0
        if do_cfg:
            embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            embeds = prompt_embeds
            pooled = pooled_prompt_embeds

        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)

        kl_values = []
        for i, t_val in enumerate(timesteps):
            t = torch.tensor([t_val], device=self.device)
            latents = latents_traj[i]

            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            # Flow matching scheduler: no input scaling needed
            if hasattr(self.scheduler, 'scale_model_input'):
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            # else: flow matching doesn't require scaling

            t_expanded = t.expand(latent_model_input.shape[0])

            # Current policy
            mu_current_raw = self.transformer(
                hidden_states=latent_model_input,
                timestep=t_expanded,
                encoder_hidden_states=embeds,
                pooled_projections=pooled,
                return_dict=False,
            )[0]

            # Reference policy (may be on different device)
            ref_device = next(ref_transformer.parameters()).device
            with torch.no_grad():
                mu_ref_raw = ref_transformer(
                    hidden_states=latent_model_input.to(ref_device),
                    timestep=t_expanded.to(ref_device),
                    encoder_hidden_states=embeds.to(ref_device),
                    pooled_projections=pooled.to(ref_device),
                    return_dict=False,
                )[0].to(self.device)

            if do_cfg:
                uc, tc = mu_current_raw.chunk(2)
                mu_current = uc + self.guidance_scale * (tc - uc)
                ur, tr = mu_ref_raw.chunk(2)
                mu_ref = ur + self.guidance_scale * (tr - ur)
            else:
                mu_current = mu_current_raw
                mu_ref = mu_ref_raw

            diff = (mu_current - mu_ref).flatten()
            kl = 0.5 * (diff ** 2).sum() / (sigma ** 2)
            kl_values.append(kl)

        return torch.stack(kl_values)

    def compute_log_prob_single(
        self,
        trajectory: Dict,
        t_idx: int,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """Compute log prob for a single timestep (memory efficient)."""
        text_cond = trajectory["text_cond"]
        latents = trajectory["latents_trajectory"][t_idx].detach()
        t_val = trajectory["timesteps"][t_idx]
        a = trajectory["model_outputs"][t_idx].detach()

        prompt_embeds = text_cond["prompt_embeds"]
        negative_prompt_embeds = text_cond["negative_prompt_embeds"]
        pooled_prompt_embeds = text_cond["pooled_prompt_embeds"]
        negative_pooled_prompt_embeds = text_cond["negative_pooled_prompt_embeds"]

        t = torch.tensor([t_val], device=self.device)

        do_cfg = self.guidance_scale > 1.0
        if do_cfg:
            latent_input = torch.cat([latents, latents], dim=0)
            embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            pooled = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
        else:
            latent_input = latents
            embeds = prompt_embeds
            pooled = pooled_prompt_embeds

        # Flow matching: no scaling needed
        if hasattr(self.scheduler, 'scale_model_input'):
            latent_input = self.scheduler.scale_model_input(latent_input, t)

        model_output = self.transformer(
            hidden_states=latent_input,
            timestep=t.expand(latent_input.shape[0]),
            encoder_hidden_states=embeds,
            pooled_projections=pooled,
            return_dict=False,
        )[0]

        if do_cfg:
            out_uncond, out_text = model_output.chunk(2)
            mu = out_uncond + self.guidance_scale * (out_text - out_uncond)
        else:
            mu = model_output

        diff = (a - mu).flatten()
        # Normalize by number of elements to keep log_prob in reasonable range
        num_elements = diff.numel()
        log_prob = -0.5 * (diff ** 2).mean() / (sigma ** 2)
        return log_prob

    def compute_weighted_loss_batched(
        self,
        trajectory: Dict,
        weights_per_t: torch.Tensor,
        batch_size: int = 10,
        sigma: float = 1.0,
    ) -> float:
        """Compute weighted policy gradient loss — all timesteps batched, NO CFG.

        Key optimizations vs naive per-timestep approach:
        1. Skip CFG during backward (only conditional forward) — 2x speedup
        2. Batch all T timesteps into 1-2 forward passes — 10-20x speedup
        3. Single backward call — avoids repeated graph construction

        Total: ~20x faster than per-timestep with CFG.
        """
        text_cond = trajectory["text_cond"]
        T = len(trajectory["timesteps"])

        prompt_embeds = text_cond["prompt_embeds"]       # [1, S, D]
        pooled_prompt_embeds = text_cond["pooled_prompt_embeds"]  # [1, D]

        total_loss = 0.0

        # Process in chunks to fit memory (no CFG = half the memory)
        for batch_start in range(0, T, batch_size):
            batch_end = min(batch_start + batch_size, T)
            batch_indices = list(range(batch_start, batch_end))
            B = len(batch_indices)

            # Stack all timestep latents and targets
            latents_batch = torch.cat(
                [trajectory["latents_trajectory"][i].detach() for i in batch_indices],
                dim=0,
            )  # [B, C, H, W]
            targets_batch = torch.cat(
                [trajectory["model_outputs"][i].detach() for i in batch_indices],
                dim=0,
            )  # [B, C, H, W]
            t_vals = torch.tensor(
                [trajectory["timesteps"][i] for i in batch_indices],
                device=self.device,
            )  # [B]
            w_batch = weights_per_t[batch_indices]  # [B]

            # Conditional-only forward (skip CFG — not needed for log_prob)
            embeds = prompt_embeds.expand(B, -1, -1)
            pooled = pooled_prompt_embeds.expand(B, -1)

            model_output = self.transformer(
                hidden_states=latents_batch,
                timestep=t_vals,
                encoder_hidden_states=embeds,
                pooled_projections=pooled,
                return_dict=False,
            )[0]  # [B, C, H, W]

            # Per-sample log_prob
            diff = (targets_batch - model_output).view(B, -1)
            log_probs = -0.5 * (diff ** 2).mean(dim=1) / (sigma ** 2)

            # Weighted loss
            batch_loss = -(w_batch * log_probs).sum() / T

            if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                continue

            batch_loss.backward()
            total_loss += batch_loss.item()

        return total_loss

    def _decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode latent tensor to PIL image."""
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        with torch.no_grad():
            image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # Handle NaN values
        if torch.isnan(image).any():
            image = torch.nan_to_num(image, nan=0.5)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()[0]
        image = (image * 255).round().astype("uint8")
        return Image.fromarray(image)
