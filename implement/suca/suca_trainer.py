"""
SUCA Trainer: Semantic Unit Credit Assignment training loop.

Implements the core SUCA algorithm:
  J(theta) = E[ Sum_k Sum_t C_{k,t} * A_k * log pi_theta(a_t | s_t) ]

where:
  - C_{k,t} is the unit-timestep responsibility matrix
  - A_k is the unit-level advantage
  - log pi_theta(a_t | s_t) is the policy log-probability at timestep t

Adapted for SD3.5 Medium (MMDiT + Flow Matching + triple text encoders).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from .attention_extractor import CrossAttentionExtractor
from .artifact_logger import ArtifactLogger
from .diffusion_policy import DiffusionPolicy
from .responsibility_matrix import ResponsibilityMatrixBuilder
from .semantic_parser import SemanticUnit, SemanticUnitParser
from .unit_reward import UnitRewardComputer

logger = logging.getLogger(__name__)


class SUCATrainer:
    """
    Main SUCA training loop.

    Algorithm:
      1. Parse prompt q -> semantic units S(q) = {s_1, ..., s_K}
      2. For each unit, construct VQA question q_k and token set T_k
      3. Sample image I and denoising trajectory {z_t}
      4. Compute unit rewards r_k
      5. Estimate unit uncertainties u_k (multi-sample)
      6. Compute corrected rewards r_tilde_k = r_k - lambda_u * u_k
      7. Extract joint attention maps -> a_{k,t}
      8. Normalize to responsibility matrix C_{k,t}
      9. Compute unit advantages A_k
      10. Update parameters via: Sum_{k,t} C_{k,t} * A_k * log pi_theta(a_t | s_t)
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Support multi-GPU: diffusion on one device, VLM on another
        self.diffusion_device = cfg.get("devices", {}).get("diffusion", "cuda:0")
        self.vlm_device = cfg.get("devices", {}).get("vlm", "cuda:1")
        self.device = self.diffusion_device

        # Will be initialized in setup()
        self.pipeline = None
        self.policy: Optional[DiffusionPolicy] = None
        self.ref_transformer: Optional[nn.Module] = None
        self.parser: Optional[SemanticUnitParser] = None
        self.reward_computer: Optional[UnitRewardComputer] = None
        self.responsibility_builder: Optional[ResponsibilityMatrixBuilder] = None
        self.attention_extractor: Optional[CrossAttentionExtractor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

        # Baselines for each unit (running average)
        self._unit_baselines: Dict[str, float] = {}
        self._baseline_momentum = 0.9

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.artifact_logger: Optional[ArtifactLogger] = None

        # Checkpoint management
        self._recent_checkpoints: List[str] = []  # paths of recent checkpoints
        self._best_checkpoint_path: Optional[str] = None
        self._best_eval_score: float = -float("inf")

    def setup(self):
        """Initialize all components."""
        logger.info("Setting up SUCA trainer...")

        # 1. Load diffusion pipeline
        self._setup_diffusion_pipeline()

        # 2. Initialize semantic parser (rule-based: lightweight, deterministic, no extra model)
        self.parser = SemanticUnitParser(use_rules_only=True)

        # 3. Initialize reward computer (on separate GPU to avoid OOM)
        self.reward_computer = UnitRewardComputer(
            vlm_model_name=self.cfg.model.vlm_model_name,
            device=self.vlm_device,
            lambda_u=self.cfg.suca.lambda_u,
        )

        # 4. Initialize responsibility matrix builder
        self.responsibility_builder = ResponsibilityMatrixBuilder(
            tau=self.cfg.suca.tau,
            top_m_spatial=self.cfg.suca.top_m_spatial,
        )

        # 5. Initialize attention extractor (on transformer, not unet)
        self.attention_extractor = CrossAttentionExtractor(
            transformer=self.pipeline.transformer,
            layer_indices=list(self.cfg.suca.attention_layers),
        )

        # 6. Create reference transformer (frozen copy on separate GPU)
        ref_device = self.cfg.get("devices", {}).get("ref", self.vlm_device)
        self.ref_transformer = copy.deepcopy(self.pipeline.transformer).to(ref_device)
        self.ref_transformer.eval()
        for p in self.ref_transformer.parameters():
            p.requires_grad = False
        self._ref_device = ref_device

        # 7. Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.pipeline.transformer.parameters(),
            lr=self.cfg.training.learning_rate,
        )

        # 8. Output directory
        os.makedirs(self.cfg.logging.output_dir, exist_ok=True)
        analysis_root = self.cfg.logging.get("analysis_dir") or self.cfg.logging.output_dir
        self.artifact_logger = ArtifactLogger(analysis_root)

        logger.info("SUCA trainer setup complete.")

    def _setup_diffusion_pipeline(self):
        """Load the SD3.5 diffusion model pipeline."""
        from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline, SD3Transformer2DModel

        model_name = self.cfg.model.pretrained_model_name
        logger.info(f"Loading diffusion model: {model_name}")

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.diffusion_device)

        # Load warmup checkpoint if specified
        warmup_ckpt = self.cfg.model.get("warmup_checkpoint", None)
        if warmup_ckpt and os.path.exists(warmup_ckpt):
            logger.info(f"Loading warmup checkpoint: {warmup_ckpt}")
            transformer = SD3Transformer2DModel.from_pretrained(
                warmup_ckpt, torch_dtype=torch.float16
            ).to(self.diffusion_device)
            self.pipeline.transformer = transformer

        # NOTE: gradient checkpointing conflicts with attention hooks in SD3.5
        # JointTransformerBlocks. Disable it and rely on torch.no_grad() for
        # generation, only computing gradients during log_prob recomputation.
        # self.pipeline.transformer.enable_gradient_checkpointing()

        # Create policy wrapper
        self.policy = DiffusionPolicy(
            pipeline=self.pipeline,
            num_inference_steps=self.cfg.diffusion.num_inference_steps,
            guidance_scale=self.cfg.diffusion.guidance_scale,
        )

    def load_prompts(self) -> List[str]:
        """Load training prompts from file."""
        prompt_file = self.cfg.data.prompt_file
        if not os.path.exists(prompt_file):
            logger.warning(f"Prompt file not found: {prompt_file}. Using demo prompts.")
            return self._demo_prompts()

        with open(prompt_file, "r") as f:
            data = json.load(f)

        if isinstance(data, list):
            if isinstance(data[0], str):
                return data
            return [item["prompt"] for item in data if "prompt" in item]
        return self._demo_prompts()

    @staticmethod
    def _demo_prompts() -> List[str]:
        return [
            "three red apples and two green pears on a wooden table",
            "a black cat wearing a red hat sitting on a blue chair",
            "five yellow butterflies flying above a garden of purple flowers",
            "a transparent glass box containing a small white rabbit on top of a wooden desk",
            "two tall giraffes standing next to a short elephant under a large tree",
        ]

    def train_epoch(self, prompts: List[str]):
        """Run one epoch of SUCA training."""
        self.pipeline.transformer.train()
        epoch_metrics = {
            "loss": 0.0,
            "mean_reward": 0.0,
            "mean_uncertainty": 0.0,
            "num_steps": 0,
        }

        max_prompts = self.cfg.data.get("max_prompts_per_epoch", len(prompts))
        prompts = prompts[:max_prompts]

        for prompt_idx, prompt in enumerate(prompts):
            try:
                metrics = self.train_step(prompt)
            except Exception as e:
                logger.error(f"Error on prompt '{prompt[:50]}...': {e}")
                continue

            epoch_metrics["loss"] += metrics["loss"]
            epoch_metrics["mean_reward"] += metrics["mean_reward"]
            epoch_metrics["mean_uncertainty"] += metrics["mean_uncertainty"]
            epoch_metrics["num_steps"] += 1
            self.global_step += 1

            if self.global_step % self.cfg.logging.log_interval == 0:
                avg_loss = epoch_metrics["loss"] / epoch_metrics["num_steps"]
                avg_reward = epoch_metrics["mean_reward"] / epoch_metrics["num_steps"]
                if hasattr(self, '_wandb') and self._wandb:
                    self._wandb.log({"loss": metrics["loss"], "reward": metrics["mean_reward"],
                                     "uncertainty": metrics["mean_uncertainty"],
                                     "kl": metrics.get("kl", 0), "step": self.global_step})
                logger.info(
                    f"[Epoch {self.epoch}] Step {self.global_step} | "
                    f"Loss: {avg_loss:.4f} | Reward: {avg_reward:.4f}"
                )

            if self.global_step % self.cfg.logging.save_interval == 0:
                self.save_checkpoint_managed()

            # Periodic evaluation
            eval_interval = self.cfg.get("eval", {}).get("eval_interval", 0)
            if eval_interval > 0 and self.global_step % eval_interval == 0:
                eval_score = self.evaluate()
                if eval_score > self._best_eval_score:
                    logger.info(
                        f"New best eval score: {eval_score:.2f} "
                        f"(prev: {self._best_eval_score:.2f})"
                    )
                    self._best_eval_score = eval_score
                    self.save_best_checkpoint(eval_score)

        self.epoch += 1
        return epoch_metrics

    def train_step(self, prompt: str) -> Dict[str, float]:
        """
        Single SUCA training step for one prompt.

        Implements the full algorithm pipeline.
        """
        import sys
        print(f"[DEBUG] train_step start: '{prompt[:50]}...'", flush=True)
        timing: Dict[str, float] = {}

        stage_start = time.time()
        # ---- Step 1: Parse prompt into semantic units ----
        units = self.parser.parse(prompt)
        vqa_questions = [u.vqa_question for u in units]
        K = len(units)

        # Resolve token indices for each unit
        # SD3 has multiple tokenizers; use the first CLIP tokenizer for token mapping
        units = self.parser.resolve_token_indices(
            units, prompt, self.pipeline.tokenizer
        )
        timing["Semantic parsing"] = time.time() - stage_start

        print(f"[DEBUG] parsed {len(units)} units", flush=True)
        # ---- Step 2: Encode prompt ----
        stage_start = time.time()
        text_cond = self.policy.encode_prompt(prompt)

        # ---- Step 3: Sample images (for uncertainty estimation) ----
        N_unc = self.cfg.training.num_uncertainty_samples
        trajectories = []
        images = []

        with torch.no_grad():
            with self.attention_extractor.capture() as extractor:
                # Primary trajectory (no grad for generation; grad in log_prob recompute)
                traj = self.policy.generate_trajectory(
                    prompt,
                    text_cond=text_cond,
                    attention_extractor=extractor,
                )
                trajectories.append(traj)
                images.append(traj["image"])

        # Additional samples for uncertainty (no need for attention)
        for _ in range(N_unc - 1):
            with torch.no_grad():
                extra_traj = self.policy.generate_trajectory(
                    prompt, text_cond=text_cond
                )
                images.append(extra_traj["image"])
        timing["Image sampling"] = time.time() - stage_start

        print(f"[DEBUG] generated {len(images)} images", flush=True)
        primary_image = images[0]
        primary_traj = trajectories[0]

        # ---- Step 4 & 5: Compute unit rewards and uncertainty ----
        stage_start = time.time()
        corrected_rewards, raw_rewards, uncertainties = (
            self.reward_computer.compute_corrected_rewards(
                image=primary_image,
                images_for_uncertainty=images,
                vqa_questions=vqa_questions,
            )
        )
        timing["VLM verification"] = time.time() - stage_start

        print(f"[DEBUG] rewards computed: {raw_rewards.tolist()}", flush=True)
        # ---- Step 6: Build responsibility matrix C[k,t] ----
        stage_start = time.time()
        C = self.responsibility_builder.build(
            extractor=self.attention_extractor,
            units=units,
        )  # (K, T)

        # ---- Step 7: Compute unit advantages ----
        baseline = self._get_baselines(units)
        advantages = UnitRewardComputer.compute_unit_advantages(
            corrected_rewards, baseline=baseline
        )  # (K,)

        # Update baselines
        self._update_baselines(units, corrected_rewards)

        # ---- Step 8-10: Per-timestep gradient accumulation ----
        # Instead of building the full computation graph across all timesteps
        # (which causes OOM), we compute loss per-timestep and accumulate gradients.
        T = len(primary_traj["timesteps"])

        # Ensure C dimensions match
        if C.shape[1] != T:
            C = C[:, :T] if C.shape[1] > T else torch.nn.functional.pad(
                C, (0, T - C.shape[1]), value=1.0 / T
            )

        C_device = C.to(self.diffusion_device)
        advantages_device = advantages.to(self.diffusion_device)

        # Precompute per-timestep weights: w_t = Sum_k C_{k,t} * A_k
        weights_per_t = (C_device * advantages_device.unsqueeze(1)).sum(dim=0)  # (T,)
        timing["Routing aggregation"] = time.time() - stage_start

        print(f"[DEBUG] starting per-timestep backward ({T} steps)", flush=True)
        stage_start = time.time()
        self.optimizer.zero_grad()
        total_policy_loss = 0.0

        for t_idx in range(T):
            log_prob_t = self.policy.compute_log_prob_single(
                primary_traj, t_idx
            )
            loss_t = -weights_per_t[t_idx] * log_prob_t / T

            # Skip NaN/Inf losses
            if torch.isnan(loss_t) or torch.isinf(loss_t):
                continue
            loss_t.backward()
            total_policy_loss += loss_t.item()

        # Skip optimizer step if loss is NaN
        if not (total_policy_loss != total_policy_loss):  # NaN check
            print(f"[DEBUG] backward done, total_loss={total_policy_loss:.4f}", flush=True)
            if self.cfg.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.pipeline.transformer.parameters(),
                    self.cfg.training.max_grad_norm,
                )
            self.optimizer.step()
            print(f"[DEBUG] optimizer step done", flush=True)
        else:
            print(f"[DEBUG] skipped NaN loss", flush=True)
            self.optimizer.zero_grad()
        timing["Backward step"] = time.time() - stage_start

        policy_loss = torch.tensor(total_policy_loss)
        weighted_kl = torch.tensor(0.0)
        total_loss = policy_loss

        # ---- Logging ----
        metrics = {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": weighted_kl.item(),
            "mean_reward": raw_rewards.mean().item(),
            "mean_corrected_reward": corrected_rewards.mean().item(),
            "mean_uncertainty": uncertainties.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "num_units": K,
        }

        if self.artifact_logger is not None:
            self.artifact_logger.append_step_timing({
                "step": self.global_step + 1,
                "epoch": self.epoch,
                "prompt": prompt,
                **timing,
            })
            self.artifact_logger.append_step_metrics({
                "step": self.global_step + 1,
                "epoch": self.epoch,
                "prompt": prompt,
                **metrics,
            })
            self.artifact_logger.append_unit_rewards(
                step=self.global_step + 1,
                prompt=prompt,
                rewards_by_type=ArtifactLogger.rewards_by_unit_type(units, raw_rewards.tolist()),
                method_label="\\method",
            )

        # Save visualization periodically
        if (
            self.cfg.logging.save_images
            and self.global_step % self.cfg.logging.log_interval == 0
        ):
            self._save_step_artifacts(
                prompt, primary_image, units, C, raw_rewards, uncertainties, advantages
            )

        return metrics

    def _get_baselines(self, units: List[SemanticUnit]) -> Optional[torch.Tensor]:
        """Get running baselines for each unit."""
        if not self.cfg.suca.use_unit_baseline:
            return None
        baselines = []
        for u in units:
            key = f"{u.unit_type.value}:{u.description}"
            baselines.append(self._unit_baselines.get(key, 0.0))
        return torch.tensor(baselines, dtype=torch.float32)

    def _update_baselines(
        self, units: List[SemanticUnit], rewards: torch.Tensor
    ):
        """Update running baselines with exponential moving average."""
        if not self.cfg.suca.use_unit_baseline:
            return
        m = self._baseline_momentum
        for i, u in enumerate(units):
            key = f"{u.unit_type.value}:{u.description}"
            old = self._unit_baselines.get(key, rewards[i].item())
            self._unit_baselines[key] = m * old + (1 - m) * rewards[i].item()

    def _save_step_artifacts(
        self,
        prompt: str,
        image: Image.Image,
        units: List[SemanticUnit],
        C: torch.Tensor,
        rewards: torch.Tensor,
        uncertainties: torch.Tensor,
        advantages: torch.Tensor,
    ):
        """Save images, responsibility matrix, and metrics for inspection."""
        out_dir = Path(self.cfg.logging.output_dir) / f"step_{self.global_step}"
        out_dir.mkdir(parents=True, exist_ok=True)

        image.save(out_dir / "generated.png")

        # Save responsibility matrix visualization
        ResponsibilityMatrixBuilder.visualize(
            C, units, save_path=str(out_dir / "responsibility_matrix.png")
        )

        # Save metrics
        info = {
            "prompt": prompt,
            "units": [
                {
                    "type": u.unit_type.value,
                    "description": u.description,
                    "vqa_question": u.vqa_question,
                    "reward": rewards[i].item(),
                    "uncertainty": uncertainties[i].item(),
                    "advantage": advantages[i].item(),
                }
                for i, u in enumerate(units)
            ],
        }
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(info, f, indent=2)

    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(
                self.cfg.logging.output_dir,
                f"checkpoint_step{self.global_step}",
            )
        os.makedirs(path, exist_ok=True)

        # Save transformer weights
        self.pipeline.transformer.save_pretrained(os.path.join(path, "transformer"))

        # Save optimizer state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "unit_baselines": self._unit_baselines,
            },
            os.path.join(path, "trainer_state.pt"),
        )

        # Save config
        OmegaConf.save(self.cfg, os.path.join(path, "config.yaml"))
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        from diffusers import SD3Transformer2DModel

        transformer_path = os.path.join(path, "transformer")
        if os.path.exists(transformer_path):
            self.pipeline.transformer = SD3Transformer2DModel.from_pretrained(
                transformer_path, torch_dtype=torch.float16
            ).to(self.device)
            self.policy.transformer = self.pipeline.transformer

        state_path = os.path.join(path, "trainer_state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state["optimizer"])
            self.global_step = state["global_step"]
            self.epoch = state["epoch"]
            self._unit_baselines = state.get("unit_baselines", {})

        logger.info(f"Checkpoint loaded from {path}")

    # ----------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self) -> float:
        """
        Quick evaluation on a small GenEval2 subset.

        Generates images for N prompts, runs VQA with the reward model,
        and computes soft_tifa_gm score. Returns the score (0-100).
        """
        eval_cfg = self.cfg.get("eval", {})
        benchmark_file = eval_cfg.get("benchmark_file", "data/geneval2/geneval2_data.jsonl")
        num_eval = eval_cfg.get("num_eval_prompts", 50)

        if not os.path.exists(benchmark_file):
            logger.warning(f"Eval benchmark not found: {benchmark_file}")
            return 0.0

        logger.info(f"[Eval] Running evaluation on {num_eval} GenEval2 prompts...")
        self.pipeline.transformer.eval()

        benchmark_data = []
        with open(benchmark_file) as f:
            for line in f:
                benchmark_data.append(json.loads(line))
        # Use a fixed subset for consistency
        eval_data = benchmark_data[:num_eval]

        all_score_lists = []
        for i, d in enumerate(eval_data):
            prompt = d["prompt"]
            vqa_list = d["vqa_list"]

            try:
                # Generate image
                text_cond = self.policy.encode_prompt(prompt)
                traj = self.policy.generate_trajectory(prompt, text_cond=text_cond)
                image = traj["image"]

                # Evaluate each VQA question
                score_list = []
                for question, answer in vqa_list:
                    if question.startswith("How many"):
                        # For counting, check if answer matches
                        vqa_q = f"{question} Answer in one word."
                    else:
                        vqa_q = f"{question} Answer in one word."

                    prob = self.reward_computer._vqa_yes_probability(image, vqa_q)
                    score_list.append(prob)

                all_score_lists.append(score_list)
            except Exception as e:
                logger.error(f"[Eval] Error on prompt {i}: {e}")
                continue

        if not all_score_lists:
            logger.warning("[Eval] No successful evaluations")
            self.pipeline.transformer.train()
            return 0.0

        # Compute soft_tifa_gm
        from scipy.stats import gmean
        per_prompt_scores = [gmean(s) if min(s) > 0 else 0.0 for s in all_score_lists]
        total_score = 100 * sum(per_prompt_scores) / len(per_prompt_scores)

        # Per-skill breakdown
        skill_counts = {}
        for d, sl in zip(eval_data[:len(all_score_lists)], all_score_lists):
            for skill, score in zip(d.get("skills", []), sl):
                if skill not in skill_counts:
                    skill_counts[skill] = []
                skill_counts[skill].append(score)

        skill_str = " | ".join(
            f"{s}: {100*sum(v)/len(v):.1f}"
            for s, v in sorted(skill_counts.items())
        )

        logger.info(
            f"[Eval] Step {self.global_step} | "
            f"soft_tifa_gm: {total_score:.2f} | {skill_str} | "
            f"n={len(all_score_lists)}"
        )

        if hasattr(self, "_wandb") and self._wandb:
            log_dict = {
                "eval/soft_tifa_gm": total_score,
                "eval/num_success": len(all_score_lists),
                "step": self.global_step,
            }
            for s, v in skill_counts.items():
                log_dict[f"eval/{s}"] = 100 * sum(v) / len(v)
            self._wandb.log(log_dict)

        self.pipeline.transformer.train()

        if self.artifact_logger is not None:
            self.artifact_logger.append_step_metrics({
                "step": self.global_step,
                "eval_soft_tifa_gm": total_score,
                "eval_num_success": len(all_score_lists),
                "type": "evaluation",
            })
        return total_score

    # ----------------------------------------------------------------
    # Checkpoint management: keep last N + best
    # ----------------------------------------------------------------
    def save_checkpoint_managed(self):
        """Save checkpoint and clean up old ones (keep last N + best)."""
        path = os.path.join(
            self.cfg.logging.output_dir,
            f"checkpoint_step{self.global_step}",
        )
        self.save_checkpoint(path)
        self._recent_checkpoints.append(path)

        # Remove old checkpoints beyond keep_last_n
        keep_n = self.cfg.logging.get("keep_last_n", 2)
        while len(self._recent_checkpoints) > keep_n:
            old_path = self._recent_checkpoints.pop(0)
            # Don't delete if it's the best checkpoint
            if old_path != self._best_checkpoint_path and os.path.exists(old_path):
                import shutil
                shutil.rmtree(old_path)
                logger.info(f"Removed old checkpoint: {old_path}")

    def save_best_checkpoint(self, eval_score: float):
        """Save/overwrite the best checkpoint."""
        best_dir = os.path.join(self.cfg.logging.output_dir, "checkpoint_best")

        # Remove previous best if it exists
        if os.path.exists(best_dir):
            import shutil
            shutil.rmtree(best_dir)

        self.save_checkpoint(best_dir)
        self._best_checkpoint_path = best_dir

        # Save eval score alongside
        score_file = os.path.join(best_dir, "eval_score.json")
        with open(score_file, "w") as f:
            json.dump({
                "eval_score": eval_score,
                "global_step": self.global_step,
                "epoch": self.epoch,
            }, f, indent=2)

        logger.info(
            f"Best checkpoint saved: score={eval_score:.2f} step={self.global_step}"
        )

    def train(self):
        """Full training loop."""
        self.setup()

        # Init wandb
        try:
            import wandb
            wandb.init(
                project=self.cfg.logging.get("wandb_project", "suca-rl"),
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )
            self._wandb = wandb
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")
            self._wandb = None

        prompts = self.load_prompts()
        logger.info(f"Loaded {len(prompts)} training prompts.")

        for epoch in range(self.cfg.training.num_epochs):
            logger.info(f"=== Epoch {epoch + 1}/{self.cfg.training.num_epochs} ===")

            # Shuffle prompts
            import random
            random.shuffle(prompts)

            metrics = self.train_epoch(prompts)
            n = max(metrics["num_steps"], 1)
            logger.info(
                f"Epoch {epoch + 1} complete | "
                f"Avg Loss: {metrics['loss']/n:.4f} | "
                f"Avg Reward: {metrics['mean_reward']/n:.4f}"
            )

        self.save_checkpoint()
        logger.info("Training complete.")
