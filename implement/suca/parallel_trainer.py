"""
Multi-GPU SUCA Trainer with parallel generation and group gradient accumulation.

GPU Layout (8 GPUs):
  - GPU 0,1,3,4,5: SD3.5 inference copies (parallel image generation)
  - GPU 2: SD3.5 trainable copy (backward)
  - GPU 6: Qwen3-VL reward model
  - GPU 7: Reference transformer

Group training: generate group_size images in parallel, score all, accumulate
gradients, single optimizer step.
"""

import copy
import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from concurrent.futures import ThreadPoolExecutor
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model, PeftModel
from PIL import Image

from .attention_extractor import CrossAttentionExtractor
from .artifact_logger import ArtifactLogger
from .diffusion_policy import DiffusionPolicy
from .responsibility_matrix import ResponsibilityMatrixBuilder
from .semantic_parser import SemanticUnit, SemanticUnitParser
from .unit_reward import UnitRewardComputer

logger = logging.getLogger(__name__)


class ParallelSUCATrainer:
    """
    Multi-GPU SUCA trainer with parallel generation and group gradient accumulation.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.group_size = cfg.get("training", {}).get("group_size", 5)

        # Device assignments (8 GPUs: all available)
        n_gpus = torch.cuda.device_count()
        if n_gpus >= 8:
            # 8-GPU layout: maximize parallelism
            self.train_device = "cuda:0"          # trainable model (~45GB w/ grads)
            self.vlm_device = "cuda:6"            # VLM reward model 1 (~17GB)
            self.vlm_device_2 = "cuda:5"          # VLM reward model 2 (~17GB)
            self.ref_device = "cuda:7"            # ref transformer (~5GB)
            self.gen_devices = ["cuda:1", "cuda:2", "cuda:3", "cuda:4"]  # 4 inference copies
        else:
            # 4-GPU fallback
            self.train_device = "cuda:0"
            self.vlm_device = "cuda:2"
            self.vlm_device_2 = None
            self.ref_device = "cuda:3"
            self.gen_devices = ["cuda:1"]

        # Components
        self.train_pipeline = None
        self.train_policy = None
        self.gen_pipelines = []   # inference-only copies
        self.gen_policies = []
        self.ref_transformer = None
        self.parser = None
        self.reward_computer = None
        self.responsibility_builder = None
        self.attention_extractor = None
        self.optimizer = None

        # Baselines
        self._unit_baselines = {}
        self._baseline_momentum = 0.9

        # Training state
        self.global_step = 0
        self.epoch = 0

        # Checkpoint management
        self._recent_checkpoints = []
        self._best_checkpoint_path = None
        self._best_eval_score = -float("inf")

        # Wandb
        self._wandb = None
        self.artifact_logger: Optional[ArtifactLogger] = None

    def setup(self):
        """Initialize all components across GPUs."""
        from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel

        model_name = self.cfg.model.pretrained_model_name
        warmup_ckpt = self.cfg.model.get("warmup_checkpoint", None)
        num_steps = self.cfg.diffusion.num_inference_steps
        guidance = self.cfg.diffusion.guidance_scale

        logger.info(f"Setting up parallel SUCA trainer on 8 GPUs...")
        logger.info(f"  Train: {self.train_device} | VLM: {self.vlm_device} | Ref: {self.ref_device}")
        logger.info(f"  Gen: {self.gen_devices} | Group size: {self.group_size}")

        # 1. Load trainable pipeline
        logger.info("Loading trainable pipeline...")
        self.train_pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(self.train_device)

        if warmup_ckpt and os.path.exists(warmup_ckpt):
            logger.info(f"Loading warmup checkpoint: {warmup_ckpt}")
            transformer = SD3Transformer2DModel.from_pretrained(
                warmup_ckpt, torch_dtype=torch.float16
            ).to(self.train_device)
            self.train_pipeline.transformer = transformer

        # --- LoRA Setup ---
        lora_rank = self.cfg.training.get("lora_rank", 16)
        lora_alpha = self.cfg.training.get("lora_alpha", 32)
        logger.info(f"Adding LoRA adapters (rank={lora_rank}, alpha={lora_alpha})")

        # Find all linear layers in the transformer for LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0,
            bias="none",
        )
        self.train_pipeline.transformer = get_peft_model(
            self.train_pipeline.transformer, lora_config
        )
        # Cast LoRA params to float32 for stable training
        for name, param in self.train_pipeline.transformer.named_parameters():
            if param.requires_grad:
                param.data = param.data.to(torch.float32)

        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.train_pipeline.transformer, 'gradient_checkpointing_enable'):
            self.train_pipeline.transformer.gradient_checkpointing_enable()
        elif hasattr(self.train_pipeline.transformer, 'base_model'):
            base = self.train_pipeline.transformer.base_model.model
            if hasattr(base, 'gradient_checkpointing_enable'):
                base.gradient_checkpointing_enable()

        trainable = sum(p.numel() for p in self.train_pipeline.transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.train_pipeline.transformer.parameters())
        logger.info(f"LoRA trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        self.train_policy = DiffusionPolicy(
            self.train_pipeline, num_steps, guidance
        )

        # 2. Inference copy (single, on gen_device - no LoRA needed)
        logger.info(f"Creating {len(self.gen_devices)} inference copies...")
        for dev in self.gen_devices:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                model_name, torch_dtype=torch.float16
            ).to(dev)
            if warmup_ckpt and os.path.exists(warmup_ckpt):
                t = SD3Transformer2DModel.from_pretrained(
                    warmup_ckpt, torch_dtype=torch.float16
                ).to(dev)
                pipe.transformer = t
            pipe.set_progress_bar_config(disable=True)
            self.gen_pipelines.append(pipe)
            self.gen_policies.append(DiffusionPolicy(pipe, num_steps, guidance))

        # 3. Semantic parser
        self.parser = SemanticUnitParser(use_rules_only=True)

        # 4. Reward computers (2 VLMs for parallel scoring if available)
        self.reward_computer = UnitRewardComputer(
            vlm_model_name=self.cfg.model.vlm_model_name,
            device=self.vlm_device,
            lambda_u=self.cfg.suca.lambda_u,
        )
        self.reward_computer_2 = None
        if getattr(self, 'vlm_device_2', None):
            logger.info(f"Loading 2nd VLM on {self.vlm_device_2} for parallel VQA...")
            self.reward_computer_2 = UnitRewardComputer(
                vlm_model_name=self.cfg.model.vlm_model_name,
                device=self.vlm_device_2,
                lambda_u=self.cfg.suca.lambda_u,
            )

        # 5. Responsibility matrix builder
        self.responsibility_builder = ResponsibilityMatrixBuilder(
            tau=self.cfg.suca.tau,
            top_m_spatial=self.cfg.suca.top_m_spatial,
        )

        # 6. Attention extractor (on trainable model's base transformer)
        # Need to access the underlying transformer through LoRA wrapper
        base_transformer = self.train_pipeline.transformer
        if hasattr(base_transformer, 'base_model'):
            base_transformer = base_transformer.base_model.model
        self.attention_extractor = CrossAttentionExtractor(
            transformer=base_transformer,
            layer_indices=list(self.cfg.suca.attention_layers),
        )

        # 7. Reference transformer (frozen copy WITHOUT LoRA - just base weights)
        self.ref_transformer = SD3Transformer2DModel.from_pretrained(
            warmup_ckpt if warmup_ckpt and os.path.exists(warmup_ckpt) else model_name + "/transformer",
            torch_dtype=torch.float16,
        ).to(self.ref_device)
        self.ref_transformer.eval()
        for p in self.ref_transformer.parameters():
            p.requires_grad = False

        # 8. Optimizer (only LoRA params)
        lora_params = [p for p in self.train_pipeline.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.cfg.training.learning_rate,
        )
        analysis_root = self.cfg.logging.get("analysis_dir") or self.cfg.logging.output_dir
        self.artifact_logger = ArtifactLogger(analysis_root)

        # 9. Output dir
        os.makedirs(self.cfg.logging.output_dir, exist_ok=True)
        logger.info("Parallel SUCA trainer setup complete.")

    def _sync_weights_to_gen(self):
        """Sync LoRA-merged weights to all inference copies.

        Merges LoRA into base weights, copies to inference pipelines,
        then unmerges to keep LoRA trainable.
        """
        # Merge LoRA weights into base model temporarily
        self.train_pipeline.transformer.merge_adapter()
        # Get merged state dict (only base model keys)
        merged_state = {}
        for k, v in self.train_pipeline.transformer.state_dict().items():
            # Skip LoRA-specific keys
            if 'lora_' not in k and 'base_layer' not in k:
                merged_state[k] = v
        # Copy to inference pipelines
        for pipe in self.gen_pipelines:
            pipe.transformer.load_state_dict(merged_state, strict=False)
        # Unmerge to keep LoRA trainable
        self.train_pipeline.transformer.unmerge_adapter()

    def _generate_image_on_device(self, policy, prompt, device):
        """Generate a single image on a specific device (for threading)."""
        with torch.no_grad():
            text_cond = policy.encode_prompt(prompt)
            traj = policy.generate_trajectory(prompt, text_cond=text_cond)
        return traj["image"]

    def _generate_images_parallel(self, prompt: str, n: int) -> List[Image.Image]:
        """Generate n images across gen_devices (sequential to avoid CUDA threading issues)."""
        images = []
        num_gen = len(self.gen_policies)
        for i in range(n):
            policy = self.gen_policies[i % num_gen]
            img = self._generate_image_on_device(policy, prompt, self.gen_devices[i % num_gen])
            images.append(img)
        return images

    def load_prompts(self) -> List[str]:
        """Load training prompts."""
        prompt_file = self.cfg.data.prompt_file
        if not os.path.exists(prompt_file):
            return ["a red cat and a blue dog on a table"] * 10

        with open(prompt_file) as f:
            data = json.load(f)
        if isinstance(data, list):
            if isinstance(data[0], str):
                return data
            return [item["prompt"] for item in data if "prompt" in item]
        return ["a red cat and a blue dog"] * 10

    def _score_prompt_vqa(self, prompt: str, prompt_idx: int) -> Optional[Dict]:
        """Phase A: Parse + generate image + VQA score (can run in parallel threads)."""
        try:
            units = self.parser.parse(prompt)
            vqa_questions = [u.vqa_question for u in units]
            units = self.parser.resolve_token_indices(
                units, prompt, self.train_pipeline.tokenizer
            )

            # Generate images on inference GPUs (parallel)
            n_images = self.cfg.training.num_uncertainty_samples
            images = self._generate_images_parallel(prompt, min(n_images, len(self.gen_policies)))

            # VQA scoring — alternate between 2 VLMs
            rc = self.reward_computer if (prompt_idx % 2 == 0 or self.reward_computer_2 is None) else self.reward_computer_2

            # Generate a quick image for scoring (use an inference copy)
            gen_idx = prompt_idx % len(self.gen_policies)
            with torch.no_grad():
                quick_image = self.gen_policies[gen_idx].generate_trajectory(prompt)["image"]
            all_images = [quick_image] + images

            corrected_rewards, raw_rewards, uncertainties = rc.compute_corrected_rewards(
                image=quick_image, images_for_uncertainty=all_images, vqa_questions=vqa_questions,
            )

            return {
                "prompt": prompt,
                "units": units,
                "vqa_questions": vqa_questions,
                "corrected_rewards": corrected_rewards,
                "raw_rewards": raw_rewards,
                "uncertainties": uncertainties,
            }
        except Exception as e:
            logger.error(f"VQA error on '{prompt[:50]}...': {e}")
            return None

    def train_group(self, prompts_batch: List[str]) -> Dict[str, float]:
        """
        Train on a group of prompts with gradient accumulation.

        Phase A (parallel): Parse + generate + VQA for all prompts simultaneously
        Phase B (sequential): Trajectory on train device + backward for each prompt
        """
        self.train_pipeline.transformer.train()
        self.optimizer.zero_grad()

        group_metrics = {
            "loss": 0.0, "mean_reward": 0.0,
            "mean_uncertainty": 0.0, "num_valid": 0,
        }

        # Phase A: VQA scoring for all prompts (serial to avoid CUDA thread conflicts)
        scored_results = []
        for i, p in enumerate(prompts_batch):
            result = self._score_prompt_vqa(p, i)
            if result is not None:
                scored_results.append(result)

        # Phase B: Sequential trajectory + backward on train device
        for result in scored_results:
            try:
                prompt = result["prompt"]
                units = result["units"]
                corrected_rewards = result["corrected_rewards"]
                raw_rewards = result["raw_rewards"]
                uncertainties = result["uncertainties"]
                stage_start = time.time()

                # Generate trajectory on train device (with attention capture)
                with torch.no_grad():
                    with self.attention_extractor.capture() as extractor:
                        text_cond = self.train_policy.encode_prompt(prompt)
                        primary_traj = self.train_policy.generate_trajectory(
                            prompt, text_cond=text_cond, attention_extractor=extractor
                        )

                # Build responsibility matrix
                C = self.responsibility_builder.build(
                    extractor=self.attention_extractor, units=units,
                )
                T = len(primary_traj["timesteps"])
                if C.shape[1] != T:
                    C = C[:, :T] if C.shape[1] > T else torch.nn.functional.pad(
                        C, (0, T - C.shape[1]), value=1.0 / T
                    )

                # Compute advantages
                baseline = self._get_baselines(units)
                advantages = UnitRewardComputer.compute_unit_advantages(
                    corrected_rewards, baseline=baseline
                )
                self._update_baselines(units, corrected_rewards)

                # Batched backward
                C_device = C.to(self.train_device)
                adv_device = advantages.to(self.train_device)
                weights_per_t = (C_device * adv_device.unsqueeze(1)).sum(dim=0)
                routing_time = time.time() - stage_start

                bwd_bs = self.cfg.training.get("backward_batch_size", 5)
                backward_start = time.time()
                total_loss = self.train_policy.compute_weighted_loss_batched(
                    primary_traj, weights_per_t, batch_size=bwd_bs, sigma=1.0
                )
                backward_time = time.time() - backward_start

                group_metrics["loss"] += total_loss
                group_metrics["mean_reward"] += raw_rewards.mean().item()
                group_metrics["mean_uncertainty"] += uncertainties.mean().item()
                group_metrics["num_valid"] += 1

                if self.artifact_logger is not None:
                    self.artifact_logger.append_step_metrics({
                        "step": self.global_step + 1,
                        "prompt": prompt,
                        "loss": float(total_loss),
                        "mean_reward": raw_rewards.mean().item(),
                        "mean_uncertainty": uncertainties.mean().item(),
                        "type": "parallel_train",
                    })
                    self.artifact_logger.append_step_timing({
                        "step": self.global_step + 1,
                        "prompt": prompt,
                        "Routing aggregation": routing_time,
                        "Backward step": backward_time,
                    })
                    self.artifact_logger.append_unit_rewards(
                        step=self.global_step + 1,
                        prompt=prompt,
                        rewards_by_type=ArtifactLogger.rewards_by_unit_type(units, raw_rewards.tolist()),
                        method_label="\\method",
                    )

            except Exception as e:
                logger.error(f"Backward error on '{result['prompt'][:50]}...': {e}")
                continue

        # Optimizer step
        n = max(group_metrics["num_valid"], 1)
        if group_metrics["num_valid"] > 0:
            for p in self.train_pipeline.transformer.parameters():
                if p.grad is not None:
                    p.grad /= n

            if self.cfg.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.train_pipeline.transformer.parameters(),
                    self.cfg.training.max_grad_norm,
                )
            self.optimizer.step()
            self._sync_weights_to_gen()

        return group_metrics

    def _process_single_prompt(self, prompt: str, accumulate: bool = True, prompt_idx: int = 0) -> Dict:
        """Process one prompt: generate, score, backward (accumulate grads)."""
        # 1. Parse
        units = self.parser.parse(prompt)
        vqa_questions = [u.vqa_question for u in units]
        units = self.parser.resolve_token_indices(
            units, prompt, self.train_pipeline.tokenizer
        )

        # 2. Generate images in parallel (for uncertainty)
        n_images = self.cfg.training.num_uncertainty_samples
        images = self._generate_images_parallel(prompt, min(n_images, len(self.gen_policies)))

        # 3. Generate primary trajectory on train device (with attention capture)
        with torch.no_grad():
            with self.attention_extractor.capture() as extractor:
                text_cond = self.train_policy.encode_prompt(prompt)
                primary_traj = self.train_policy.generate_trajectory(
                    prompt, text_cond=text_cond, attention_extractor=extractor
                )
        primary_image = primary_traj["image"]
        all_images = [primary_image] + images

        # 4. Compute rewards (alternate between 2 VLMs if available)
        rc = self.reward_computer if (prompt_idx % 2 == 0 or self.reward_computer_2 is None) else self.reward_computer_2
        corrected_rewards, raw_rewards, uncertainties = (
            rc.compute_corrected_rewards(
                image=primary_image,
                images_for_uncertainty=all_images,
                vqa_questions=vqa_questions,
            )
        )

        # 5. Build responsibility matrix
        C = self.responsibility_builder.build(
            extractor=self.attention_extractor, units=units,
        )
        T = len(primary_traj["timesteps"])
        if C.shape[1] != T:
            C = C[:, :T] if C.shape[1] > T else torch.nn.functional.pad(
                C, (0, T - C.shape[1]), value=1.0 / T
            )

        # 6. Compute advantages
        baseline = self._get_baselines(units)
        advantages = UnitRewardComputer.compute_unit_advantages(
            corrected_rewards, baseline=baseline
        )
        self._update_baselines(units, corrected_rewards)

        # 7. Batched backward (much faster than per-timestep)
        C_device = C.to(self.train_device)
        adv_device = advantages.to(self.train_device)
        weights_per_t = (C_device * adv_device.unsqueeze(1)).sum(dim=0)

        # batch_size=20 = all timesteps in one forward+backward (no CFG in backward)
        bwd_bs = self.cfg.training.get("backward_batch_size", 20)
        total_loss = self.train_policy.compute_weighted_loss_batched(
            primary_traj, weights_per_t, batch_size=bwd_bs, sigma=1.0
        )

        return {
            "loss": total_loss,
            "mean_reward": raw_rewards.mean().item(),
            "mean_corrected_reward": corrected_rewards.mean().item(),
            "mean_uncertainty": uncertainties.mean().item(),
            "mean_advantage": advantages.mean().item(),
            "num_units": len(units),
        }

    def _get_baselines(self, units):
        if not self.cfg.suca.use_unit_baseline:
            return None
        baselines = []
        for u in units:
            key = f"{u.unit_type.value}:{u.description}"
            baselines.append(self._unit_baselines.get(key, 0.0))
        return torch.tensor(baselines, dtype=torch.float32)

    def _update_baselines(self, units, rewards):
        if not self.cfg.suca.use_unit_baseline:
            return
        m = self._baseline_momentum
        for i, u in enumerate(units):
            key = f"{u.unit_type.value}:{u.description}"
            old = self._unit_baselines.get(key, rewards[i].item())
            self._unit_baselines[key] = m * old + (1 - m) * rewards[i].item()

    @torch.no_grad()
    def evaluate(self) -> float:
        """Quick eval on GenEval2 subset."""
        eval_cfg = self.cfg.get("eval", {})
        benchmark_file = eval_cfg.get("benchmark_file", "data/geneval2/geneval2_data.jsonl")
        num_eval = eval_cfg.get("num_eval_prompts", 50)

        if not os.path.exists(benchmark_file):
            return 0.0

        logger.info(f"[Eval] Running on {num_eval} prompts...")
        self.train_pipeline.transformer.eval()

        benchmark_data = [json.loads(l) for l in open(benchmark_file)][:num_eval]
        all_score_lists = []

        for d in benchmark_data:
            try:
                # Generate on a gen device (fast)
                image = self._generate_image_on_device(
                    self.gen_policies[0], d["prompt"], self.gen_devices[0]
                )
                score_list = []
                for question, answer in d["vqa_list"]:
                    prob = self.reward_computer._vqa_yes_probability(
                        image, f"{question} Answer in one word."
                    )
                    score_list.append(prob)
                all_score_lists.append(score_list)
            except Exception as e:
                continue

        if not all_score_lists:
            self.train_pipeline.transformer.train()
            return 0.0

        from scipy.stats import gmean
        per_prompt = [gmean(s) if min(s) > 0 else 0.0 for s in all_score_lists]
        score = 100 * sum(per_prompt) / len(per_prompt)

        # Per-skill
        skill_scores = {}
        for d, sl in zip(benchmark_data[:len(all_score_lists)], all_score_lists):
            for skill, s in zip(d.get("skills", []), sl):
                skill_scores.setdefault(skill, []).append(s)

        skill_str = " | ".join(f"{s}: {100*sum(v)/len(v):.1f}" for s, v in sorted(skill_scores.items()))
        logger.info(f"[Eval] Step {self.global_step} | score: {score:.2f} | {skill_str}")

        if self._wandb:
            log_dict = {"eval/soft_tifa_gm": score, "step": self.global_step}
            for s, v in skill_scores.items():
                log_dict[f"eval/{s}"] = 100 * sum(v) / len(v)
            self._wandb.log(log_dict)

        self.train_pipeline.transformer.train()
        return score

    def save_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.cfg.logging.output_dir, f"checkpoint_step{self.global_step}")
        os.makedirs(path, exist_ok=True)
        self.train_pipeline.transformer.save_pretrained(os.path.join(path, "transformer"))
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "unit_baselines": self._unit_baselines,
        }, os.path.join(path, "trainer_state.pt"))
        OmegaConf.save(self.cfg, os.path.join(path, "config.yaml"))
        logger.info(f"Checkpoint saved: {path}")

    def save_managed(self):
        path = os.path.join(self.cfg.logging.output_dir, f"checkpoint_step{self.global_step}")
        self.save_checkpoint(path)
        self._recent_checkpoints.append(path)
        keep_n = self.cfg.logging.get("keep_last_n", 2)
        while len(self._recent_checkpoints) > keep_n:
            old = self._recent_checkpoints.pop(0)
            if old != self._best_checkpoint_path and os.path.exists(old):
                shutil.rmtree(old)
                logger.info(f"Removed old checkpoint: {old}")

    def save_best(self, score):
        best_dir = os.path.join(self.cfg.logging.output_dir, "checkpoint_best")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        self.save_checkpoint(best_dir)
        self._best_checkpoint_path = best_dir
        with open(os.path.join(best_dir, "eval_score.json"), "w") as f:
            json.dump({"score": score, "step": self.global_step, "epoch": self.epoch}, f)
        logger.info(f"Best checkpoint: score={score:.2f} step={self.global_step}")

    def train(self):
        """Full training loop with group training."""
        self.setup()

        # Wandb
        try:
            import wandb
            wandb.init(
                project=self.cfg.logging.get("wandb_project", "suca-rl"),
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )
            self._wandb = wandb
        except Exception as e:
            logger.warning(f"wandb init failed: {e}")

        prompts = self.load_prompts()
        logger.info(f"Loaded {len(prompts)} prompts. Group size: {self.group_size}")

        import random
        max_prompts = self.cfg.data.get("max_prompts_per_epoch", len(prompts))
        eval_interval = self.cfg.get("eval", {}).get("eval_interval", 100)
        save_interval = self.cfg.logging.save_interval

        for epoch in range(self.cfg.training.num_epochs):
            logger.info(f"=== Epoch {epoch + 1}/{self.cfg.training.num_epochs} ===")
            random.shuffle(prompts)
            epoch_prompts = prompts[:max_prompts]

            epoch_loss, epoch_reward, epoch_steps = 0.0, 0.0, 0

            # Process in groups
            for g_start in range(0, len(epoch_prompts), self.group_size):
                group = epoch_prompts[g_start:g_start + self.group_size]
                t0 = time.time()

                metrics = self.train_group(group)
                n = max(metrics["num_valid"], 1)
                self.global_step += 1

                epoch_loss += metrics["loss"] / n
                epoch_reward += metrics["mean_reward"] / n
                epoch_steps += 1

                elapsed = time.time() - t0
                print(
                    f"[Step {self.global_step}] loss={metrics['loss']/n:.8f} "
                    f"reward={metrics['mean_reward']/n:.4f} "
                    f"valid={metrics['num_valid']}/{len(group)} "
                    f"time={elapsed:.0f}s",
                    flush=True,
                )

                if self._wandb:
                    self._wandb.log({
                        "loss": metrics["loss"] / n,
                        "reward": metrics["mean_reward"] / n,
                        "uncertainty": metrics["mean_uncertainty"] / n,
                        "step": self.global_step,
                        "step_time": elapsed,
                    })

                # Save checkpoint
                if self.global_step % save_interval == 0:
                    self.save_managed()

                # Evaluate
                if eval_interval > 0 and self.global_step % eval_interval == 0:
                    score = self.evaluate()
                    if score > self._best_eval_score:
                        self._best_eval_score = score
                        self.save_best(score)

            self.epoch += 1
            if epoch_steps > 0:
                logger.info(
                    f"Epoch {epoch+1} done | "
                    f"Loss: {epoch_loss/epoch_steps:.4f} | "
                    f"Reward: {epoch_reward/epoch_steps:.4f}"
                )

        self.save_checkpoint()
        logger.info("Training complete.")
