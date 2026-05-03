"""
Pipeline SUCA Trainer — 3-stage decoupled architecture.

GPU Layout (8× A800-80GB):
  Stage 1 — Rollout:   GPU 1,2,3,4  (4× SD3.5 inference, ~20GB each)
  Stage 2 — Reward:    GPU 5,6      (2× Qwen3-VL VQA judge, ~17GB each)
  Stage 3 — Trainer:   GPU 0        (SD3.5 LoRA backward, ~45GB)
  Shared  — Ref:       GPU 7        (frozen reference transformer, ~5GB)

Data flows through queues:
  prompts → [Rollout] → rollout_queue → [Reward] → reward_queue → [Trainer]

The 3 stages run concurrently as threads (sharing CUDA context per-GPU).
Rollout and Reward stages prefetch while Trainer does backward.
"""

import copy
import json
import logging
import os
import queue
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from PIL import Image

from .attention_extractor import CrossAttentionExtractor
from .diffusion_policy import DiffusionPolicy
from .responsibility_matrix import ResponsibilityMatrixBuilder
from .semantic_parser import SemanticUnitParser
from .unit_reward import UnitRewardComputer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Worker functions (each runs in its own thread, pinned to specific GPUs)
# ---------------------------------------------------------------------------

def _rollout_worker(
    worker_id: int,
    device: str,
    policy: DiffusionPolicy,
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    stop_event: threading.Event,
):
    """Generate images from prompts. Runs on a dedicated rollout GPU."""
    torch.cuda.set_device(device)
    while not stop_event.is_set():
        try:
            item = in_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:  # poison pill
            break
        prompt, idx = item
        try:
            with torch.no_grad():
                traj = policy.generate_trajectory(prompt)
            out_queue.put({
                "prompt": prompt,
                "idx": idx,
                "image": traj["image"],
                "worker": worker_id,
            })
        except Exception as e:
            logger.error(f"[Rollout-{worker_id}] Error: {e}")
            out_queue.put({"prompt": prompt, "idx": idx, "image": None, "worker": worker_id})


def _reward_worker(
    worker_id: int,
    reward_computer: UnitRewardComputer,
    parser: SemanticUnitParser,
    tokenizer,
    in_queue: queue.Queue,
    out_queue: queue.Queue,
    stop_event: threading.Event,
):
    """Score images with VLM. Runs on a dedicated reward GPU."""
    while not stop_event.is_set():
        try:
            item = in_queue.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            break
        prompt = item["prompt"]
        idx = item["idx"]
        image = item["image"]
        if image is None:
            out_queue.put({"prompt": prompt, "idx": idx, "scored": None})
            continue
        try:
            units = parser.parse(prompt)
            vqa_questions = [u.vqa_question for u in units]
            units = parser.resolve_token_indices(units, prompt, tokenizer)

            corrected_rewards, raw_rewards, uncertainties = (
                reward_computer.compute_corrected_rewards(
                    image=image,
                    images_for_uncertainty=[image],
                    vqa_questions=vqa_questions,
                )
            )
            out_queue.put({
                "prompt": prompt,
                "idx": idx,
                "scored": {
                    "units": units,
                    "corrected_rewards": corrected_rewards,
                    "raw_rewards": raw_rewards,
                    "uncertainties": uncertainties,
                },
            })
        except Exception as e:
            logger.error(f"[Reward-{worker_id}] Error on '{prompt[:40]}..': {e}")
            out_queue.put({"prompt": prompt, "idx": idx, "scored": None})


# ---------------------------------------------------------------------------
# Main Trainer
# ---------------------------------------------------------------------------

class PipelineSUCATrainer:
    """
    3-stage pipeline trainer: Rollout → Reward → Train.

    Rollout and Reward stages prefetch the NEXT group while Trainer processes
    the CURRENT group, achieving ~2× throughput compared to sequential.
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.group_size = cfg.training.get("group_size", 5)

        n_gpus = torch.cuda.device_count()
        assert n_gpus >= 4, f"Need ≥4 GPUs, got {n_gpus}"

        if n_gpus >= 8:
            self.train_device = "cuda:0"
            self.rollout_devices = ["cuda:1", "cuda:2", "cuda:3", "cuda:4"]
            self.reward_devices = ["cuda:5", "cuda:6"]
            self.ref_device = "cuda:7"
        else:
            self.train_device = "cuda:0"
            self.rollout_devices = ["cuda:1"]
            self.reward_devices = ["cuda:2"]
            self.ref_device = "cuda:3"

        # State
        self.global_step = 0
        self.epoch = 0
        self._unit_baselines = {}
        self._baseline_momentum = 0.9
        self._recent_checkpoints = []
        self._best_checkpoint_path = None
        self._best_eval_score = -float("inf")
        self._wandb = None

        # Components (filled in setup)
        self.train_pipeline = None
        self.train_policy = None
        self.rollout_policies = []
        self.reward_computers = []
        self.parser = None
        self.responsibility_builder = None
        self.attention_extractor = None
        self.optimizer = None

        # Pipeline queues & threads
        self._prompt_queue = queue.Queue(maxsize=50)
        self._rollout_queue = queue.Queue(maxsize=50)
        self._reward_queue = queue.Queue(maxsize=50)
        self._stop_event = threading.Event()
        self._workers = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel

        model_name = self.cfg.model.pretrained_model_name
        warmup_ckpt = self.cfg.model.get("warmup_checkpoint", None)
        num_steps = self.cfg.diffusion.num_inference_steps
        guidance = self.cfg.diffusion.guidance_scale

        logger.info("=" * 60)
        logger.info("Pipeline SUCA Trainer — Setup")
        logger.info(f"  Train:   {self.train_device}")
        logger.info(f"  Rollout: {self.rollout_devices}")
        logger.info(f"  Reward:  {self.reward_devices}")
        logger.info(f"  Ref:     {self.ref_device}")
        logger.info("=" * 60)

        # 1. Trainable pipeline (GPU 0) with LoRA
        logger.info("Loading trainable pipeline...")
        self.train_pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(self.train_device)

        if warmup_ckpt and os.path.exists(warmup_ckpt):
            logger.info(f"Loading warmup: {warmup_ckpt}")
            t = SD3Transformer2DModel.from_pretrained(warmup_ckpt, torch_dtype=torch.float16).to(self.train_device)
            self.train_pipeline.transformer = t

        lora_rank = self.cfg.training.get("lora_rank", 64)
        lora_alpha = self.cfg.training.get("lora_alpha", 128)
        lora_cfg = LoraConfig(
            r=lora_rank, lora_alpha=lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            lora_dropout=0.0, bias="none",
        )
        self.train_pipeline.transformer = get_peft_model(self.train_pipeline.transformer, lora_cfg)
        for _, p in self.train_pipeline.transformer.named_parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)

        trainable = sum(p.numel() for p in self.train_pipeline.transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.train_pipeline.transformer.parameters())
        logger.info(f"LoRA: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        self.train_policy = DiffusionPolicy(self.train_pipeline, num_steps, guidance)

        # 2. Rollout pipelines (GPU 1–4)
        logger.info(f"Loading {len(self.rollout_devices)} rollout workers...")
        for dev in self.rollout_devices:
            pipe = StableDiffusion3Pipeline.from_pretrained(model_name, torch_dtype=torch.float16).to(dev)
            if warmup_ckpt and os.path.exists(warmup_ckpt):
                t = SD3Transformer2DModel.from_pretrained(warmup_ckpt, torch_dtype=torch.float16).to(dev)
                pipe.transformer = t
            pipe.set_progress_bar_config(disable=True)
            self.rollout_policies.append(DiffusionPolicy(pipe, num_steps, guidance))

        # 3. Reward computers (GPU 5–6)
        logger.info(f"Loading {len(self.reward_devices)} reward workers...")
        for dev in self.reward_devices:
            rc = UnitRewardComputer(
                vlm_model_name=self.cfg.model.vlm_model_name,
                device=dev,
                lambda_u=self.cfg.suca.lambda_u,
            )
            self.reward_computers.append(rc)

        # 4. Parser, responsibility, attention
        self.parser = SemanticUnitParser(use_rules_only=True)
        self.responsibility_builder = ResponsibilityMatrixBuilder(
            tau=self.cfg.suca.tau,
            top_m_spatial=self.cfg.suca.top_m_spatial,
        )
        base_t = self.train_pipeline.transformer
        if hasattr(base_t, 'base_model'):
            base_t = base_t.base_model.model
        self.attention_extractor = CrossAttentionExtractor(
            transformer=base_t,
            layer_indices=list(self.cfg.suca.attention_layers),
        )

        # 5. Reference transformer (GPU 7)
        ref_path = warmup_ckpt if warmup_ckpt and os.path.exists(warmup_ckpt) else model_name + "/transformer"
        self.ref_transformer = SD3Transformer2DModel.from_pretrained(
            ref_path, torch_dtype=torch.float16,
        ).to(self.ref_device).eval()
        for p in self.ref_transformer.parameters():
            p.requires_grad = False

        # 6. Optimizer
        lora_params = [p for p in self.train_pipeline.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(lora_params, lr=self.cfg.training.learning_rate)

        # 7. Output dir
        os.makedirs(self.cfg.logging.output_dir, exist_ok=True)

        # 8. Start worker threads
        self._start_workers()
        logger.info("Pipeline setup complete.")

    def _start_workers(self):
        """Launch rollout and reward worker threads."""
        # Rollout workers
        for i, dev in enumerate(self.rollout_devices):
            t = threading.Thread(
                target=_rollout_worker,
                args=(i, dev, self.rollout_policies[i],
                      self._prompt_queue, self._rollout_queue, self._stop_event),
                daemon=True,
            )
            t.start()
            self._workers.append(t)

        # Reward workers
        for i, rc in enumerate(self.reward_computers):
            t = threading.Thread(
                target=_reward_worker,
                args=(i, rc, self.parser, self.train_pipeline.tokenizer,
                      self._rollout_queue, self._reward_queue, self._stop_event),
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def _stop_workers(self):
        self._stop_event.set()
        for _ in self.rollout_policies:
            self._prompt_queue.put(None)
        for _ in self.reward_computers:
            self._rollout_queue.put(None)
        for t in self._workers:
            t.join(timeout=10)

    # ------------------------------------------------------------------
    # Weight sync
    # ------------------------------------------------------------------

    def _sync_weights_to_rollout(self):
        """Copy LoRA-merged weights to all rollout pipelines."""
        self.train_pipeline.transformer.merge_adapter()
        merged = {}
        for k, v in self.train_pipeline.transformer.state_dict().items():
            if 'lora_' not in k and 'base_layer' not in k:
                merged[k] = v
        for pol in self.rollout_policies:
            pol.pipeline.transformer.load_state_dict(merged, strict=False)
        self.train_pipeline.transformer.unmerge_adapter()

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def prefetch_group(self, prompts: List[str]):
        """Submit prompts to the rollout pipeline (non-blocking).
        Call this BEFORE processing the current group so that rollout+reward
        happen in parallel with the current group's backward."""
        for i, p in enumerate(prompts):
            self._prompt_queue.put((p, i))

    def collect_scored(self, expected: int, timeout: float = 600) -> List[Dict]:
        """Collect scored results from the reward queue."""
        scored = []
        deadline = time.time() + timeout
        received = 0
        while received < expected and time.time() < deadline:
            try:
                item = self._reward_queue.get(timeout=5.0)
                received += 1
                if item["scored"] is not None:
                    scored.append(item)
            except queue.Empty:
                continue
        return scored

    def _do_backward_for_item(self, item: Dict) -> Optional[Dict]:
        """Run trajectory + backward for a single scored prompt on train device."""
        prompt = item["prompt"]
        data = item["scored"]
        units = data["units"]
        corrected_rewards = data["corrected_rewards"]
        raw_rewards = data["raw_rewards"]
        uncertainties = data["uncertainties"]

        # Generate trajectory on TRAIN device (with attention)
        with torch.no_grad():
            with self.attention_extractor.capture() as ext:
                text_cond = self.train_policy.encode_prompt(prompt)
                traj = self.train_policy.generate_trajectory(
                    prompt, text_cond=text_cond, attention_extractor=ext
                )

        # Responsibility matrix
        C = self.responsibility_builder.build(extractor=self.attention_extractor, units=units)
        T = len(traj["timesteps"])
        if C.shape[1] != T:
            C = C[:, :T] if C.shape[1] > T else torch.nn.functional.pad(C, (0, T - C.shape[1]), value=1.0/T)

        # Advantages
        baseline = self._get_baselines(units)
        advantages = UnitRewardComputer.compute_unit_advantages(corrected_rewards, baseline=baseline)
        self._update_baselines(units, corrected_rewards)

        # Batched backward
        C_dev = C.to(self.train_device)
        adv_dev = advantages.to(self.train_device)
        weights_per_t = (C_dev * adv_dev.unsqueeze(1)).sum(dim=0)

        bwd_bs = self.cfg.training.get("backward_batch_size", 5)
        loss = self.train_policy.compute_weighted_loss_batched(
            traj, weights_per_t, batch_size=bwd_bs, sigma=1.0
        )
        return {
            "loss": loss,
            "mean_reward": raw_rewards.mean().item(),
            "mean_uncertainty": uncertainties.mean().item(),
        }

    def train_group_streaming(self, scored_items: List[Dict]) -> Dict[str, float]:
        """
        Process pre-collected scored items: backward + optimizer step.
        Scored items come from a PREVIOUS prefetch, so rollout+reward
        for the NEXT group runs concurrently with this backward.
        """
        self.train_pipeline.transformer.train()
        self.optimizer.zero_grad()

        metrics = {"loss": 0.0, "mean_reward": 0.0, "mean_uncertainty": 0.0, "num_valid": 0}

        for item in scored_items:
            try:
                result = self._do_backward_for_item(item)
                if result:
                    metrics["loss"] += result["loss"]
                    metrics["mean_reward"] += result["mean_reward"]
                    metrics["mean_uncertainty"] += result["mean_uncertainty"]
                    metrics["num_valid"] += 1
            except Exception as e:
                logger.error(f"[Train] Error on '{item['prompt'][:40]}..': {e}")

        # Optimizer step
        n = max(metrics["num_valid"], 1)
        if metrics["num_valid"] > 0:
            for p in self.train_pipeline.transformer.parameters():
                if p.grad is not None:
                    p.grad /= n
            if self.cfg.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.train_pipeline.transformer.parameters(),
                    self.cfg.training.max_grad_norm,
                )
            self.optimizer.step()
            self._sync_weights_to_rollout()

        return metrics

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    def _get_baselines(self, units):
        if not self.cfg.suca.use_unit_baseline:
            return None
        return torch.tensor(
            [self._unit_baselines.get(f"{u.unit_type.value}:{u.description}", 0.0) for u in units],
            dtype=torch.float32,
        )

    def _update_baselines(self, units, rewards):
        if not self.cfg.suca.use_unit_baseline:
            return
        m = self._baseline_momentum
        for i, u in enumerate(units):
            key = f"{u.unit_type.value}:{u.description}"
            old = self._unit_baselines.get(key, rewards[i].item())
            self._unit_baselines[key] = m * old + (1 - m) * rewards[i].item()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> float:
        eval_cfg = self.cfg.get("eval", {})
        benchmark_file = eval_cfg.get("benchmark_file", "data/geneval2/geneval2_data.jsonl")
        num_eval = eval_cfg.get("num_eval_prompts", 50)

        if not os.path.exists(benchmark_file):
            return 0.0

        logger.info(f"[Eval] Running {num_eval} prompts...")
        self.train_pipeline.transformer.eval()

        data = [json.loads(l) for l in open(benchmark_file)][:num_eval]
        all_scores = []
        rc = self.reward_computers[0]

        for d in data:
            try:
                pol = self.rollout_policies[0]
                image = pol.generate_trajectory(d["prompt"])["image"]
                scores = []
                for question, answer in d["vqa_list"]:
                    prob = rc._vqa_yes_probability(image, f"{question} Answer in one word.")
                    scores.append(prob)
                all_scores.append(scores)
            except Exception:
                continue

        if not all_scores:
            self.train_pipeline.transformer.train()
            return 0.0

        from scipy.stats import gmean
        per_prompt = [gmean(s) if min(s) > 0 else 0.0 for s in all_scores]
        score = 100 * sum(per_prompt) / len(per_prompt)

        skill_scores = {}
        for d, sl in zip(data[:len(all_scores)], all_scores):
            for skill, s in zip(d.get("skills", []), sl):
                skill_scores.setdefault(skill, []).append(s)

        skill_str = " | ".join(f"{s}:{100*sum(v)/len(v):.1f}" for s, v in sorted(skill_scores.items()))
        logger.info(f"[Eval] Step {self.global_step} | score={score:.2f} | {skill_str}")

        if self._wandb:
            log_dict = {"eval/soft_tifa_gm": score, "step": self.global_step}
            for s, v in skill_scores.items():
                log_dict[f"eval/{s}"] = 100 * sum(v) / len(v)
            self._wandb.log(log_dict)

        self.train_pipeline.transformer.train()
        return score

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

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
        logger.info(f"Saved: {path}")

    def save_managed(self):
        path = os.path.join(self.cfg.logging.output_dir, f"checkpoint_step{self.global_step}")
        self.save_checkpoint(path)
        self._recent_checkpoints.append(path)
        keep_n = self.cfg.logging.get("keep_last_n", 2)
        while len(self._recent_checkpoints) > keep_n:
            old = self._recent_checkpoints.pop(0)
            if old != self._best_checkpoint_path and os.path.exists(old):
                shutil.rmtree(old)

    def save_best(self, score):
        best_dir = os.path.join(self.cfg.logging.output_dir, "checkpoint_best")
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        self.save_checkpoint(best_dir)
        self._best_checkpoint_path = best_dir
        with open(os.path.join(best_dir, "eval_score.json"), "w") as f:
            json.dump({"score": score, "step": self.global_step, "epoch": self.epoch}, f)
        logger.info(f"Best: score={score:.2f} step={self.global_step}")

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def load_prompts(self) -> List[str]:
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

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self):
        self.setup()

        try:
            import wandb
            wandb.init(
                project=self.cfg.logging.get("wandb_project", "suca-rl"),
                config=OmegaConf.to_container(self.cfg, resolve=True),
            )
            self._wandb = wandb
        except Exception as e:
            logger.warning(f"wandb failed: {e}")

        prompts = self.load_prompts()
        logger.info(f"Loaded {len(prompts)} prompts. group_size={self.group_size}")

        import random
        max_prompts = self.cfg.data.get("max_prompts_per_epoch", len(prompts))
        eval_interval = self.cfg.get("eval", {}).get("eval_interval", 50)
        save_interval = self.cfg.logging.save_interval

        try:
            for epoch in range(self.cfg.training.num_epochs):
                logger.info(f"=== Epoch {epoch + 1}/{self.cfg.training.num_epochs} ===")
                random.shuffle(prompts)
                ep = prompts[:max_prompts]
                ep_loss, ep_reward, ep_steps = 0.0, 0.0, 0

                # Build list of groups
                groups = [ep[i:i+self.group_size] for i in range(0, len(ep), self.group_size)]

                # Prefetch first group
                self.prefetch_group(groups[0])

                for g_idx, group in enumerate(groups):
                    t0 = time.time()

                    # Collect scored results for CURRENT group
                    # (rollout+reward already running from prefetch)
                    scored = self.collect_scored(len(group))

                    # Prefetch NEXT group into pipeline (overlaps with backward below)
                    if g_idx + 1 < len(groups):
                        self.prefetch_group(groups[g_idx + 1])

                    # Backward + optimizer step for current group
                    metrics = self.train_group_streaming(scored)
                    n = max(metrics["num_valid"], 1)
                    self.global_step += 1
                    ep_loss += metrics["loss"] / n
                    ep_reward += metrics["mean_reward"] / n
                    ep_steps += 1

                    elapsed = time.time() - t0
                    print(
                        f"[Step {self.global_step}] "
                        f"loss={metrics['loss']/n:.8f} "
                        f"reward={metrics['mean_reward']/n:.4f} "
                        f"valid={metrics['num_valid']}/{len(group)} "
                        f"time={elapsed:.0f}s",
                        flush=True,
                    )
                    if self._wandb:
                        self._wandb.log({
                            "loss": metrics["loss"] / n,
                            "reward": metrics["mean_reward"] / n,
                            "step": self.global_step,
                            "step_time": elapsed,
                        })

                    if self.global_step % save_interval == 0:
                        self.save_managed()
                    if eval_interval > 0 and self.global_step % eval_interval == 0:
                        score = self.evaluate()
                        if score > self._best_eval_score:
                            self._best_eval_score = score
                            self.save_best(score)

                self.epoch += 1
                if ep_steps:
                    logger.info(f"Epoch {epoch+1} | loss={ep_loss/ep_steps:.4f} reward={ep_reward/ep_steps:.4f}")

        finally:
            self._stop_workers()
            self.save_checkpoint()
            logger.info("Training complete.")
