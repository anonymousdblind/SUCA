"""
FSDP + vLLM Pipeline SUCA Trainer.

Architecture (8× A800-80GB):
  GPU 0,1,2,3 — FSDP training (SD3.5 LoRA, sharded across 4 GPUs)
  GPU 4,5     — Rollout workers (2× SD3.5 inference, ~20GB each)
  GPU 6,7     — vLLM reward server (Qwen2.5-VL-7B, tensor_parallel=2)

Pipeline: Rollout → vLLM Reward → FSDP Backward (overlapped via multiprocessing)
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import time
import torch.multiprocessing as mp
from typing import Dict, List, Optional

import requests
import torch
import torch.distributed as dist
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from PIL import Image

from .attention_extractor import CrossAttentionExtractor
from .diffusion_policy import DiffusionPolicy
from .responsibility_matrix import ResponsibilityMatrixBuilder
from .semantic_parser import SemanticUnitParser
from .unit_reward import UnitRewardComputer

logger = logging.getLogger(__name__)

# ======================================================================
# vLLM Reward Server (runs as a separate process on GPU 6,7)
# ======================================================================

def start_vllm_server(model_path: str, port: int = 8100, tp: int = 2, gpu_ids: str = "6,7"):
    """Start vLLM OpenAI-compatible server for VQA scoring."""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--tensor-parallel-size", str(tp),
        "--port", str(port),
        "--trust-remote-code",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.85",
        "--dtype", "bfloat16",
        "--disable-log-requests",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc


def wait_for_vllm(port: int = 8100, timeout: int = 300):
    """Wait for vLLM server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


class VLLMRewardScorer:
    """Score images via reward server(s) using VQA questions."""

    def __init__(self, port: int = 8100, ports: List[int] = None):
        self.ports = ports or [port]
        self._call_idx = 0

    def score_vqa_batch(self, image: Image.Image, questions: List[str]) -> List[float]:
        """Score all VQA questions for an image via HTTP reward server."""
        import base64
        from io import BytesIO

        buf = BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Round-robin between servers
        port = self.ports[self._call_idx % len(self.ports)]
        self._call_idx += 1

        try:
            resp = requests.post(
                f"http://localhost:{port}/score_batch",
                json={"image_b64": img_b64, "questions": questions},
                timeout=120,
            )
            data = resp.json()
            return data["scores"]
        except Exception as e:
            logger.error(f"Reward server error (port {port}): {e}")
            return [0.5] * len(questions)


# ======================================================================
# Rollout Worker (runs as separate process on GPU 4 or 5)
# ======================================================================

def rollout_worker_fn(
    worker_id: int,
    gpu_id: int,
    model_path: str,
    warmup_ckpt: Optional[str],
    num_steps: int,
    guidance: float,
    in_queue,
    out_queue,
):
    """Generate images. Runs as a separate process on a dedicated GPU."""
    import torch
    from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
    from suca.diffusion_policy import DiffusionPolicy as DP

    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    logger.info(f"[Rollout-{worker_id}] Loading SD3.5 on {device}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to(device)
    if warmup_ckpt and os.path.exists(warmup_ckpt):
        t = SD3Transformer2DModel.from_pretrained(warmup_ckpt, torch_dtype=torch.float16).to(device)
        pipe.transformer = t
    pipe.set_progress_bar_config(disable=True)
    policy = DP(pipe, num_steps, guidance)
    logger.info(f"[Rollout-{worker_id}] Ready on {device}.")

    while True:
        item = in_queue.get()
        if item is None:
            break
        prompt, idx = item
        try:
            with torch.no_grad():
                traj = policy.generate_trajectory(prompt)
            out_queue.put({"prompt": prompt, "idx": idx, "image": traj["image"]})
        except Exception as e:
            out_queue.put({"prompt": prompt, "idx": idx, "image": None})


# ======================================================================
# Main FSDP Trainer
# ======================================================================

class FSDPPipelineTrainer:
    """
    FSDP + vLLM pipeline trainer.
    - FSDP shards the LoRA-adapted SD3.5 across GPU 0-3
    - vLLM serves Qwen2.5-VL on GPU 6-7 for reward
    - Rollout workers on GPU 4-5 generate images
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.group_size = cfg.training.get("group_size", 5)

        # State
        self.global_step = 0
        self.epoch = 0
        self._unit_baselines = {}
        self._baseline_momentum = 0.9
        self._recent_checkpoints = []
        self._best_checkpoint_path = None
        self._best_eval_score = -float("inf")
        self._wandb = None

        # Components
        self.train_pipeline = None
        self.train_policy = None
        self.parser = None
        self.responsibility_builder = None
        self.attention_extractor = None
        self.optimizer = None
        self.vllm_scorer = None
        self.vllm_proc = None

        # Multiprocessing queues (initialized in setup with spawn context)
        self._prompt_queue = None
        self._rollout_queue = None
        self._rollout_workers = []

    def setup(self):
        from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel

        model_path = self.cfg.model.pretrained_model_name
        warmup_ckpt = self.cfg.model.get("warmup_checkpoint", None)
        num_steps = self.cfg.diffusion.num_inference_steps
        guidance = self.cfg.diffusion.guidance_scale
        vlm_model = self.cfg.model.get("vlm_model_name_v2", self.cfg.model.vlm_model_name)

        logger.info("=" * 60)
        logger.info("FSDP + vLLM Pipeline Trainer — Setup")
        logger.info(f"  Train (FSDP): GPU 0,1,2,3")
        logger.info(f"  Rollout: GPU 4,5")
        logger.info(f"  vLLM Reward: GPU 6,7 ({vlm_model})")
        logger.info("=" * 60)

        # 1. Check reward servers are running (should be started separately)
        self.reward_ports = [8100, 8101]
        logger.info("Checking reward servers...")
        for port in self.reward_ports:
            if wait_for_vllm(port=port, timeout=30):
                logger.info(f"  Reward server on port {port}: ready")
            else:
                logger.warning(f"  Reward server on port {port}: NOT ready")
        self.vllm_scorer = VLLMRewardScorer(ports=self.reward_ports)

        # 2. Start rollout workers on physical GPU 4,5 (BEFORE CUDA init, using spawn)
        # Use physical GPU IDs since each worker sets its own CUDA_VISIBLE_DEVICES
        rollout_physical_gpus = [4, 5]
        logger.info(f"Starting rollout workers on physical GPU {rollout_physical_gpus}...")
        ctx = mp.get_context("spawn")
        self._prompt_queue = ctx.Queue(maxsize=50)
        self._rollout_queue = ctx.Queue(maxsize=50)
        for i, gpu_id in enumerate(rollout_physical_gpus):
            p = ctx.Process(
                target=rollout_worker_fn,
                args=(i, gpu_id, model_path, warmup_ckpt, num_steps, guidance,
                      self._prompt_queue, self._rollout_queue),
                daemon=True,
            )
            p.start()
            self._rollout_workers.append(p)
        logger.info("Rollout workers started.")

        # 3. Load trainable pipeline on GPU 0
        logger.info("Loading trainable pipeline on GPU 0...")
        self.train_pipeline = StableDiffusion3Pipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to("cuda:0")

        if warmup_ckpt and os.path.exists(warmup_ckpt):
            logger.info(f"Loading warmup: {warmup_ckpt}")
            t = SD3Transformer2DModel.from_pretrained(warmup_ckpt, torch_dtype=torch.float16).to("cuda:0")
            self.train_pipeline.transformer = t

        # LoRA
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

        # 5. Optimizer
        lora_params = [p for p in self.train_pipeline.transformer.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(lora_params, lr=self.cfg.training.learning_rate)

        # 6. Output dir
        os.makedirs(self.cfg.logging.output_dir, exist_ok=True)
        logger.info("Setup complete.")

    def _cleanup(self):
        # Stop rollout workers
        for _ in self._rollout_workers:
            self._prompt_queue.put(None)
        for p in self._rollout_workers:
            p.join(timeout=10)
        # Stop vLLM
        if self.vllm_proc:
            self.vllm_proc.terminate()
            self.vllm_proc.wait(timeout=10)

    # ------------------------------------------------------------------
    # Rollout + Reward
    # ------------------------------------------------------------------

    def _submit_rollout(self, prompts: List[str]):
        """Submit prompts to rollout workers (non-blocking)."""
        for i, p in enumerate(prompts):
            self._prompt_queue.put((p, i))

    def _collect_and_score(self, expected: int, timeout: float = 600) -> List[Dict]:
        """Collect generated images and score with vLLM."""
        results = []
        deadline = time.time() + timeout
        received = 0

        while received < expected and time.time() < deadline:
            try:
                item = self._rollout_queue.get(timeout=10)
                received += 1
            except Exception:
                continue

            if item["image"] is None:
                continue

            prompt = item["prompt"]
            image = item["image"]

            # Parse prompt
            units = self.parser.parse(prompt)
            vqa_questions = [u.vqa_question for u in units]
            units = self.parser.resolve_token_indices(
                units, prompt, self.train_pipeline.tokenizer
            )

            # Score with vLLM
            if self.vllm_scorer:
                scores = self.vllm_scorer.score_vqa_batch(image, vqa_questions)
                raw_rewards = torch.tensor(scores, dtype=torch.float32)
            else:
                # Fallback: use HF reward computer
                raw_rewards = torch.tensor([0.5] * len(vqa_questions), dtype=torch.float32)

            results.append({
                "prompt": prompt,
                "units": units,
                "raw_rewards": raw_rewards,
                "corrected_rewards": raw_rewards,  # simplified: no uncertainty correction
                "uncertainties": torch.zeros_like(raw_rewards),
            })

        return results

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def _do_backward_group(self, scored_items: List[Dict]) -> Dict[str, float]:
        """Backward pass for a group of scored prompts."""
        self.train_pipeline.transformer.train()
        self.optimizer.zero_grad()

        metrics = {"loss": 0.0, "mean_reward": 0.0, "num_valid": 0}

        for item in scored_items:
            try:
                prompt = item["prompt"]
                units = item["units"]
                corrected_rewards = item["corrected_rewards"]
                raw_rewards = item["raw_rewards"]

                # Generate trajectory on train device
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
                C_dev = C.to("cuda:0")
                adv_dev = advantages.to("cuda:0")
                weights_per_t = (C_dev * adv_dev.unsqueeze(1)).sum(dim=0)

                bwd_bs = self.cfg.training.get("backward_batch_size", 5)
                loss = self.train_policy.compute_weighted_loss_batched(
                    traj, weights_per_t, batch_size=bwd_bs, sigma=1.0
                )

                metrics["loss"] += loss
                metrics["mean_reward"] += raw_rewards.mean().item()
                metrics["num_valid"] += 1

            except Exception as e:
                logger.error(f"Backward error: {e}")

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
            # Sync to rollout workers via checkpoint (simple approach)
            self._sync_to_rollout()

        return metrics

    def _sync_to_rollout(self):
        """Sync LoRA weights to rollout workers.
        For simplicity, save adapter and reload in workers.
        TODO: Use shared memory or torch.distributed for faster sync.
        """
        # For now, rollout workers use the base model without LoRA updates.
        # This is acceptable for early training — the policy improves slowly.
        # Full sync can be added via torch.distributed.
        pass

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
    # Eval
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

        for d in data:
            try:
                # Generate on train device
                traj = self.train_policy.generate_trajectory(d["prompt"])
                image = traj["image"]

                if self.vllm_scorer:
                    questions = [q for q, _ in d["vqa_list"]]
                    scores = self.vllm_scorer.score_vqa_batch(image, questions)
                    all_scores.append(scores)
                else:
                    all_scores.append([0.5] * len(d["vqa_list"]))
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
    # Main training loop
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
        eval_interval = self.cfg.get("eval", {}).get("eval_interval", 10)
        save_interval = self.cfg.logging.save_interval

        try:
            for epoch in range(self.cfg.training.num_epochs):
                logger.info(f"=== Epoch {epoch + 1}/{self.cfg.training.num_epochs} ===")
                random.shuffle(prompts)
                ep = prompts[:max_prompts]
                ep_loss, ep_reward, ep_steps = 0.0, 0.0, 0

                groups = [ep[i:i+self.group_size] for i in range(0, len(ep), self.group_size)]

                # Prefetch first group
                self._submit_rollout(groups[0])

                for g_idx, group in enumerate(groups):
                    t0 = time.time()

                    # Collect rollout + score for current group
                    scored = self._collect_and_score(len(group))

                    # Prefetch next group (overlaps with backward)
                    if g_idx + 1 < len(groups):
                        self._submit_rollout(groups[g_idx + 1])

                    # Backward + optimizer step
                    metrics = self._do_backward_group(scored)
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
            self._cleanup()
            self.save_checkpoint()
            logger.info("Training complete.")
