"""
Single-GPU SUCA Trainer with HTTP rollout/reward pipeline.

Rollout: 4× HTTP workers on ports 8200-8203
Reward:  2× vLLM servers on ports 8100-8101
Train:   GPU 0, LoRA on SD3.5
"""
import base64
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from typing import Dict, List, Optional

import requests
import torch
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from PIL import Image

from suca.attention_extractor import CrossAttentionExtractor
from suca.diffusion_policy import DiffusionPolicy
from suca.responsibility_matrix import ResponsibilityMatrixBuilder
from suca.semantic_parser import SemanticUnitParser
from suca.unit_reward import UnitRewardComputer

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("suca-trainer")


# ======================================================================
# HTTP Clients
# ======================================================================

class RolloutPool:
    """Round-robin across multiple rollout HTTP workers."""
    def __init__(self, ports=(8200, 8201, 8202, 8203)):
        self.ports = list(ports)
        self._idx = 0

    def generate(self, prompt: str) -> Optional[Image.Image]:
        port = self.ports[self._idx % len(self.ports)]
        self._idx += 1
        try:
            resp = requests.post(
                f"http://localhost:{port}/generate",
                json={"prompt": prompt}, timeout=120,
            )
            data = resp.json()
            return Image.open(BytesIO(base64.b64decode(data["image_b64"]))).convert("RGB")
        except Exception as e:
            logger.warning(f"Rollout:{port} error: {e}")
            return None

    def generate_batch(self, prompts: List[str]) -> List[Optional[Image.Image]]:
        """Generate images for multiple prompts in parallel using thread pool."""
        with ThreadPoolExecutor(max_workers=len(self.ports)) as ex:
            futures = [ex.submit(self.generate, p) for p in prompts]
            return [f.result() for f in futures]

    def sync_weights(self, adapter_path: str):
        for port in self.ports:
            try:
                requests.post(f"http://localhost:{port}/sync_weights",
                              json={"adapter_path": adapter_path}, timeout=30)
            except Exception:
                pass


class RewardPool:
    """Round-robin across multiple vLLM reward servers."""
    def __init__(self, ports=(8100, 8101), model_name="Qwen2.5-VL-7B-Instruct"):
        self.ports = list(ports)
        self.model_name = model_name
        self._idx = 0

    def score(self, image: Image.Image, questions: List[str]) -> List[float]:
        port = self.ports[self._idx % len(self.ports)]
        self._idx += 1
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        scores = []
        for q in questions:
            try:
                resp = requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text", "text": f"{q} Answer in one word."},
                        ]}],
                        "max_tokens": 5, "temperature": 0.0,
                        "logprobs": True, "top_logprobs": 5,
                    },
                    timeout=60,
                )
                data = resp.json()
                choice = data["choices"][0]
                text = choice["message"]["content"].strip().lower()
                yes_prob = 0.5
                if choice.get("logprobs") and choice["logprobs"].get("content"):
                    top = choice["logprobs"]["content"][0]["top_logprobs"]
                    yes_lp, no_lp = -10, -10
                    for t in top:
                        tok = t["token"].strip().lower()
                        if tok in ("yes", "true"):
                            yes_lp = max(yes_lp, t["logprob"])
                        elif tok in ("no", "false"):
                            no_lp = max(no_lp, t["logprob"])
                    if yes_lp > -10 or no_lp > -10:
                        yes_prob = math.exp(yes_lp) / (math.exp(yes_lp) + math.exp(no_lp))
                elif "yes" in text:
                    yes_prob = 0.9
                elif "no" in text:
                    yes_prob = 0.1
                scores.append(yes_prob)
            except Exception as e:
                scores.append(0.5)
        return scores

    def score_parallel(self, images_and_questions: List[tuple]) -> List[List[float]]:
        """Score multiple (image, questions) pairs in parallel."""
        with ThreadPoolExecutor(max_workers=len(self.ports)) as ex:
            futures = [ex.submit(self.score, img, qs) for img, qs in images_and_questions]
            return [f.result() for f in futures]


# ======================================================================
# Trainer
# ======================================================================

def main():
    cfg = OmegaConf.load("config/default.yaml")

    # Overrides
    group_size = 5
    num_epochs = 10
    lr = 1e-4
    max_grad_norm = 1.0
    lora_rank = 64
    lora_alpha = 128
    bwd_batch_size = 5
    num_steps = 15
    guidance = 4.5
    max_prompts = 500
    save_interval = 50
    eval_interval = 10
    outdir = "outputs/rl"
    os.makedirs(outdir, exist_ok=True)

    device = "cuda:0"

    # ── Load pipeline ──
    logger.info("Loading SD3.5 pipeline...")
    model_path = cfg.model.pretrained_model_name
    warmup_ckpt = cfg.model.get("warmup_checkpoint", None)
    pipe = StableDiffusion3Pipeline.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    if warmup_ckpt and os.path.exists(warmup_ckpt):
        logger.info(f"Loading warmup: {warmup_ckpt}")
        t = SD3Transformer2DModel.from_pretrained(warmup_ckpt, torch_dtype=torch.float16).to(device)
        pipe.transformer = t

    # ── LoRA ──
    lora_cfg = LoraConfig(
        r=lora_rank, lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0, bias="none",
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_cfg)
    for _, p in pipe.transformer.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    trainable = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pipe.transformer.parameters())
    logger.info(f"LoRA: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    train_policy = DiffusionPolicy(pipe, num_steps, guidance)

    # ── Attention extractor ──
    base_t = pipe.transformer
    if hasattr(base_t, 'base_model'):
        base_t = base_t.base_model.model
    attn_ext = CrossAttentionExtractor(
        transformer=base_t,
        layer_indices=list(cfg.suca.attention_layers),
    )

    # ── Parser, responsibility ──
    parser = SemanticUnitParser(use_rules_only=True)
    resp_builder = ResponsibilityMatrixBuilder(
        tau=cfg.suca.tau, top_m_spatial=cfg.suca.top_m_spatial,
    )

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(
        [p for p in pipe.transformer.parameters() if p.requires_grad],
        lr=lr,
    )

    # ── HTTP clients ──
    rollout = RolloutPool(ports=[8200, 8201, 8202, 8203])
    reward = RewardPool(ports=[8100, 8101], model_name="Qwen2.5-VL-7B-Instruct")

    # ── Baselines ──
    baselines = {}
    baseline_mom = 0.9

    # ── Wandb ──
    wandb_run = None
    try:
        import wandb
        wandb.init(project="suca-rl", config={"lr": lr, "lora_rank": lora_rank, "group_size": group_size})
        wandb_run = wandb
    except Exception:
        pass

    # ── Load prompts ──
    prompts = json.load(open(cfg.data.prompt_file)) if os.path.exists(cfg.data.prompt_file) else ["a red cat and a blue dog"] * 10
    if isinstance(prompts[0], dict):
        prompts = [p["prompt"] for p in prompts]
    logger.info(f"Loaded {len(prompts)} prompts, group_size={group_size}")

    # ── Training loop ──
    global_step = 0
    best_score = -1
    recent_ckpts = []

    for epoch in range(num_epochs):
        logger.info(f"=== Epoch {epoch+1}/{num_epochs} ===")
        random.shuffle(prompts)
        ep = prompts[:max_prompts]

        for g_start in range(0, len(ep), group_size):
            group = ep[g_start:g_start + group_size]
            t0 = time.time()

            # ── Phase A: Parallel rollout + reward ──
            t_rollout = time.time()
            images = rollout.generate_batch(group)
            dt_rollout = time.time() - t_rollout

            t_reward = time.time()
            scored = []
            for i, (prompt, image) in enumerate(zip(group, images)):
                if image is None:
                    continue
                units = parser.parse(prompt)
                vqa_qs = [u.vqa_question for u in units]
                units = parser.resolve_token_indices(units, prompt, pipe.tokenizer)
                scores = reward.score(image, vqa_qs)
                scored.append({
                    "prompt": prompt, "units": units,
                    "rewards": torch.tensor(scores, dtype=torch.float32),
                })
            dt_reward = time.time() - t_reward

            # ── Phase B: Trajectory + backward on train device ──
            pipe.transformer.train()
            optimizer.zero_grad()
            total_loss = 0.0
            total_reward_val = 0.0
            num_valid = 0

            t_bwd = time.time()
            for item in scored:
                try:
                    prompt = item["prompt"]
                    units = item["units"]
                    rewards = item["rewards"]

                    with torch.no_grad():
                        with attn_ext.capture() as ext:
                            text_cond = train_policy.encode_prompt(prompt)
                            traj = train_policy.generate_trajectory(
                                prompt, text_cond=text_cond, attention_extractor=ext,
                            )

                    C = resp_builder.build(extractor=attn_ext, units=units)
                    T = len(traj["timesteps"])
                    if C.shape[1] != T:
                        C = C[:, :T] if C.shape[1] > T else torch.nn.functional.pad(C, (0, T-C.shape[1]), value=1.0/T)

                    baseline = torch.tensor(
                        [baselines.get(f"{u.unit_type.value}:{u.description}", 0.0) for u in units],
                        dtype=torch.float32,
                    ) if cfg.suca.use_unit_baseline else None
                    advantages = UnitRewardComputer.compute_unit_advantages(rewards, baseline=baseline)

                    for i, u in enumerate(units):
                        key = f"{u.unit_type.value}:{u.description}"
                        old = baselines.get(key, rewards[i].item())
                        baselines[key] = baseline_mom * old + (1 - baseline_mom) * rewards[i].item()

                    weights_per_t = (C.to(device) * advantages.to(device).unsqueeze(1)).sum(dim=0)
                    loss = train_policy.compute_weighted_loss_batched(
                        traj, weights_per_t, batch_size=bwd_batch_size, sigma=1.0,
                    )

                    total_loss += loss
                    total_reward_val += rewards.mean().item()
                    num_valid += 1
                except Exception as e:
                    logger.error(f"Backward error: {e}")
            dt_bwd = time.time() - t_bwd

            # ── Optimizer step ──
            n = max(num_valid, 1)
            if num_valid > 0:
                for p in pipe.transformer.parameters():
                    if p.grad is not None:
                        p.grad /= n
                torch.nn.utils.clip_grad_norm_(
                    pipe.transformer.parameters(), max_grad_norm,
                )
                optimizer.step()

            global_step += 1
            elapsed = time.time() - t0
            print(
                f"[Step {global_step}] loss={total_loss/n:.6f} "
                f"reward={total_reward_val/n:.4f} valid={num_valid}/{len(group)} "
                f"time={elapsed:.0f}s (roll={dt_rollout:.0f} rew={dt_reward:.0f} bwd={dt_bwd:.0f})",
                flush=True,
            )
            if wandb_run:
                wandb_run.log({
                    "loss": total_loss / n, "reward": total_reward_val / n,
                    "step": global_step, "step_time": elapsed,
                    "time_rollout": dt_rollout, "time_reward": dt_reward, "time_backward": dt_bwd,
                })

            # ── Save ──
            if global_step % save_interval == 0:
                path = os.path.join(outdir, f"checkpoint_step{global_step}")
                os.makedirs(path, exist_ok=True)
                pipe.transformer.save_pretrained(os.path.join(path, "transformer"))
                torch.save({"optimizer": optimizer.state_dict(), "step": global_step, "baselines": baselines},
                           os.path.join(path, "state.pt"))
                logger.info(f"Saved: {path}")
                rollout.sync_weights(os.path.join(path, "transformer"))
                recent_ckpts.append(path)
                while len(recent_ckpts) > 2:
                    old = recent_ckpts.pop(0)
                    if os.path.exists(old):
                        shutil.rmtree(old)

            # ── Eval ──
            if eval_interval > 0 and global_step % eval_interval == 0:
                score = run_eval(rollout, reward, cfg)
                if wandb_run:
                    wandb_run.log({"eval/score": score, "step": global_step})
                if score > best_score:
                    best_score = score
                    best_dir = os.path.join(outdir, "checkpoint_best")
                    if os.path.exists(best_dir):
                        shutil.rmtree(best_dir)
                    os.makedirs(best_dir, exist_ok=True)
                    pipe.transformer.save_pretrained(os.path.join(best_dir, "transformer"))
                    logger.info(f"Best: score={score:.2f} step={global_step}")

    logger.info("Training complete.")


def run_eval(rollout, reward, cfg) -> float:
    benchmark_file = "data/geneval2/geneval2_data.jsonl"
    if not os.path.exists(benchmark_file):
        return 0.0
    logger.info("[Eval] Running 50 prompts...")
    data = [json.loads(l) for l in open(benchmark_file)][:50]
    all_scores = []
    for d in data:
        try:
            image = rollout.generate(d["prompt"])
            if image is None:
                continue
            questions = [q for q, _ in d["vqa_list"]]
            scores = reward.score(image, questions)
            all_scores.append(scores)
        except Exception:
            continue
    if not all_scores:
        return 0.0
    from scipy.stats import gmean
    per_prompt = [gmean(s) if min(s) > 0 else 0.0 for s in all_scores]
    score = 100 * sum(per_prompt) / len(per_prompt)
    logger.info(f"[Eval] score={score:.2f}")
    return score


if __name__ == "__main__":
    main()
