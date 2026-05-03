"""
FSDP Training Script for SUCA.

Launched via: torchrun --nproc_per_node=4 scripts/train_fsdp.py
Expects rollout workers on ports 8200-8201 and reward servers on ports 8100-8101.

GPU layout (via CUDA_VISIBLE_DEVICES=0,1,2,3):
  rank 0-3: FSDP-sharded SD3.5 LoRA training
"""
import base64
import contextlib
import json
import logging
import os
import random
import shutil
import sys
import time
from io import BytesIO
from typing import Dict, List, Optional

import requests
import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model
from peft.utils.other import fsdp_auto_wrap_policy
from PIL import Image

from suca.attention_extractor import CrossAttentionExtractor
from suca.diffusion_policy import DiffusionPolicy
from suca.responsibility_matrix import ResponsibilityMatrixBuilder
from suca.semantic_parser import SemanticUnitParser
from suca.unit_reward import UnitRewardComputer

logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("fsdp-trainer")


# ======================================================================
# HTTP clients for rollout and reward
# ======================================================================

class RolloutClient:
    """Send generation requests to rollout workers."""
    def __init__(self, ports=(8200, 8201)):
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
            img_bytes = base64.b64decode(data["image_b64"])
            return Image.open(BytesIO(img_bytes)).convert("RGB")
        except Exception as e:
            logger.warning(f"Rollout error (port {port}): {e}")
            return None

    def sync_weights(self, adapter_path: str):
        for port in self.ports:
            try:
                requests.post(
                    f"http://localhost:{port}/sync_weights",
                    json={"adapter_path": adapter_path}, timeout=30,
                )
            except Exception:
                pass


class RewardClient:
    """Score images via vLLM OpenAI-compatible server."""
    def __init__(self, port=8100, model_name="Qwen2.5-VL-7B-Instruct"):
        self.port = port
        self.model_name = model_name
        self.base_url = f"http://localhost:{port}"

    def score(self, image: Image.Image, questions: List[str]) -> List[float]:
        import math
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        scores = []
        for q in questions:
            try:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [{
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                                {"type": "text", "text": f"{q} Answer in one word."},
                            ],
                        }],
                        "max_tokens": 5,
                        "temperature": 0.0,
                        "logprobs": True,
                        "top_logprobs": 5,
                    },
                    timeout=60,
                )
                data = resp.json()
                choice = data["choices"][0]
                text = choice["message"]["content"].strip().lower()
                # Extract yes/no probability from logprobs
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
                logger.warning(f"VQA error: {e}")
                scores.append(0.5)
        return scores


# ======================================================================
# Main
# ======================================================================

def main():
    # ── Init distributed ──
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    is_main = (rank == 0)
    if is_main:
        logger.info(f"FSDP trainer: {world_size} ranks")

    # ── Config ──
    cfg = OmegaConf.load("config/default.yaml")
    cfg.training.group_size = 5
    cfg.training.learning_rate = 1e-4
    cfg.training.max_grad_norm = 1.0
    cfg.training.lora_rank = 64
    cfg.training.lora_alpha = 128
    cfg.training.backward_batch_size = 5
    cfg.training.num_epochs = 10
    cfg.diffusion.num_inference_steps = 15
    cfg.diffusion.guidance_scale = 4.5
    cfg.data.max_prompts_per_epoch = 500

    model_path = cfg.model.pretrained_model_name
    warmup_ckpt = cfg.model.get("warmup_checkpoint", None)
    num_steps = cfg.diffusion.num_inference_steps
    guidance = cfg.diffusion.guidance_scale
    outdir = "outputs/rl"
    os.makedirs(outdir, exist_ok=True)

    # ── Load pipeline ──
    if is_main:
        logger.info("Loading SD3.5 pipeline...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_path, torch_dtype=torch.float16,
    )
    if warmup_ckpt and os.path.exists(warmup_ckpt):
        if is_main:
            logger.info(f"Loading warmup: {warmup_ckpt}")
        t = SD3Transformer2DModel.from_pretrained(warmup_ckpt, torch_dtype=torch.float16)
        pipe.transformer = t

    # ── LoRA (BEFORE FSDP) ──
    lora_cfg = LoraConfig(
        r=cfg.training.lora_rank,
        lora_alpha=cfg.training.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0, bias="none",
    )
    pipe.transformer = get_peft_model(pipe.transformer, lora_cfg)
    for _, p in pipe.transformer.named_parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)

    if is_main:
        trainable = sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad)
        total = sum(p.numel() for p in pipe.transformer.parameters())
        logger.info(f"LoRA: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ── FSDP wrap transformer ──
    if is_main:
        logger.info("Wrapping transformer with FSDP...")
    auto_wrap = fsdp_auto_wrap_policy(pipe.transformer)
    # Don't set param_dtype — let LoRA stay float32, base stays float16
    mixed_prec = MixedPrecision(
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float16,
    )
    pipe.transformer = FSDP(
        pipe.transformer,
        auto_wrap_policy=auto_wrap,
        mixed_precision=mixed_prec,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        sync_module_states=True,
        use_orig_params=True,
    )

    # Move rest of pipeline to rank's device
    pipe.vae = pipe.vae.to(device)
    pipe.text_encoder = pipe.text_encoder.to(device) if pipe.text_encoder else None
    pipe.text_encoder_2 = pipe.text_encoder_2.to(device) if pipe.text_encoder_2 else None
    pipe.text_encoder_3 = pipe.text_encoder_3.to(device) if pipe.text_encoder_3 else None

    train_policy = DiffusionPolicy(pipe, num_steps, guidance)

    # ── Optimizer (on FSDP params) ──
    optimizer = torch.optim.AdamW(
        [p for p in pipe.transformer.parameters() if p.requires_grad],
        lr=cfg.training.learning_rate,
    )

    # ── Attention extractor (rank 0 only uses it) ──
    base_t = pipe.transformer
    if hasattr(base_t, 'module'):
        base_t = base_t.module
    if hasattr(base_t, 'base_model'):
        base_t = base_t.base_model.model
    attention_extractor = CrossAttentionExtractor(
        transformer=base_t,
        layer_indices=list(cfg.suca.attention_layers),
    )

    # ── Other components (rank 0 only) ──
    parser = SemanticUnitParser(use_rules_only=True)
    resp_builder = ResponsibilityMatrixBuilder(
        tau=cfg.suca.tau, top_m_spatial=cfg.suca.top_m_spatial,
    )

    # ── HTTP clients (rank 0 only) ──
    rollout_client = RolloutClient(ports=[8200, 8201])
    reward_client = RewardClient(port=8100, model_name="Qwen2.5-VL-7B-Instruct")
    reward_client_2 = RewardClient(port=8101, model_name="Qwen2.5-VL-7B-Instruct")
    reward_clients = [reward_client, reward_client_2]

    # ── Baselines ──
    unit_baselines = {}
    baseline_momentum = 0.9

    # ── Wandb (rank 0 only) ──
    wandb_run = None
    if is_main:
        try:
            import wandb
            wandb.init(project="suca-rl", config=OmegaConf.to_container(cfg))
            wandb_run = wandb
        except Exception:
            pass

    # ── Load prompts ──
    prompts = json.load(open(cfg.data.prompt_file)) if os.path.exists(cfg.data.prompt_file) else ["a red cat"] * 10
    if isinstance(prompts[0], dict):
        prompts = [p["prompt"] for p in prompts]
    if is_main:
        logger.info(f"Loaded {len(prompts)} prompts")

    # ── Training loop ──
    global_step = 0
    group_size = cfg.training.group_size
    max_prompts = cfg.data.get("max_prompts_per_epoch", len(prompts))
    save_interval = 50
    eval_interval = 10
    best_score = -float("inf")
    recent_ckpts = []

    for epoch in range(cfg.training.num_epochs):
        if is_main:
            logger.info(f"=== Epoch {epoch + 1}/{cfg.training.num_epochs} ===")
        random.shuffle(prompts)
        ep_prompts = prompts[:max_prompts]

        for g_start in range(0, len(ep_prompts), group_size):
            group = ep_prompts[g_start:g_start + group_size]
            t0 = time.time()

            pipe.transformer.train()
            optimizer.zero_grad()

            total_loss = 0.0
            total_reward = 0.0
            num_valid = 0

            # ── Process each prompt in group ──
            for prompt in group:
                # Step 1: Rollout (rank 0 sends HTTP, others wait)
                image = None
                units = None
                rewards = None

                if is_main:
                    image = rollout_client.generate(prompt)
                    if image is None:
                        continue

                    # Step 2: Parse + Reward
                    units = parser.parse(prompt)
                    vqa_qs = [u.vqa_question for u in units]
                    units = parser.resolve_token_indices(units, prompt, pipe.tokenizer)
                    rc = reward_clients[num_valid % len(reward_clients)]
                    scores = rc.score(image, vqa_qs)
                    rewards = torch.tensor(scores, dtype=torch.float32)

                # Broadcast whether this prompt is valid
                valid_tensor = torch.tensor([1 if is_main and image is not None else 0], device=device)
                dist.broadcast(valid_tensor, src=0)
                if valid_tensor.item() == 0:
                    continue

                # Broadcast rewards to all ranks
                if not is_main:
                    rewards = torch.zeros(20, dtype=torch.float32)  # placeholder
                n_units = torch.tensor([rewards.shape[0]], device=device)
                dist.broadcast(n_units, src=0)
                if not is_main:
                    rewards = torch.zeros(n_units.item(), dtype=torch.float32)
                rewards = rewards.to(device)
                dist.broadcast(rewards, src=0)

                # Step 3: Generate trajectory on ALL ranks (FSDP requires collective)
                # Broadcast prompt to all ranks
                prompt_bytes = prompt.encode('utf-8') if is_main else b""
                prompt_len = torch.tensor([len(prompt_bytes)], device=device)
                dist.broadcast(prompt_len, src=0)
                if not is_main:
                    prompt_bytes = bytes(prompt_len.item())
                prompt_tensor = torch.ByteTensor(list(prompt_bytes)).to(device)
                if not is_main:
                    prompt_tensor = torch.zeros(prompt_len.item(), dtype=torch.uint8, device=device)
                dist.broadcast(prompt_tensor, src=0)
                prompt = bytes(prompt_tensor.cpu().tolist()).decode('utf-8')

                with torch.no_grad():
                    with attention_extractor.capture() as ext:
                        text_cond = train_policy.encode_prompt(prompt)
                        traj = train_policy.generate_trajectory(
                            prompt, text_cond=text_cond, attention_extractor=ext,
                        )

                # Step 4: Responsibility matrix + advantages (rank 0)
                T = len(traj["timesteps"])
                weights_per_t = torch.zeros(T, device=device)

                if is_main:
                    C = resp_builder.build(extractor=attention_extractor, units=units)
                    if C.shape[1] != T:
                        C = C[:, :T] if C.shape[1] > T else torch.nn.functional.pad(
                            C, (0, T - C.shape[1]), value=1.0/T)

                    baseline = torch.tensor(
                        [unit_baselines.get(f"{u.unit_type.value}:{u.description}", 0.0) for u in units],
                        dtype=torch.float32,
                    ) if cfg.suca.use_unit_baseline else None
                    advantages = UnitRewardComputer.compute_unit_advantages(rewards.cpu(), baseline=baseline)

                    # Update baselines
                    for i, u in enumerate(units):
                        key = f"{u.unit_type.value}:{u.description}"
                        old = unit_baselines.get(key, rewards[i].item())
                        unit_baselines[key] = baseline_momentum * old + (1 - baseline_momentum) * rewards[i].item()

                    weights_per_t = (C.to(device) * advantages.to(device).unsqueeze(1)).sum(dim=0)

                # Broadcast weights_per_t to all ranks
                dist.broadcast(weights_per_t, src=0)

                # Step 5: Batched backward (ALL ranks participate — FSDP collective)
                bwd_bs = cfg.training.backward_batch_size
                n_chunks = (T + bwd_bs - 1) // bwd_bs

                for chunk_idx in range(n_chunks):
                    start = chunk_idx * bwd_bs
                    end = min(start + bwd_bs, T)
                    indices = list(range(start, end))
                    B = len(indices)

                    latents_batch = torch.cat(
                        [traj["latents_trajectory"][i].detach() for i in indices], dim=0,
                    )
                    targets_batch = torch.cat(
                        [traj["model_outputs"][i].detach() for i in indices], dim=0,
                    )
                    t_vals = torch.tensor(
                        [traj["timesteps"][i] for i in indices], device=device,
                    )
                    w_batch = weights_per_t[indices]

                    # Conditional-only forward (skip CFG for backward — 2x faster)
                    text_cond = traj["text_cond"]
                    embeds = text_cond["prompt_embeds"].expand(B, -1, -1)
                    pooled = text_cond["pooled_prompt_embeds"].expand(B, -1)

                    # Use no_sync for all but last chunk
                    is_last = (chunk_idx == n_chunks - 1)
                    ctx = contextlib.nullcontext() if is_last else pipe.transformer.no_sync()
                    with ctx:
                        model_out = pipe.transformer(
                            hidden_states=latents_batch,
                            timestep=t_vals,
                            encoder_hidden_states=embeds,
                            pooled_projections=pooled,
                            return_dict=False,
                        )[0]
                        diff = (targets_batch - model_out).view(B, -1)
                        log_probs = -0.5 * (diff ** 2).mean(dim=1)
                        batch_loss = -(w_batch * log_probs).sum() / T

                        if not (torch.isnan(batch_loss) or torch.isinf(batch_loss)):
                            batch_loss.backward()
                            total_loss += batch_loss.item()

                total_reward += rewards.mean().item()
                num_valid += 1

            # ── Optimizer step ──
            n = max(num_valid, 1)
            if num_valid > 0:
                # Scale grads by group size
                for p in pipe.transformer.parameters():
                    if p.grad is not None:
                        p.grad /= n
                pipe.transformer.clip_grad_norm_(cfg.training.max_grad_norm)
                optimizer.step()

            global_step += 1

            if is_main:
                elapsed = time.time() - t0
                print(
                    f"[Step {global_step}] loss={total_loss/n:.8f} "
                    f"reward={total_reward/n:.4f} valid={num_valid}/{len(group)} "
                    f"time={elapsed:.0f}s",
                    flush=True,
                )
                if wandb_run:
                    wandb_run.log({
                        "loss": total_loss / n,
                        "reward": total_reward / n,
                        "step": global_step,
                        "step_time": elapsed,
                    })

            # ── Save + Sync ──
            if global_step % save_interval == 0 and is_main:
                ckpt_path = os.path.join(outdir, f"checkpoint_step{global_step}")
                os.makedirs(ckpt_path, exist_ok=True)
                save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FSDP.state_dict_type(pipe.transformer, StateDictType.FULL_STATE_DICT, save_cfg):
                    state = pipe.transformer.state_dict()
                if is_main:
                    torch.save(state, os.path.join(ckpt_path, "adapter_model.bin"))
                    logger.info(f"Saved: {ckpt_path}")
                    # Sync to rollout workers
                    rollout_client.sync_weights(ckpt_path)
                    # Manage checkpoints
                    recent_ckpts.append(ckpt_path)
                    while len(recent_ckpts) > 2:
                        old = recent_ckpts.pop(0)
                        if os.path.exists(old):
                            shutil.rmtree(old)

            # ── Eval ──
            if eval_interval > 0 and global_step % eval_interval == 0 and is_main:
                score = run_eval(train_policy, rollout_client, reward_client, cfg, attention_extractor)
                if wandb_run:
                    wandb_run.log({"eval/soft_tifa_gm": score, "step": global_step})
                if score > best_score:
                    best_score = score
                    best_dir = os.path.join(outdir, "checkpoint_best")
                    if os.path.exists(best_dir):
                        shutil.rmtree(best_dir)
                    save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                    with FSDP.state_dict_type(pipe.transformer, StateDictType.FULL_STATE_DICT, save_cfg):
                        state = pipe.transformer.state_dict()
                    os.makedirs(best_dir, exist_ok=True)
                    torch.save(state, os.path.join(best_dir, "adapter_model.bin"))
                    logger.info(f"Best: score={score:.2f} step={global_step}")

            dist.barrier()

    # Cleanup
    if is_main:
        logger.info("Training complete.")
    dist.destroy_process_group()


def run_eval(policy, rollout_client, reward_client, cfg, attn_ext) -> float:
    """Quick eval on GenEval2 subset."""
    benchmark_file = "data/geneval2/geneval2_data.jsonl"
    if not os.path.exists(benchmark_file):
        return 0.0

    logger.info("[Eval] Running...")
    data = [json.loads(l) for l in open(benchmark_file)][:50]
    all_scores = []

    for d in data:
        try:
            image = rollout_client.generate(d["prompt"])
            if image is None:
                continue
            questions = [q for q, _ in d["vqa_list"]]
            scores = reward_client.score(image, questions)
            all_scores.append(scores)
        except Exception:
            continue

    if not all_scores:
        return 0.0

    from scipy.stats import gmean
    per_prompt = [gmean(s) if min(s) > 0 else 0.0 for s in all_scores]
    score = 100 * sum(per_prompt) / len(per_prompt)
    logger.info(f"[Eval] Step score={score:.2f}")
    return score


if __name__ == "__main__":
    main()
