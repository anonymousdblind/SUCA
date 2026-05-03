from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusion3Pipeline
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd3_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.sd3_sde_with_logprob import sde_step_with_logprob
from flow_grpo.diffusers_patch.train_dreambooth_lora_sd3 import encode_prompt
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

# === SUCA imports ===
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from suca.attention_extractor import CrossAttentionExtractor
from suca.responsibility_matrix import ResponsibilityMatrixBuilder
from suca.semantic_parser import SemanticUnitParser
from suca.unit_reward import UnitRewardComputer
import base64, requests, math
from io import BytesIO

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


# === SUCA VLM Reward Client (HTTP to vLLM/FastAPI server) ===
class SUCARewardClient:
    """Score images via VLM server for per-unit rewards."""
    def __init__(self, ports=(8100, 8101), model_name="Qwen2.5-VL-7B-Instruct"):
        self.ports = list(ports)
        self.model_name = model_name
        self._idx = 0

    def score_units(self, image: Image.Image, vqa_questions: list) -> list:
        """Score VQA questions via FastAPI /score_batch endpoint."""
        port = self.ports[self._idx % len(self.ports)]
        self._idx += 1
        buf = BytesIO()
        image.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            resp = requests.post(
                f"http://localhost:{port}/score_batch",
                json={"image_b64": img_b64, "questions": vqa_questions},
                timeout=120,
            )
            return resp.json()["scores"]
        except Exception as e:
            return [0.5] * len(vqa_questions)


def compute_suca_advantages(
    images_pt, prompts, num_train_timesteps, pipeline, attention_extractor,
    parser, resp_builder, reward_client, unit_baselines, baseline_momentum=0.9,
    accelerator=None, precomputed_rewards=None, per_unit_scores=None,
    precomputed_C_matrices=None, anchor_process_scores=None, process_lambda=0.3,
):
    """
    Compute SUCA per-timestep advantages for a batch of (image, prompt) pairs.

    Key: uses actual per-unit VQA scores (per_unit_scores) for true credit assignment.
    Each unit gets its own reward (e.g., "is the cat red?" → 0.9, "are there two dogs?" → 0.3),
    so the model learns to improve specific aspects rather than everything uniformly.

    Falls back to scalar precomputed_rewards (uniform) if per-unit scores unavailable.

    Returns: advantages tensor of shape (batch_size, num_train_timesteps)
    """
    batch_size = len(prompts)
    advantages = torch.zeros(batch_size, num_train_timesteps)
    # Diagnostic counters
    n_with_per_unit = 0
    n_fallback_scalar = 0
    n_skipped = 0
    n_attn_C = 0  # count samples where attention-based C was used

    for idx in range(batch_size):
        try:
            prompt = prompts[idx]

            # 1. Parse prompt into semantic units (online_mode: only hard units)
            units = parser.parse(prompt, online_mode=True)
            if not units:
                n_skipped += 1
                continue
            units = parser.resolve_token_indices(units, prompt, pipeline.tokenizer)
            K = len(units)

            # 2. Get per-unit rewards — use actual VQA scores when available
            if per_unit_scores is not None and idx < len(per_unit_scores) and per_unit_scores[idx] is not None and len(per_unit_scores[idx]) > 0:
                # TRUE per-unit credit: each unit scored independently by VLM
                unit_scores = per_unit_scores[idx]
                if len(unit_scores) == K:
                    rewards = torch.tensor(unit_scores, dtype=torch.float32)
                    n_with_per_unit += 1
                else:
                    # Mismatch (parser changed units) — fallback to scalar
                    total_reward = float(precomputed_rewards[idx]) if precomputed_rewards is not None else 0.5
                    rewards = torch.full((K,), total_reward, dtype=torch.float32)
                    n_fallback_scalar += 1
            elif precomputed_rewards is not None:
                # Fallback: uniform scalar reward to all units
                total_reward = float(precomputed_rewards[idx])
                rewards = torch.full((K,), total_reward, dtype=torch.float32)
                n_fallback_scalar += 1
            else:
                n_skipped += 1
                continue

            # 3. Compute per-unit advantages (with running baseline per unit type)
            unit_advs = []
            for i, u in enumerate(units):
                key = f"{u.unit_type.value}:{u.description}"
                baseline = unit_baselines.get(key, 0.0)
                adv = rewards[i].item() - baseline
                unit_advs.append(adv)
                unit_baselines[key] = baseline_momentum * baseline + (1 - baseline_momentum) * rewards[i].item()
            unit_advs = torch.tensor(unit_advs, dtype=torch.float32)

            # 5. Build responsibility matrix C[k, t] from attention maps
            K = len(units)
            C = torch.ones(K, num_train_timesteps) / num_train_timesteps

            # Use pre-computed C matrix from attention capture during sampling
            if precomputed_C_matrices is not None and idx < len(precomputed_C_matrices) and precomputed_C_matrices[idx] is not None:
                try:
                    C_pre = torch.tensor(precomputed_C_matrices[idx], dtype=torch.float32) if not isinstance(precomputed_C_matrices[idx], torch.Tensor) else precomputed_C_matrices[idx]
                    if C_pre.shape[0] == K:
                        T_pre = C_pre.shape[1]
                        if T_pre >= num_train_timesteps:
                            C = C_pre[:, :num_train_timesteps]
                        else:
                            C[:, :T_pre] = C_pre
                        n_attn_C += 1
                except Exception:
                    pass  # keep uniform fallback

            # 6. Per-timestep weights: w_t = sum_k C[k,t] * A_k
            # C: (K, T), unit_advs: (K,)
            w_t = (C * unit_advs.unsqueeze(1)).sum(dim=0)  # (T,)

            # 7. Add sparse process reward at anchor timesteps
            if anchor_process_scores is not None and idx < len(anchor_process_scores):
                for a_step, unit_scores_at_anchor in anchor_process_scores[idx].items():
                    if a_step >= num_train_timesteps:
                        continue
                    proc_adv = 0.0
                    n_valid = 0
                    for unit_idx, score in unit_scores_at_anchor.items():
                        if score is None:  # confidence-gated out
                            continue
                        unit_key = f"proc_{units[unit_idx].unit_type.value}:{units[unit_idx].description}_t{a_step}"
                        proc_baseline = unit_baselines.get(unit_key, 0.0)
                        proc_adv += score - proc_baseline
                        n_valid += 1
                        unit_baselines[unit_key] = baseline_momentum * proc_baseline + (1 - baseline_momentum) * score
                    if n_valid > 0:
                        w_t[a_step] += process_lambda * (proc_adv / n_valid)

            advantages[idx] = w_t

        except Exception as e:
            n_skipped += 1

    # Log SUCA diagnostics
    if accelerator and accelerator.is_main_process:
        import logging as _logging
        _logger = _logging.getLogger(__name__)
        _logger.info(
            f"[SUCA] per_unit={n_with_per_unit} | scalar_fallback={n_fallback_scalar} | "
            f"skipped={n_skipped} | attn_C={n_attn_C} | total={batch_size}"
        )
        # Log per-unit type reward stats
        type_rewards = {}
        for idx in range(batch_size):
            if per_unit_scores and idx < len(per_unit_scores) and per_unit_scores[idx]:
                try:
                    units = parser.parse(prompts[idx], online_mode=True)
                except Exception:
                    continue
                for i, u in enumerate(units):
                    if i < len(per_unit_scores[idx]):
                        utype = u.unit_type.value
                        type_rewards.setdefault(utype, []).append(per_unit_scores[idx][i])
        if type_rewards:
            type_str = " | ".join(f"{t}={np.mean(v):.3f}({len(v)})" for t, v in sorted(type_rewards.items()))
            _logger.info(f"[SUCA] per-type rewards: {type_str}")

        # Log process reward stats
        if anchor_process_scores:
            proc_stats = {}  # {step: [scores]}
            for sample_anchors in anchor_process_scores:
                if not sample_anchors:
                    continue
                for a_step, unit_scores in sample_anchors.items():
                    for k, score in unit_scores.items():
                        if score is not None:
                            proc_stats.setdefault(a_step, []).append(score)
            if proc_stats:
                proc_str = " | ".join(
                    f"t={s}: mean={np.mean(v):.2f} std={np.std(v):.2f} n={len(v)} gated={sum(1 for sa in anchor_process_scores for us in [sa.get(s,{})] for sc in us.values() if sc is None)}"
                    for s, v in sorted(proc_stats.items())
                )
                _logger.info(f"[SUCA] process reward: {proc_str}")

    return advantages


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def compute_text_embeddings(prompt, text_encoders, tokenizers, max_sequence_length, device):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config):
    if config.train.cfg:
        noise_pred = transformer(
            hidden_states=torch.cat([sample["latents"][:, j]] * 2),
            timestep=torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = transformer(
            hidden_states=sample["latents"][:, j],
            timestep=sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            pooled_projections=pooled_embeds,
            return_dict=False,
        )[0]
    
    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = sde_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        prev_sample=sample["next_latents"][:, j].float(),
        noise_level=config.sample.noise_level,
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters, eval_tag="Eval"):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.test_batch_size, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    for test_batch in tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
        prompts, prompt_metadata = test_batch
        prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
            prompts, 
            text_encoders, 
            tokenizers, 
            max_sequence_length=128, 
            device=accelerator.device
        )
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
            sample_neg_pooled_prompt_embeds = sample_neg_pooled_prompt_embeds[:len(prompt_embeds)]
        with autocast():
            with torch.no_grad():
                images, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution, 
                    noise_level=0,
                )
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device).float()).cpu().numpy()
    last_batch_prompt_ids = tokenizers[0](
        prompts,
        padding="max_length",
        max_length=256,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][index] for k in last_batch_rewards_gather} for index in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"{eval_tag}_reward/{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
            # Print detailed eval to console
            eval_str = " | ".join(f"{k}: {np.mean(v[v != -10]):.4f}" for k, v in all_rewards.items())
            # Reward distribution
            avg_rewards = all_rewards.get("avg", all_rewards.get("suca_vqa", np.array([0])))
            valid_r = avg_rewards[avg_rewards != -10]
            r_detail = f"mean={np.mean(valid_r):.4f} std={np.std(valid_r):.4f} min={np.min(valid_r):.3f} max={np.max(valid_r):.3f}" if len(valid_r) > 0 else "N/A"
            logger.info(f"[{eval_tag}] step={global_step} | {eval_str}")
            logger.info(f"[{eval_tag}] reward dist: {r_detail} | n={len(valid_r)}")

            # === Per-unit-type eval breakdown (diagnose which types improve/regress) ===
            try:
                sys_path_bak = list(sys.path)
                if os.getcwd() not in sys.path:
                    sys.path.insert(0, os.getcwd())
                from suca.semantic_parser import SemanticUnitParser as _EvalParser
                _eval_parser = _EvalParser(use_rules_only=True)
                eval_type_scores = {}
                eval_per_prompt = []
                # Reconstruct all eval prompts from gathered data
                all_eval_prompts = []
                for test_batch in test_dataloader:
                    batch_prompts, _ = test_batch
                    all_eval_prompts.extend(batch_prompts)
                # Score each eval prompt's units
                for ep_idx, ep in enumerate(all_eval_prompts):
                    if ep_idx >= len(valid_r):
                        break
                    try:
                        units = _eval_parser.parse(ep, online_mode=True)
                        prompt_score = valid_r[ep_idx]
                        n_units = len(units) if units else 0
                        eval_per_prompt.append({"prompt": ep[:80], "score": float(prompt_score), "n_units": n_units})
                    except Exception:
                        pass
                # Reparse and group by type using the avg score as proxy
                if True:
                    for ep_idx, ep in enumerate(all_eval_prompts):
                        if ep_idx >= len(valid_r):
                            break
                        try:
                            units = _eval_parser.parse(ep, online_mode=True)
                            for u in units:
                                eval_type_scores.setdefault(u.unit_type.value, []).append(float(valid_r[ep_idx]))
                        except Exception:
                            pass
                if eval_type_scores:
                    type_str = " | ".join(f"{t}={np.mean(v):.4f}(n={len(v)})" for t, v in sorted(eval_type_scores.items()))
                    logger.info(f"[{eval_tag}] per-type (by prompt avg): {type_str}")
                # Log worst/best prompts
                eval_per_prompt.sort(key=lambda x: x["score"])
                if eval_per_prompt:
                    logger.info(f"[{eval_tag}] worst 5 prompts:")
                    for item in eval_per_prompt[:5]:
                        logger.info(f"  score={item['score']:.4f} units={item['n_units']} | {item['prompt']}")
                    logger.info(f"[{eval_tag}] best 5 prompts:")
                    for item in eval_per_prompt[-5:]:
                        logger.info(f"  score={item['score']:.4f} units={item['n_units']} | {item['prompt']}")
                # Score distribution buckets
                if len(valid_r) > 0:
                    buckets = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
                    hist = np.histogram(valid_r, bins=buckets)[0]
                    bucket_str = " | ".join(f"[{buckets[i]:.1f},{buckets[i+1]:.1f})={hist[i]}" for i in range(len(hist)))
                    logger.info(f"[{eval_tag}] score distribution: {bucket_str}")
            except Exception as e:
                logger.warning(f"[{eval_tag}] per-type breakdown failed: {e}")

    # === Fixed Prompt Tracking: same prompts every eval to show progression ===
    FIXED_PROMPTS = [
        "a red cat sitting on a blue chair",
        "three green apples and two yellow bananas on a wooden table",
        "a large elephant to the left of a small dog, both wearing hats",
        "a woman in a red dress holding a blue umbrella in the rain",
    ]
    if accelerator.is_main_process:
        try:
            fixed_images = []
            for fp in FIXED_PROMPTS:
                fp_embeds, fp_pooled = compute_text_embeddings(
                    [fp], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device
                )
                with autocast():
                    with torch.no_grad():
                        img, _, _ = pipeline_with_logprob(
                            pipeline,
                            prompt_embeds=fp_embeds,
                            pooled_prompt_embeds=fp_pooled,
                            negative_prompt_embeds=neg_prompt_embed,
                            negative_pooled_prompt_embeds=neg_pooled_prompt_embed,
                            num_inference_steps=config.sample.eval_num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=0,
                        )
                pil = Image.fromarray((img[0].float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                fixed_images.append(pil)

            wandb.log({
                "fixed_prompt_tracking": [
                    wandb.Image(img, caption=f"step={global_step} | {prompt}")
                    for img, prompt in zip(fixed_images, FIXED_PROMPTS)
                ],
            }, step=global_step)
            logger.info(f"[{eval_tag}] Logged {len(FIXED_PROMPTS)} fixed prompt images at step={global_step}")
        except Exception as e:
            logger.warning(f"[Eval] Fixed prompt tracking failed: {e}")

    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

    # Return average reward for checkpoint selection
    try:
        avg_rewards = all_rewards.get("avg", all_rewards.get("suca_vqa", np.array([0])))
        valid_r = avg_rewards[avg_rewards != -10] if isinstance(avg_rewards, np.ndarray) else np.array(avg_rewards)
        return float(np.mean(valid_r)) if len(valid_r) > 0 else 0.0
    except:
        return 0.0

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        wandb.init(
            project="flow_grpo",
        )
        # accelerator.init_trackers(
        #     project_name="flow-grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        config.pretrained.model
    )

    # Load SFT warmup transformer weights (replaces base before LoRA)
    sft_warmup_path = getattr(config, 'sft_warmup_path', '')
    if sft_warmup_path and os.path.exists(sft_warmup_path):
        from diffusers.models import SD3Transformer2DModel
        logger.info(f"Loading SFT warmup transformer from {sft_warmup_path}")
        sft_transformer = SD3Transformer2DModel.from_pretrained(sft_warmup_path)
        pipeline.transformer.load_state_dict(sft_transformer.state_dict())
        del sft_transformer
        logger.info("SFT warmup weights loaded successfully")

    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.text_encoder_3.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)

    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2, pipeline.text_encoder_3]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2, pipeline.tokenizer_3]

    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_2.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder_3.to(accelerator.device, dtype=inference_dtype)
    
    if config.use_lora:
        pipeline.transformer.to(accelerator.device)
    else:
        # Full param training: use bf16 for memory efficiency, autocast handles precision
        pipeline.transformer.to(accelerator.device, dtype=torch.bfloat16)
        # Enable gradient checkpointing to save ~30% memory
        if getattr(config, 'activation_checkpointing', False):
            if hasattr(pipeline.transformer, 'gradient_checkpointing_enable'):
                pipeline.transformer.gradient_checkpointing_enable()
            elif hasattr(pipeline.transformer, 'enable_gradient_checkpointing'):
                pipeline.transformer.enable_gradient_checkpointing()
            else:
                from torch.utils.checkpoint import checkpoint
                pipeline.transformer._gradient_checkpointing_func = checkpoint
            logger.info("[Full param] Gradient checkpointing enabled")
        logger.info("[Full param] bf16 training enabled")

    if config.use_lora:
        # Set correct lora layers
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=config.lora_rank,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.transformer = PeftModel.from_pretrained(pipeline.transformer, config.train.lora_path, is_trainable=True)
            pipeline.transformer.set_adapter("default")
        else:
            pipeline.transformer = get_peft_model(pipeline.transformer, transformer_lora_config)

    transformer = pipeline.transformer
    # Ensure LoRA parameters have requires_grad=True
    n_trainable = 0
    n_total = 0
    for name, param in transformer.named_parameters():
        n_total += 1
        if "lora_" in name:
            param.requires_grad = True
            n_trainable += 1
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    logger.info(f"Trainable params: {len(transformer_trainable_parameters)} / {n_total} | "
                f"LoRA params forced grad: {n_trainable} | "
                f"Total trainable: {sum(p.numel() for p in transformer_trainable_parameters):,}")
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)

    # === SUCA components ===
    suca_parser = None
    suca_resp_builder = None
    suca_attention_extractor = None
    suca_reward_client = None
    suca_unit_baselines = {}
    if getattr(config, 'suca_enabled', False):
        logger.info("[SUCA] Initializing unit credit assignment components...")
        suca_parser = SemanticUnitParser(use_rules_only=True)
        suca_resp_builder = ResponsibilityMatrixBuilder(
            tau=getattr(config, 'suca_tau', 0.1),
            top_m_spatial=getattr(config, 'suca_top_m_spatial', 3),
        )
        # Attention extractor on the training transformer
        base_t = pipeline.transformer
        if hasattr(base_t, 'base_model'):
            base_t = base_t.base_model.model
        suca_attention_extractor = CrossAttentionExtractor(
            transformer=base_t,
            layer_indices=getattr(config, 'suca_attention_layers', [5, 10, 15, 20]),
        )
        suca_reward_client = SUCARewardClient(
            ports=getattr(config, 'suca_reward_ports', [8100, 8101]),
            model_name=getattr(config, 'suca_vlm_model', "Qwen2.5-VL-7B-Instruct"),
        )
        logger.info("[SUCA] Ready.")

    test_samedist_dataloader = None  # default: no same-dist eval
    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler(
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=42
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )

        # OOD eval set (different distribution from train, e.g. Fine-T2I)
        test_ood_dataloader = None
        _ood_path = os.path.join(config.dataset, "test_ood_metadata.jsonl")
        if os.path.exists(_ood_path):
            test_ood_dataset = GenevalPromptDataset(config.dataset, 'test_ood')
            test_ood_dataloader = DataLoader(
                test_ood_dataset,
                batch_size=config.sample.test_batch_size,
                collate_fn=GenevalPromptDataset.collate_fn,
                shuffle=False,
                num_workers=8,
            )
            logger.info(f"[Data] Loaded OOD eval set: {len(test_ood_dataset)} prompts from {_ood_path}")
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")


    neg_prompt_embed, neg_pooled_prompt_embed = compute_text_embeddings([""], text_encoders, tokenizers, max_sequence_length=128, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)
    sample_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.sample.train_batch_size, 1)
    train_neg_pooled_prompt_embeds = neg_pooled_prompt_embed.repeat(config.train.batch_size, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)
    if test_ood_dataloader is not None:
        test_ood_dataloader = accelerator.prepare(test_ood_dataloader)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    epoch = 0
    global_step = 0
    # Resume: extract step from checkpoint path (e.g. "checkpoint-200" → step 200)
    resume_from = getattr(config, 'resume_from', '')
    if resume_from and config.train.lora_path:
        import re as _re
        _m = _re.search(r'checkpoint-(\d+)', resume_from)
        if _m:
            global_step = int(_m.group(1))
            # Each epoch does num_batches_per_epoch gradient updates
            epoch = global_step * config.train.gradient_accumulation_steps // config.sample.num_batches_per_epoch
            logger.info(f"Resuming from step={global_step}, epoch={epoch}")
    train_iter = iter(train_dataloader)

    best_ood_score = -float("inf")
    recent_checkpoints = []  # track last 2 checkpoints for cleanup
    MAX_RECENT_CKPTS = 2

    while True:
        #################### EVAL ####################
        pipeline.transformer.eval()
        if epoch % config.eval_freq == 0:
            # ID eval (same distribution as training — geneval compositional prompts)
            id_score = eval(pipeline, test_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters, eval_tag="Eval-ID")
            # OOD eval (different distribution — Fine-T2I general prompts)
            ood_score = 0.0
            if test_ood_dataloader is not None:
                ood_score = eval(pipeline, test_ood_dataloader, text_encoders, tokenizers, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters, eval_tag="Eval-OOD")

            # Save best OOD checkpoint
            if ood_score > best_ood_score and accelerator.is_main_process:
                best_ood_score = ood_score
                best_dir = os.path.join(config.save_dir, "checkpoints", "checkpoint-best-ood")
                # Remove old best
                if os.path.exists(best_dir):
                    import shutil
                    shutil.rmtree(best_dir)
                save_ckpt(config.save_dir, transformer, "best-ood", accelerator, ema, transformer_trainable_parameters, config)
                # Save metadata
                with open(os.path.join(best_dir, "best_info.json"), "w") as f:
                    json.dump({"ood_score": ood_score, "id_score": id_score, "step": global_step, "epoch": epoch}, f)
                logger.info(f"[Best OOD] New best! score={ood_score:.4f} step={global_step}")

        # Regular checkpoint (keep only last 2)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
            ckpt_path = os.path.join(config.save_dir, "checkpoints", f"checkpoint-{global_step}")
            recent_checkpoints.append(ckpt_path)
            # Remove old checkpoints (keep last 2 + best-ood)
            while len(recent_checkpoints) > MAX_RECENT_CKPTS:
                old = recent_checkpoints.pop(0)
                if os.path.exists(old) and "best-ood" not in old:
                    import shutil
                    shutil.rmtree(old)
                    logger.info(f"[Checkpoint] Removed old: {old}")

        #################### SAMPLING ####################
        pipeline.transformer.eval()
        samples = []
        prompts = []
        local_C_matrices = []  # per-sample responsibility matrices for SUCA
        local_anchor_scores = []  # per-sample anchor process scores
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)

            prompt_embeds, pooled_prompt_embeds = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=accelerator.device
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                generator = create_generator(prompts, base_seed=epoch*10000+i)
            else:
                generator = None

            # Activate SUCA attention capture during sampling
            if suca_attention_extractor is not None:
                suca_attention_extractor.clear()
                suca_attention_extractor.register_hooks()

            with autocast():
                with torch.no_grad():
                    images, latents, log_probs = pipeline_with_logprob(
                        pipeline,
                        prompt_embeds=prompt_embeds,
                        pooled_prompt_embeds=pooled_prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        negative_pooled_prompt_embeds=sample_neg_pooled_prompt_embeds,
                        num_inference_steps=config.sample.num_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        noise_level=config.sample.noise_level,
                        generator=generator,
                        attention_extractor=suca_attention_extractor,
                )

            # Deactivate hooks and compute per-sample C matrices
            if suca_attention_extractor is not None:
                suca_attention_extractor.remove_hooks()
                for s in range(len(prompts)):
                    try:
                        units = suca_parser.parse(prompts[s], online_mode=True)
                        if not units:
                            local_C_matrices.append(None)
                            continue
                        units = suca_parser.resolve_token_indices(units, prompts[s], pipeline.tokenizer)
                        C = suca_resp_builder.build_for_sample(suca_attention_extractor, units, s)
                        local_C_matrices.append(C.numpy().tolist())
                    except Exception:
                        local_C_matrices.append(None)
                suca_attention_extractor.clear()  # free memory

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, num_steps + 1, 16, 96, 96)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)

            timesteps = pipeline.scheduler.timesteps.repeat(
                config.sample.train_batch_size, 1
            )  # (batch_size, num_steps)

            # --- Sparse Process Reward: decode anchor timestep latents ---
            if getattr(config, 'suca_enabled', False) and getattr(config, 'process_reward', False):
                anchor_steps = getattr(config, 'process_anchor_steps', [3, 7])
                anchor_unit_filters = {3: ["entity", "count"], 7: None}
                confidence_gate = getattr(config, 'process_confidence_gate', [0.3, 0.7])

                from flow_grpo.rewards import score_anchor_images
                anchor_futures = {}
                for a_step in anchor_steps:
                    if a_step < latents.shape[1] - 1:  # latents has num_steps+1 entries
                        # Decode latent at anchor step (latents[:, a_step+1] = state after step a_step)
                        anchor_latent = latents[:, a_step + 1].to(pipeline.vae.dtype)
                        anchor_latent = (anchor_latent / pipeline.vae.config.scaling_factor) + pipeline.vae.config.shift_factor
                        with torch.no_grad():
                            anchor_imgs = pipeline.vae.decode(anchor_latent, return_dict=False)[0]
                        anchor_imgs = anchor_imgs.float().clamp(0, 1)
                        anchor_imgs_np = (anchor_imgs * 255).byte().cpu().numpy().transpose(0, 2, 3, 1)
                        anchor_futures[a_step] = executor.submit(
                            score_anchor_images, anchor_imgs_np, prompts, suca_parser,
                            getattr(config, 'suca_reward_ports', [8100]),
                            a_step, anchor_unit_filters.get(a_step),
                            tuple(confidence_gate),
                        )
                        del anchor_imgs, anchor_latent  # free GPU memory

                # Build per-sample anchor scores: list of {step: {unit_idx: score}}
                resolved_anchors = {}
                for a_step, fut in anchor_futures.items():
                    resolved_anchors[a_step] = fut.result()

                for s_idx in range(len(prompts)):
                    sample_anchors = {}
                    for a_step, scores_list in resolved_anchors.items():
                        if s_idx < len(scores_list):
                            sample_anchors[a_step] = scores_list[s_idx]
                    local_anchor_scores.append(sample_anchors)
            else:
                local_anchor_scores.extend([{}] * len(prompts))

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=True)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[
                        :, :-1
                    ],  # each entry is the latent before timestep t
                    "next_latents": latents[
                        :, 1:
                    ],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards,
                }
            )

        # wait for all rewards to be computed
        _reward_t0 = time.time()
        local_per_unit_data = []  # this rank's per-unit scores
        _n_fallback = 0  # count samples where reward = 0.5 (likely fallback)
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            sample["rewards"] = {
                key: torch.as_tensor(value, device=accelerator.device).float()
                for key, value in rewards.items()
            }
            # Save per-unit breakdown for SUCA credit assignment
            if "per_unit_scores" in reward_metadata:
                local_per_unit_data.extend(reward_metadata["per_unit_scores"])
                # Check for all-0.5 fallback scores
                for pu in reward_metadata["per_unit_scores"]:
                    if pu and all(abs(s - 0.5) < 1e-6 for s in pu):
                        _n_fallback += 1
        _reward_dt = time.time() - _reward_t0
        if accelerator.is_local_main_process:
            _n_total = len(local_per_unit_data)
            logger.info(
                f"[Reward] {_n_total} samples scored in {_reward_dt:.1f}s | "
                f"fallback(all-0.5)={_n_fallback}/{_n_total} ({100*_n_fallback/max(_n_total,1):.1f}%)"
            )

        # Gather per-unit data from ALL ranks (variable-length lists, use all_gather_object)
        all_per_unit_gathered = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(all_per_unit_gathered, local_per_unit_data)
        # Flatten: each rank contributed its local samples in order
        # After gather, all_per_unit_gathered[rank_i] = list of per-unit scores for rank i's samples
        all_per_unit_data = []
        for rank_data in all_per_unit_gathered:
            if rank_data:
                all_per_unit_data.extend(rank_data)
            else:
                all_per_unit_data.extend([])

        # Gather SUCA responsibility matrices (C) from ALL ranks
        # Use config check (same on all ranks) to avoid NCCL deadlock
        all_C_matrices = None
        all_anchor_scores = None
        if getattr(config, 'suca_enabled', False):
            all_C_gathered = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(all_C_gathered, local_C_matrices)
            all_C_matrices = []
            for rank_data in all_C_gathered:
                if rank_data:
                    all_C_matrices.extend(rank_data)
                else:
                    all_C_matrices.extend([None] * len(local_C_matrices))

            # Gather anchor process scores from ALL ranks
            if getattr(config, 'process_reward', False):
                all_anchor_gathered = [None] * accelerator.num_processes
                torch.distributed.all_gather_object(all_anchor_gathered, local_anchor_scores)
                all_anchor_scores = []
                for rank_data in all_anchor_gathered:
                    if rank_data:
                        all_anchor_scores.extend(rank_data)
                    else:
                        all_anchor_scores.extend([{}] * len(local_anchor_scores))

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards['avg'][i] for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]

        # === Advantage computation (same pattern as original Flow-GRPO) ===
        # Step 1: Expand rewards to timestep dimension (ALL ranks, identical code)
        samples["rewards"]["avg"] = samples["rewards"]["avg"].unsqueeze(1).repeat(1, num_train_timesteps)

        # Step 2: Gather rewards across ALL processes (NCCL collective — all ranks)
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}

        # Step 3: Log rewards (only main process logs, but code path is same)
        if accelerator.is_main_process:
            wandb.log(
                {"epoch": epoch, **{f"reward_{key}": value.mean() for key, value in gathered_rewards.items()
                 if '_strict_accuracy' not in key and '_accuracy' not in key}},
                step=global_step,
            )

        # Step 4: Compute advantages — SUCA or standard GRPO
        # (ALL ranks execute the SAME branch — no NCCL mismatch)
        if getattr(config, 'suca_enabled', False):
            # Gather prompts (ALL ranks — NCCL collective)
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            all_prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)

            # all_per_unit_data now has ALL ranks' per-unit scores (gathered above)
            all_rewards_scalar = gathered_rewards["ori_avg"]  # (total_samples,) fallback

            # SUCA credit assignment on rank 0 only (CPU-bound, no NCCL)
            if accelerator.is_main_process:
                n_with_units = sum(1 for x in all_per_unit_data if x is not None and len(x) > 0)
                logger.info(f"[SUCA] Computing credit assignment for {len(all_prompts)} prompts "
                           f"({n_with_units}/{len(all_per_unit_data)} with per-unit scores)...")
                suca_advantages = compute_suca_advantages(
                    images_pt=None,
                    prompts=all_prompts,
                    precomputed_rewards=all_rewards_scalar,
                    per_unit_scores=all_per_unit_data,
                    precomputed_C_matrices=all_C_matrices,
                    num_train_timesteps=num_train_timesteps,
                    pipeline=pipeline,
                    attention_extractor=suca_attention_extractor,
                    parser=suca_parser,
                    resp_builder=suca_resp_builder,
                    reward_client=suca_reward_client,
                    unit_baselines=suca_unit_baselines,
                    baseline_momentum=0.9,
                    accelerator=accelerator,
                    anchor_process_scores=all_anchor_scores,
                    process_lambda=getattr(config, 'process_lambda', 0.3),
                )
                # Pad/trim to match num_train_timesteps
                T_suca = suca_advantages.shape[1] if suca_advantages.dim() == 2 else 1
                if T_suca < num_train_timesteps:
                    pad = torch.zeros(suca_advantages.shape[0], num_train_timesteps - T_suca)
                    suca_advantages = torch.cat([suca_advantages, pad], dim=1)
                elif T_suca > num_train_timesteps:
                    suca_advantages = suca_advantages[:, :num_train_timesteps]

                suca_mean = suca_advantages.mean()
                suca_std = suca_advantages.std()
                if suca_std > 1e-6:
                    suca_advantages = (suca_advantages - suca_mean) / (suca_std + 1e-4)
                advantages = suca_advantages.numpy()
                logger.info(f"[SUCA] Done. advantages shape={advantages.shape}, "
                           f"raw_mean={suca_mean:.4f}, raw_std={suca_std:.4f}, "
                           f"normed_mean={advantages.mean():.4f}")
            else:
                # Other ranks: wait and receive via broadcast
                total_samples = len(all_prompts)
                advantages = np.zeros((total_samples, num_train_timesteps), dtype=np.float32)

            # Broadcast advantages from rank 0 to all ranks (ALL ranks — NCCL collective)
            adv_tensor = torch.as_tensor(advantages, device=accelerator.device)
            torch.distributed.broadcast(adv_tensor, src=0)
            advantages = adv_tensor.cpu().numpy()
        elif config.per_prompt_stat_tracking:
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
            advantages = stat_tracker.update(prompts, gathered_rewards['avg'])
            stat_tracker.clear()
        # ungather advantages; keep only entries for this process
        advantages = torch.as_tensor(advantages)
        if accelerator.is_local_main_process:
            print(f"[DEBUG] advantages before ungather: shape={advantages.shape}, abs_mean={advantages.abs().mean():.4f}")
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print(f"[DEBUG] advantages after ungather: shape={samples['advantages'].shape}, abs_mean={samples['advantages'].abs().mean():.4f}")

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        if accelerator.is_local_main_process:
            print(f"[DEBUG] mask: {mask.sum().item()}/{mask.shape[0]} valid samples")
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        num_batches = config.sample.num_batches_per_epoch
        true_count = mask.sum()
        if true_count % num_batches != 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            # --- Comprehensive training metrics ---
            adv_tensor = samples["advantages"]
            adv_stats = {
                # Batch info
                "batch/actual_size": mask.sum().item()//config.sample.num_batches_per_epoch,
                "batch/valid_ratio": mask.sum().item() / mask.shape[0],
                # Advantage distribution
                "suca/advantages_mean": adv_tensor.mean().item(),
                "suca/advantages_std": adv_tensor.std().item(),
                "suca/advantages_abs_mean": adv_tensor.abs().mean().item(),
                "suca/advantages_max": adv_tensor.max().item(),
                "suca/advantages_min": adv_tensor.min().item(),
                # Reward distribution
                "train_reward/mean": float(np.mean(gathered_rewards.get("ori_avg", [0]))),
                "train_reward/std": float(np.std(gathered_rewards.get("ori_avg", [0]))),
                "train_reward/max": float(np.max(gathered_rewards.get("ori_avg", [0]))),
                "train_reward/min": float(np.min(gathered_rewards.get("ori_avg", [0]))),
                "train_reward/median": float(np.median(gathered_rewards.get("ori_avg", [0]))),
                # Reward percentiles (see if top rewards improve)
                "train_reward/p25": float(np.percentile(gathered_rewards.get("ori_avg", [0]), 25)),
                "train_reward/p75": float(np.percentile(gathered_rewards.get("ori_avg", [0]), 75)),
            }
            # Per reward function breakdown
            for key, value in gathered_rewards.items():
                if isinstance(value, np.ndarray) and value.size > 0:
                    valid = value[value != -10] if value.ndim == 1 else value
                    if valid.size > 0:
                        adv_stats[f"train_reward/{key}_mean"] = float(np.mean(valid))

            # Per-unit-type reward breakdown (diagnose which types improve)
            if all_per_unit_data:
                type_rewards = {}
                for pu_scores in all_per_unit_data:
                    if not pu_scores:
                        continue
                    # Re-parse to get unit types (cheap, rule-based)
                    # Just aggregate all scores by position heuristic
                    for i, s in enumerate(pu_scores):
                        type_rewards.setdefault(f"unit_{i}", []).append(s)
                # Also get actual type breakdown from gathered prompts
                all_type_scores = {}
                for batch_idx, pu_scores in enumerate(all_per_unit_data):
                    if not pu_scores:
                        continue
                    try:
                        units = suca_parser.parse(all_prompts[batch_idx] if batch_idx < len(all_prompts) else "", online_mode=True)
                        for i, u in enumerate(units):
                            if i < len(pu_scores):
                                all_type_scores.setdefault(u.unit_type.value, []).append(pu_scores[i])
                    except Exception:
                        pass
                for utype, scores in all_type_scores.items():
                    adv_stats[f"suca_unit/{utype}_mean"] = float(np.mean(scores))
                    adv_stats[f"suca_unit/{utype}_count"] = len(scores)

                # Prompt complexity: number of units per prompt
                n_units = [len(pu) for pu in all_per_unit_data if pu]
                if n_units:
                    adv_stats["suca/avg_units_per_prompt"] = float(np.mean(n_units))
                    adv_stats["suca/max_units_per_prompt"] = max(n_units)

            wandb.log(adv_stats, step=global_step)
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape

        # Skip if no valid samples (all advantages=0, e.g. strict binary reward gave all 0s)
        if total_batch_size == 0:
            if accelerator.is_main_process:
                print(f"[WARNING] Epoch {epoch}: no valid samples (all advantages=0), skipping", flush=True)
            epoch += 1
            continue

        assert num_timesteps == config.sample.num_steps

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # shuffle samples along batch dimension
            perm = torch.randperm(total_batch_size, device=accelerator.device)
            samples = {k: v[perm] for k, v in samples.items()}

            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//config.sample.num_batches_per_epoch, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.transformer.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                    pooled_embeds = torch.cat(
                        [train_neg_pooled_prompt_embeds[:len(sample["pooled_prompt_embeds"])], sample["pooled_prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]
                    pooled_embeds = sample["pooled_prompt_embeds"]

                train_timesteps = [step_index  for step_index in range(num_train_timesteps)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    with accelerator.accumulate(transformer):
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    if config.use_lora:
                                        with transformer.module.disable_adapter():
                                            _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, pooled_embeds, config)
                                    else:
                                        # Full param: skip KL ref (use EMA as implicit reference)
                                        prev_sample_mean_ref = prev_sample_mean.detach()

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        unclipped_loss = -advantages * ratio
                        clipped_loss = -advantages * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )
                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                            kl_loss = torch.mean(kl_loss)
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        accelerator.backward(loss)
                        if accelerator.sync_gradients:
                            # Track gradient norm before clipping
                            grad_norm = torch.nn.utils.clip_grad_norm_(
                                transformer.parameters(), config.train.max_grad_norm
                            )
                            info["grad_norm"].append(grad_norm.detach())
                        optimizer.step()
                        optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                            # Print every gradient step to console
                            logger.info(
                                f"  [Step {global_step}] "
                                f"loss={info['loss']:.4f} policy={info['policy_loss']:.4f} "
                                f"kl={info.get('kl_loss', 0.0):.4f} "
                                f"ratio≈{torch.exp(info['approx_kl'] * 2).item():.4f} "
                                f"clip={info['clipfrac']:.3f}(↑{info['clipfrac_gt_one']:.3f} ↓{info['clipfrac_lt_one']:.3f}) "
                                f"grad={info.get('grad_norm', 0.0):.3f} "
                                f"kl_approx={info['approx_kl']:.6f}"
                            )
                        last_step_info = {k: v.item() if torch.is_tensor(v) else v for k, v in info.items()}
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)
            # make sure we did an optimization step at the end of the inner epoch
            # assert accelerator.sync_gradients
        
        # === End of epoch summary (printed to console for quick monitoring) ===
        if accelerator.is_main_process:
            ori_avg = gathered_rewards.get("ori_avg", np.array([0]))
            r_mean = float(np.mean(ori_avg))
            r_std = float(np.std(ori_avg))
            r_max = float(np.max(ori_avg))
            r_min = float(np.min(ori_avg))
            adv_mean = samples["advantages"].mean().item() if "advantages" in samples else 0.0
            # Use cached last_step_info from training loop
            _lsi = last_step_info if 'last_step_info' in dir() else {}
            logger.info(
                f"[Epoch {epoch}] step={global_step} | "
                f"reward={r_mean:.4f}±{r_std:.4f} (min={r_min:.3f} max={r_max:.3f}) | "
                f"adv={adv_mean:.4f} | valid={mask.sum().item()}/{mask.shape[0]} | "
                f"loss={_lsi.get('loss', 0):.4f} clip={_lsi.get('clipfrac', 0):.3f} "
                f"kl={_lsi.get('approx_kl', 0):.6f} grad={_lsi.get('grad_norm', 0):.3f}"
            )
            # Reward percentile breakdown
            pcts = np.percentile(ori_avg, [10, 25, 50, 75, 90])
            logger.info(
                f"  reward pct: p10={pcts[0]:.3f} p25={pcts[1]:.3f} p50={pcts[2]:.3f} "
                f"p75={pcts[3]:.3f} p90={pcts[4]:.3f}"
            )
        epoch+=1

if __name__ == "__main__":
    app.run(main)

