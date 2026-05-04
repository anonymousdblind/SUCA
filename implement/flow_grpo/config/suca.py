"""
SUCA config — restored to the working full-param version that ran 3900+ steps.
"""
import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))
grpo = imp.load_source("grpo", os.path.join(os.path.dirname(__file__), "grpo.py"))


def _common_config():
    """Full param bf16 — matches the version that ran 3900+ steps successfully."""
    config = grpo.compressibility()

    # Model — full parameter (no LoRA)
    config.pretrained.model = "models/stable-diffusion-3.5-medium"
    config.use_lora = False
    config.lora_rank = 0

    # Sampling
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5
    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = 4
    config.sample.test_batch_size = 4
    config.sample.same_latent = False
    config.sample.global_std = True
    config.sample.noise_level = 0.4
    config.sample.kl_reward = 0

    # Training — same as the version that ran 3900+ steps
    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1       # 1 pass (2 was too aggressive)
    config.activation_checkpointing = True
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.1                 # stronger KL constraint
    config.train.learning_rate = 1e-6       # 10x smaller
    config.train.clip_range = 5e-4          # tighter PPO clip
    config.train.clip_range_gt = 1e-4
    config.train.clip_range_lt = 1e-4
    config.train.adv_clip_max = 3           # smaller advantage clip
    config.train.max_grad_norm = 0.1        # stricter grad clip
    config.train.ema = True
    config.train.sft = 0.0

    # Rewards — default to ImageReward (global quality) + suca_vqa margin (per-unit, for SUCA credit).
    # In the current single-environment pipeline, ImageReward may be incompatible with the
    # transformers version required by Qwen3-VL. In that case the launcher sets
    # SUCA_DISABLE_IMAGEREWARD=1 and we fall back to a suca_vqa-only reward stack.
    if os.environ.get("SUCA_DISABLE_IMAGEREWARD", "0") == "1":
        config.reward_fn = {"suca_vqa": 1.0}
    else:
        # suca_vqa now returns log-margin (range ~[-5,+10]), weight 0.2 to balance with ImageReward [-2,+2]
        config.reward_fn = {"imagereward": 1.0, "suca_vqa": 0.2}
    config.prompt_fn = "geneval"
    config.per_prompt_stat_tracking = True

    # Dataset — T2I-CompBench: compositional prompts (color/shape/texture/spatial/numeracy/complex)
    config.dataset = os.path.join(os.getcwd(), "dataset/t2i_compbench")
    config.sft_warmup_path = "outputs/warmup/checkpoint-7000/transformer"

    # SUCA params
    config.suca_tau = 0.1
    config.suca_top_m_spatial = 3
    config.suca_attention_layers = [5, 10, 15, 20]
    _ports_str = os.environ.get("SUCA_REWARD_PORTS", "8100,8101")
    config.suca_reward_ports = [int(p) for p in _ports_str.split(",")]
    config.suca_vlm_model = "Qwen3-VL-8B-Instruct"

    # Sparse Process Reward: score intermediate images at anchor timesteps
    config.process_reward = True
    config.process_anchor_steps = [3, 7]   # t=3 (30%, structure), t=7 (70%, attributes)
    config.process_lambda = 0.3            # weight of process reward vs terminal
    config.process_confidence_gate = [0.3, 0.7]  # VQA probs in this range are uninformative

    # Logging
    config.mixed_precision = "bf16"
    config.num_epochs = 100000
    config.save_freq = 30   # save every 30 epochs
    config.eval_freq = 3    # eval every 3 epochs
    config.num_checkpoint_limit = 2  # keep last 2 + best-ood
    config.resume_from = ""
    config.train.lora_path = None

    return config


def ablation_no_suca():
    """Full param Flow-GRPO with ImageReward."""
    config = _common_config()
    config.suca_enabled = False
    config.save_dir = "logs/imagereward_grpo"
    config.run_name = "imagereward-grpo-lr1e6"
    return config


def ablation_with_suca():
    """Full param Flow-GRPO WITH SUCA — ImageReward + per-unit credit assignment."""
    config = _common_config()
    config.suca_enabled = True
    config.save_dir = "logs/suca_compbench"
    config.run_name = "suca-compbench-imagereward"
    return config


def suca_compositional():
    """SUCA with relaxed clip + compositional training data (same dist as eval).
    NOTE: Deprecated — clip=0.02 too loose, PPO constraint ineffective. Use suca_sweep_* instead.
    """
    config = _common_config()
    config.suca_enabled = True

    # --- Relaxed training: let the model actually update ---
    config.train.learning_rate = 1e-5       # was 1e-6 (10x larger)
    config.train.clip_range = 0.02          # was 5e-4 (40x looser)
    config.train.clip_range_gt = 0.02
    config.train.clip_range_lt = 0.02
    config.train.max_grad_norm = 1.0        # was 0.1 (10x looser)
    config.train.beta = 0.01                # was 0.1 (less KL penalty)

    # --- Compositional training data (T2I-CompBench) ---
    config.dataset = os.path.join(os.getcwd(), "dataset/t2i_compbench")

    config.save_dir = "logs/suca_compositional"
    config.run_name = "suca-compbench-relaxed"
    return config


# =========================================================================
# Exp 11: PPO Tightening Sweep (Round 1)
# Fixed: group_size=8, process_reward on, lambda=0.3, lr=1e-5
# Sweep: clip × beta
# =========================================================================

def _sweep_base():
    """Common base for all sweep configs: process reward + group_size=8."""
    config = _common_config()
    config.suca_enabled = True

    # Fixed: group_size=8 (Exp 10 showed 16 is OOM and no better)
    config.sample.num_image_per_prompt = 8

    # Fixed: process reward from Exp 9
    config.process_reward = True
    config.process_anchor_steps = [3, 7]
    config.process_lambda = 0.3
    config.process_confidence_gate = [0.3, 0.7]

    # Training base — moderate (not relaxed)
    config.train.learning_rate = 1e-5
    config.train.max_grad_norm = 1.0
    config.train.num_inner_epochs = 1

    # Compositional data
    config.dataset = os.path.join(os.getcwd(), "dataset/t2i_compbench")

    return config


def suca_sweep_clip003_beta003():
    config = _sweep_base()
    config.train.clip_range = 0.003
    config.train.clip_range_gt = 0.003
    config.train.clip_range_lt = 0.003
    config.train.beta = 0.03
    config.save_dir = "logs/sweep_r1/clip003_beta003"
    config.run_name = "sweep-clip003-beta003"
    return config


def suca_sweep_clip003_beta005():
    config = _sweep_base()
    config.train.clip_range = 0.003
    config.train.clip_range_gt = 0.003
    config.train.clip_range_lt = 0.003
    config.train.beta = 0.05
    config.save_dir = "logs/sweep_r1/clip003_beta005"
    config.run_name = "sweep-clip003-beta005"
    return config


def suca_sweep_clip005_beta003():
    config = _sweep_base()
    config.train.clip_range = 0.005
    config.train.clip_range_gt = 0.005
    config.train.clip_range_lt = 0.005
    config.train.beta = 0.03
    config.save_dir = "logs/sweep_r1/clip005_beta003"
    config.run_name = "sweep-clip005-beta003"
    return config


def suca_sweep_clip005_beta005():
    config = _sweep_base()
    config.train.clip_range = 0.005
    config.train.clip_range_gt = 0.005
    config.train.clip_range_lt = 0.005
    config.train.beta = 0.05
    config.save_dir = "logs/sweep_r1/clip005_beta005"
    config.run_name = "sweep-clip005-beta005"
    return config


def suca_sweep_clip008_beta003():
    config = _sweep_base()
    config.train.clip_range = 0.008
    config.train.clip_range_gt = 0.008
    config.train.clip_range_lt = 0.008
    config.train.beta = 0.03
    config.save_dir = "logs/sweep_r1/clip008_beta003"
    config.run_name = "sweep-clip008-beta003"
    return config


def suca_sweep_clip008_beta005():
    config = _sweep_base()
    config.train.clip_range = 0.008
    config.train.clip_range_gt = 0.008
    config.train.clip_range_lt = 0.008
    config.train.beta = 0.05
    config.save_dir = "logs/sweep_r1/clip008_beta005"
    config.run_name = "sweep-clip008-beta005"
    return config


# =========================================================================
# Round 2 (after Round 1 best clip/beta is known): lambda × lr sweep
# Placeholder — update BEST_CLIP/BEST_BETA after Round 1
# =========================================================================

def _sweep_r2_base(process_lambda, lr):
    """Round 2 base: sweep lambda and lr with best clip/beta from Round 1."""
    config = _sweep_base()
    # TODO: fill in best clip/beta from Round 1 results
    config.train.clip_range = 0.005       # placeholder — update after R1
    config.train.clip_range_gt = 0.005
    config.train.clip_range_lt = 0.005
    config.train.beta = 0.03              # placeholder — update after R1
    config.train.learning_rate = lr
    config.process_lambda = process_lambda
    config.save_dir = f"logs/sweep_r2/lam{str(process_lambda).replace('.','')}_lr{lr:.0e}"
    config.run_name = f"sweep-lam{process_lambda}-lr{lr:.0e}"
    return config


def suca_sweep_r2_lam02_lr1e5():
    return _sweep_r2_base(0.2, 1e-5)

def suca_sweep_r2_lam03_lr1e5():
    return _sweep_r2_base(0.3, 1e-5)

def suca_sweep_r2_lam04_lr1e5():
    return _sweep_r2_base(0.4, 1e-5)

def suca_sweep_r2_lam02_lr5e6():
    return _sweep_r2_base(0.2, 5e-6)

def suca_sweep_r2_lam03_lr5e6():
    return _sweep_r2_base(0.3, 5e-6)

def suca_sweep_r2_lam04_lr5e6():
    return _sweep_r2_base(0.4, 5e-6)


# Compatibility
def suca_sd3_8gpu():
    return ablation_with_suca()

def suca_sd3_4gpu():
    config = ablation_with_suca()
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 2
    config.train.batch_size = 2
    config.sample.test_batch_size = 2
    return config

def suca_sd3_2gpu():
    config = ablation_with_suca()
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 4
    config.sample.num_batches_per_epoch = 2
    config.train.batch_size = 2
    config.sample.test_batch_size = 2
    return config

def suca_sd3_3gpu():
    config = ablation_with_suca()
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 3
    config.sample.num_batches_per_epoch = 2
    config.train.batch_size = 2
    config.sample.test_batch_size = 2
    return config

def get_config(config_string="suca_sd3_8gpu"):
    configs = {
        "suca_sd3_8gpu": suca_sd3_8gpu,
        "suca_sd3_2gpu": suca_sd3_2gpu,
        "suca_sd3_4gpu": suca_sd3_4gpu,
        "suca_sd3_3gpu": suca_sd3_3gpu,
        "ablation_no_suca": ablation_no_suca,
        "ablation_with_suca": ablation_with_suca,
        "suca_compositional": suca_compositional,
        # Round 1: clip × beta sweep (fixed lambda=0.3, lr=1e-5, group=8)
        "sweep_clip003_beta003": suca_sweep_clip003_beta003,
        "sweep_clip003_beta005": suca_sweep_clip003_beta005,
        "sweep_clip005_beta003": suca_sweep_clip005_beta003,
        "sweep_clip005_beta005": suca_sweep_clip005_beta005,
        "sweep_clip008_beta003": suca_sweep_clip008_beta003,
        "sweep_clip008_beta005": suca_sweep_clip008_beta005,
        # Round 2: lambda × lr sweep (fill best clip/beta after R1)
        "sweep_r2_lam02_lr1e5": suca_sweep_r2_lam02_lr1e5,
        "sweep_r2_lam03_lr1e5": suca_sweep_r2_lam03_lr1e5,
        "sweep_r2_lam04_lr1e5": suca_sweep_r2_lam04_lr1e5,
        "sweep_r2_lam02_lr5e6": suca_sweep_r2_lam02_lr5e6,
        "sweep_r2_lam03_lr5e6": suca_sweep_r2_lam03_lr5e6,
        "sweep_r2_lam04_lr5e6": suca_sweep_r2_lam04_lr5e6,
    }
    return configs.get(config_string, suca_sd3_8gpu)()
