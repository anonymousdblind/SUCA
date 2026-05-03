"""
Unit-Level Reward and Uncertainty Estimation.

Uses Qwen3-VL-8B-Instruct as the VQA reward model for superior fine-grained
semantic verification (counting, attribute binding, spatial relations).

For each semantic unit s_k:
  - r_k = P_VLM(Yes | I, q_k)           — unit reward
  - u_k = -Σ P̄(v) log P̄(v)             — unit uncertainty (semantic entropy)
  - r̃_k = r_k - λ_u * u_k              — corrected unit reward
  - A_k = Norm(r̃_k - b_k)              — unit advantage
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


class UnitRewardComputer:
    """
    Computes per-unit VQA rewards using Qwen3-VL.

    Why Qwen3-VL over BLIP-2:
      - Much better counting accuracy (critical for count units)
      - Stronger attribute-object binding (color/material detection)
      - Better spatial relation understanding
      - Supports structured instruction following for precise Yes/No
    """

    def __init__(
        self,
        vlm_model_name: str = "models/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        lambda_u: float = 0.5,
    ):
        self.device = device
        self.lambda_u = lambda_u
        self._vlm = None
        self._vlm_processor = None
        self._vlm_model_name = vlm_model_name

    def _load_vlm(self):
        """Lazy-load VLM to save memory until needed."""
        if self._vlm is not None:
            return

        from transformers import AutoProcessor, AutoModelForImageTextToText

        self._vlm_processor = AutoProcessor.from_pretrained(
            self._vlm_model_name,
            trust_remote_code=True,
        )
        self._vlm = AutoModelForImageTextToText.from_pretrained(
            self._vlm_model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).to(self.device)
        self._vlm.eval()

        # Cache token IDs for Yes/No
        self._yes_token_ids = self._vlm_processor.tokenizer.encode(
            "Yes", add_special_tokens=False
        )
        self._no_token_ids = self._vlm_processor.tokenizer.encode(
            "No", add_special_tokens=False
        )

    @torch.no_grad()
    def compute_unit_rewards(
        self,
        image: Image.Image,
        vqa_questions: List[str],
    ) -> torch.Tensor:
        """
        Compute reward for each VQA question: P(Yes | image, question).
        Batched: all questions processed in one forward pass (image encoded once).

        Args:
            image: Generated PIL image.
            vqa_questions: List of K yes/no questions.

        Returns:
            Tensor of shape (K,) with reward values in [0, 1].
        """
        self._load_vlm()

        if len(vqa_questions) == 0:
            return torch.tensor([], dtype=torch.float32)

        # Build all messages at once
        texts = []
        images_list = []
        for question in vqa_questions:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": (
                                f"Look at this image carefully. {question}\n"
                                "Answer with only 'Yes' or 'No'."
                            ),
                        },
                    ],
                }
            ]
            text_prompt = self._vlm_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text_prompt)
            images_list.append(image)

        # Batch process - all questions in one call
        inputs = self._vlm_processor(
            text=texts,
            images=images_list,
            return_tensors="pt",
            padding=True,
        ).to(self._vlm.device)

        # Single batched generate
        outputs = self._vlm.generate(
            **inputs,
            max_new_tokens=5,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

        # Extract P(Yes) for each question from first token logits
        rewards = []
        first_token_logits = outputs.scores[0]  # [batch_size, vocab_size]
        for i in range(len(vqa_questions)):
            prob = self._extract_yes_prob_from_logits_single(first_token_logits[i])
            if prob is None:
                # Fallback: parse generated text
                gen_ids = outputs.sequences[i][inputs["input_ids"].shape[1]:]
                gen_text = self._vlm_processor.tokenizer.decode(
                    gen_ids, skip_special_tokens=True
                ).strip().lower()
                prob = self._parse_yes_no_text(gen_text)
            rewards.append(prob)

        return torch.tensor(rewards, dtype=torch.float32)

    @torch.no_grad()
    def _extract_yes_prob_from_logits_single(self, logits: torch.Tensor) -> Optional[float]:
        """Extract P(Yes) from a single sample's first-token logits."""
        yes_logits = logits[self._yes_token_ids].max()
        no_logits = logits[self._no_token_ids].max()
        probs = F.softmax(torch.stack([yes_logits, no_logits]), dim=0)
        return probs[0].item()

    @torch.no_grad()
    def _vqa_yes_probability(self, image: Image.Image, question: str) -> float:
        """
        Compute P(Yes | image, question) using Qwen3-VL.

        Uses two approaches and takes the more reliable one:
          1. Token-level logit: P(Yes) vs P(No) from first generated token
          2. Text-level fallback: parse the generated answer if logits are ambiguous
        """
        # Build Qwen3-VL conversation format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            f"Look at this image carefully. {question}\n"
                            "Answer with only 'Yes' or 'No'."
                        ),
                    },
                ],
            }
        ]

        # Process with Qwen3-VL processor
        text_prompt = self._vlm_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._vlm_processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(self._vlm.device)

        # Generate with logit output
        outputs = self._vlm.generate(
            **inputs,
            max_new_tokens=10,
            output_scores=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

        # Method 1: Token logit probability
        prob_yes = self._extract_yes_prob_from_logits(outputs.scores)

        # Method 2: Text fallback if logits are uncertain
        if prob_yes is None:
            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            generated_text = self._vlm_processor.tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip().lower()
            prob_yes = self._parse_yes_no_text(generated_text)

        return prob_yes

    def _extract_yes_prob_from_logits(
        self, scores: Tuple[torch.Tensor, ...]
    ) -> Optional[float]:
        """Extract P(Yes) from the first token's logits."""
        if not scores:
            return None

        first_logits = scores[0][0]  # (vocab_size,)

        # Try all possible Yes/No token IDs
        yes_logits = []
        no_logits = []
        for tid in self._yes_token_ids:
            if tid < first_logits.shape[0]:
                yes_logits.append(first_logits[tid])
        for tid in self._no_token_ids:
            if tid < first_logits.shape[0]:
                no_logits.append(first_logits[tid])

        if not yes_logits or not no_logits:
            return None

        yes_logit = max(yes_logits)
        no_logit = max(no_logits)

        probs = F.softmax(torch.stack([yes_logit, no_logit]), dim=0)
        return probs[0].item()

    @staticmethod
    def _parse_yes_no_text(text: str) -> float:
        """Parse Yes/No from generated text as fallback."""
        text = text.strip().lower()
        if text.startswith("yes"):
            return 1.0
        elif text.startswith("no"):
            return 0.0
        # Fuzzy matching
        yes_count = len(re.findall(r"\byes\b", text))
        no_count = len(re.findall(r"\bno\b", text))
        if yes_count > no_count:
            return 0.8
        elif no_count > yes_count:
            return 0.2
        return 0.5  # Truly ambiguous

    def compute_unit_uncertainty(
        self,
        images: List[Image.Image],
        vqa_questions: List[str],
    ) -> torch.Tensor:
        """
        Compute per-unit semantic entropy from multiple sampled images.

        u_k = -Σ_{v ∈ {Yes,No}} P̄_k(v) log P̄_k(v)
        where P̄_k(v) = (1/N) Σ_n P_VLM(v | I_n, q_k)

        Args:
            images: List of N sampled images for the same prompt.
            vqa_questions: List of K VQA questions.

        Returns:
            Tensor of shape (K,) with entropy values.
        """
        self._load_vlm()
        N = len(images)
        K = len(vqa_questions)

        # Collect P(Yes) for each image × question
        all_probs = torch.zeros(N, K)
        for n, img in enumerate(images):
            all_probs[n] = self.compute_unit_rewards(img, vqa_questions)

        # Average probability per unit
        p_bar_yes = all_probs.mean(dim=0)  # (K,)
        p_bar_no = 1.0 - p_bar_yes

        # Binary entropy
        entropy = torch.zeros(K)
        for k in range(K):
            py, pn = p_bar_yes[k].item(), p_bar_no[k].item()
            h = 0.0
            if py > 1e-8:
                h -= py * math.log(py)
            if pn > 1e-8:
                h -= pn * math.log(pn)
            entropy[k] = h

        return entropy

    def compute_corrected_rewards(
        self,
        image: Image.Image,
        images_for_uncertainty: List[Image.Image],
        vqa_questions: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute corrected rewards: r̃_k = r_k - λ_u * u_k

        Args:
            image: The primary generated image.
            images_for_uncertainty: Multiple samples for uncertainty estimation.
            vqa_questions: List of K VQA questions.

        Returns:
            (corrected_rewards, raw_rewards, uncertainties) — each of shape (K,)
        """
        raw_rewards = self.compute_unit_rewards(image, vqa_questions)
        uncertainties = self.compute_unit_uncertainty(images_for_uncertainty, vqa_questions)
        corrected = raw_rewards - self.lambda_u * uncertainties
        return corrected, raw_rewards, uncertainties

    @staticmethod
    def compute_unit_advantages(
        corrected_rewards: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute normalized unit-level advantages.

        A_k = Norm(r̃_k - b_k)

        Args:
            corrected_rewards: Tensor of shape (K,) or (batch, K).
            baseline: Optional baseline of shape (K,). If None, uses batch mean.
            eps: Small constant for numerical stability.

        Returns:
            Advantages of same shape as corrected_rewards.
        """
        if corrected_rewards.dim() == 1:
            if baseline is not None:
                adv = corrected_rewards - baseline
            else:
                adv = corrected_rewards - corrected_rewards.mean()
            std = adv.std()
            if std > eps:
                adv = adv / std
            return adv
        else:
            if baseline is not None:
                adv = corrected_rewards - baseline.unsqueeze(0)
            else:
                adv = corrected_rewards - corrected_rewards.mean(dim=0, keepdim=True)
            std = adv.std(dim=0, keepdim=True)
            adv = adv / (std + eps)
            return adv
