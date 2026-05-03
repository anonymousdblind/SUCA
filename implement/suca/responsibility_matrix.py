"""
Responsibility Matrix Builder: Constructs the unit-timestep responsibility
matrix C ∈ R^{K×T} from cross-attention maps.

C[k,t] represents how much supervision weight timestep t should receive
for semantic unit k.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F

from .attention_extractor import CrossAttentionExtractor
from .semantic_parser import SemanticUnit


class ResponsibilityMatrixBuilder:
    """
    Build the responsibility matrix C[k,t] from cross-attention maps.

    For each semantic unit s_k with token indices T_k:
        a_{k,t} = Agg_{l,h,p}(A_t^{(l,h)}(p, T_k))
        C_{k,t} = softmax(a_{k,t} / tau) over t
    """

    def __init__(self, tau: float = 0.1, top_m_spatial: int = 64):
        """
        Args:
            tau: Temperature for softmax normalization.
            top_m_spatial: Number of top spatial positions to aggregate.
        """
        self.tau = tau
        self.top_m_spatial = top_m_spatial

    def build(
        self,
        extractor: CrossAttentionExtractor,
        units: List[SemanticUnit],
    ) -> torch.Tensor:
        """
        Build responsibility matrix from captured cross-attention maps.

        Args:
            extractor: CrossAttentionExtractor with captured attention maps.
            units: List of semantic units with resolved token_indices.

        Returns:
            C: Tensor of shape (K, T), the responsibility matrix.
        """
        K = len(units)
        attention_maps = extractor.get_attention_maps()
        timesteps = sorted(attention_maps.keys())
        T = len(timesteps)

        if K == 0 or T == 0:
            return torch.ones(max(K, 1), max(T, 1)) / max(T, 1)

        # Build raw attention scores a[k, t]
        a = torch.zeros(K, T, dtype=torch.float32)

        for k, unit in enumerate(units):
            token_idx = unit.token_indices
            score_per_t = extractor.get_aggregated_attention(
                token_indices=token_idx,
                top_m=self.top_m_spatial,
            )
            # score_per_t has length = number of timesteps
            if score_per_t.shape[0] == T:
                a[k] = score_per_t
            else:
                # Pad or truncate
                min_len = min(score_per_t.shape[0], T)
                a[k, :min_len] = score_per_t[:min_len]

        # Softmax over time dimension with temperature
        C = F.softmax(a / self.tau, dim=1)  # (K, T)

        return C

    def build_for_sample(
        self,
        extractor: CrossAttentionExtractor,
        units: List[SemanticUnit],
        sample_idx: int,
    ) -> torch.Tensor:
        """
        Build responsibility matrix for a specific sample in a batched capture.

        Args:
            extractor: CrossAttentionExtractor with captured (batched) attention maps.
            units: List of semantic units with resolved token_indices.
            sample_idx: Index of the sample within the positive batch.

        Returns:
            C: Tensor of shape (K, T), the responsibility matrix.
        """
        K = len(units)
        attention_maps = extractor.get_attention_maps()
        timesteps = sorted(attention_maps.keys())
        T = len(timesteps)

        if K == 0 or T == 0:
            return torch.ones(max(K, 1), max(T, 1)) / max(T, 1)

        # Build raw attention scores a[k, t]
        a = torch.zeros(K, T, dtype=torch.float32)

        for k, unit in enumerate(units):
            token_idx = unit.token_indices
            score_per_t = extractor.get_aggregated_attention_for_sample(
                sample_idx=sample_idx,
                token_indices=token_idx,
                top_m=self.top_m_spatial,
            )
            if score_per_t.shape[0] == T:
                a[k] = score_per_t
            else:
                min_len = min(score_per_t.shape[0], T)
                a[k, :min_len] = score_per_t[:min_len]

        # Softmax over time dimension with temperature
        C = F.softmax(a / self.tau, dim=1)  # (K, T)

        return C

    def build_from_raw_attention(
        self,
        attention_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build responsibility matrix from pre-computed attention scores.

        Args:
            attention_scores: Tensor of shape (K, T) — raw aggregated
                attention scores per unit per timestep.

        Returns:
            C: Tensor of shape (K, T), the responsibility matrix.
        """
        return F.softmax(attention_scores / self.tau, dim=1)

    @staticmethod
    def visualize(
        C: torch.Tensor,
        units: List[SemanticUnit],
        save_path: str = "responsibility_matrix.png",
    ):
        """Save a heatmap visualization of the responsibility matrix."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return

        fig, ax = plt.subplots(figsize=(12, max(3, len(units) * 0.5)))
        im = ax.imshow(C.numpy(), aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("Denoising Timestep")
        ax.set_ylabel("Semantic Unit")
        ax.set_yticks(range(len(units)))
        ax.set_yticklabels(
            [f"{u.unit_type.value}: {u.description[:30]}" for u in units],
            fontsize=8,
        )
        plt.colorbar(im, ax=ax, label="Responsibility Weight")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
