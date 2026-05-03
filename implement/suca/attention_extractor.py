"""
Joint Attention Extractor for SD3.5 MMDiT: Hooks into the transformer to
capture image-to-text attention maps at each denoising timestep.

In SD3's MMDiT, text and image tokens are concatenated and processed through
joint self-attention. We extract the image->text portion of the attention
matrix, which tells us how much each image patch attends to each text token.

These maps are used to build the unit-timestep responsibility matrix C[k,t].
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionExtractor:
    """
    Registers custom attention processors on SD3's JointTransformerBlock
    layers to capture image-to-text attention weights during denoising.
    """

    def __init__(
        self,
        transformer: nn.Module,
        layer_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            transformer: The SD3Transformer2DModel.
            layer_indices: Which transformer block indices to capture.
                If None, captures all blocks.
        """
        self.transformer = transformer
        self.layer_indices = layer_indices
        # storage: timestep -> list of (layer_idx, attn_map)
        # attn_map shape: (batch, heads, img_tokens, text_tokens)
        self._attention_maps: Dict[int, List[Tuple[int, torch.Tensor]]] = {}
        self._current_timestep: int = 0
        self._target_blocks: List[Tuple[int, str, nn.Module]] = []
        self._original_processors: Dict[int, object] = {}
        self._discover_joint_attention_layers()

    def _discover_joint_attention_layers(self):
        """Find JointTransformerBlock attention modules (attn = joint attention)."""
        blocks = []
        for name, module in self.transformer.named_modules():
            if type(module).__name__ == "JointTransformerBlock":
                # Extract block index from name, e.g. "transformer_blocks.5"
                parts = name.split(".")
                try:
                    idx = int(parts[-1])
                except (ValueError, IndexError):
                    idx = len(blocks)
                blocks.append((idx, name, module))

        if self.layer_indices is not None:
            self._target_blocks = [
                (idx, name, mod) for idx, name, mod in blocks if idx in self.layer_indices
            ]
        else:
            self._target_blocks = blocks

    def set_timestep(self, t: int):
        """Set the current timestep index for attention map storage."""
        self._current_timestep = t

    def register_hooks(self):
        """Replace attention processors on target blocks to capture attention."""
        self.remove_hooks()
        for block_idx, name, block in self._target_blocks:
            attn_module = block.attn  # The joint attention module
            original_proc = attn_module.processor
            self._original_processors[block_idx] = original_proc
            wrapped = _JointAttentionCaptureProcessor(
                original_processor=original_proc,
                extractor=self,
                layer_idx=block_idx,
            )
            attn_module.set_processor(wrapped)

    def remove_hooks(self):
        """Restore original processors."""
        for block_idx, name, block in self._target_blocks:
            if block_idx in self._original_processors:
                block.attn.set_processor(self._original_processors[block_idx])
        self._original_processors.clear()

    def store_attention(self, layer_idx: int, attn_weights: torch.Tensor):
        """Called by the wrapped processor to store attention weights.

        Args:
            layer_idx: transformer block index
            attn_weights: (batch, heads, img_tokens, text_tokens) — the
                image-to-text portion of the joint attention matrix.
        """
        t = self._current_timestep
        if t not in self._attention_maps:
            self._attention_maps[t] = []
        self._attention_maps[t].append(
            (layer_idx, attn_weights.detach().cpu())
        )

    def get_attention_maps(self) -> Dict[int, List[Tuple[int, torch.Tensor]]]:
        """Return all captured attention maps. {timestep -> [(layer_idx, attn_map)]}"""
        return self._attention_maps

    def get_aggregated_attention(
        self,
        token_indices: List[int],
        top_m: int = 64,
    ) -> torch.Tensor:
        """
        Aggregate attention over layers/heads/spatial for given text token indices.

        Returns:
            Tensor of shape (T,) — aggregated attention score per timestep.
        """
        timesteps = sorted(self._attention_maps.keys())
        scores = []
        for t in timesteps:
            layer_maps = self._attention_maps[t]
            t_score = 0.0
            n_layers = 0
            for layer_idx, attn in layer_maps:
                # attn shape: (batch, heads, img_tokens, text_tokens)
                attn_mean = attn.mean(dim=0)  # (heads, img_tokens, text_tokens)
                # Select target text token indices and sum
                if token_indices:
                    token_attn = attn_mean[:, :, token_indices].sum(dim=-1)  # (heads, img_tokens)
                else:
                    token_attn = attn_mean.sum(dim=-1)  # (heads, img_tokens)
                # Top-m spatial (image) positions
                spatial_scores = token_attn.mean(dim=0)  # (img_tokens,)
                if 0 < top_m < spatial_scores.shape[0]:
                    topk = spatial_scores.topk(top_m).values
                    layer_score = topk.mean().item()
                else:
                    layer_score = spatial_scores.mean().item()
                t_score += layer_score
                n_layers += 1
            scores.append(t_score / max(n_layers, 1))
        return torch.tensor(scores, dtype=torch.float32)

    def get_aggregated_attention_for_sample(
        self,
        sample_idx: int,
        token_indices: List[int],
        top_m: int = 64,
    ) -> torch.Tensor:
        """
        Aggregate attention over layers/heads/spatial for given text token indices,
        for a specific sample in the batch.

        Args:
            sample_idx: Index of the sample within the (positive) batch.
            token_indices: Text token indices for the semantic unit.
            top_m: Number of top spatial positions to aggregate.

        Returns:
            Tensor of shape (T,) — aggregated attention score per timestep.
        """
        timesteps = sorted(self._attention_maps.keys())
        scores = []
        for t in timesteps:
            layer_maps = self._attention_maps[t]
            t_score = 0.0
            n_layers = 0
            for layer_idx, attn in layer_maps:
                # attn shape: (batch, heads, img_tokens, text_tokens)
                if sample_idx >= attn.shape[0]:
                    continue
                attn_sample = attn[sample_idx]  # (heads, img_tokens, text_tokens)
                # Select target text token indices and sum
                if token_indices:
                    token_attn = attn_sample[:, :, token_indices].sum(dim=-1)  # (heads, img_tokens)
                else:
                    token_attn = attn_sample.sum(dim=-1)  # (heads, img_tokens)
                # Top-m spatial (image) positions
                spatial_scores = token_attn.mean(dim=0)  # (img_tokens,)
                if 0 < top_m < spatial_scores.shape[0]:
                    topk = spatial_scores.topk(top_m).values
                    layer_score = topk.mean().item()
                else:
                    layer_score = spatial_scores.mean().item()
                t_score += layer_score
                n_layers += 1
            scores.append(t_score / max(n_layers, 1))
        return torch.tensor(scores, dtype=torch.float32)

    def clear(self):
        """Clear stored attention maps."""
        self._attention_maps.clear()

    @contextmanager
    def capture(self):
        """Context manager for attention capture."""
        self.clear()
        self.register_hooks()
        try:
            yield self
        finally:
            self.remove_hooks()


class _JointAttentionCaptureProcessor:
    """
    Wraps SD3's JointAttnProcessor2_0 to capture image-to-text attention.

    In SD3 joint attention:
      - hidden_states = image tokens (from the latent patches)
      - encoder_hidden_states = text tokens (from text encoders)
      - They are projected separately, concatenated, and run through
        joint self-attention: Q,K,V = [img_q; txt_q], [img_k; txt_k], [img_v; txt_v]
      - We extract attention[img_positions, txt_positions] from the full matrix.
    """

    def __init__(
        self,
        original_processor,
        extractor: CrossAttentionExtractor,
        layer_idx: int,
    ):
        self.original_processor = original_processor
        self.extractor = extractor
        self.layer_idx = layer_idx

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, *args, **kwargs):
        if encoder_hidden_states is None:
            # Not joint attention, forward normally
            return self.original_processor(
                attn, hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                *args, **kwargs,
            )

        # --- Replicate JointAttnProcessor2_0 logic but capture attention weights ---
        residual = hidden_states
        batch_size = hidden_states.shape[0]
        img_seq_len = hidden_states.shape[1]
        txt_seq_len = encoder_hidden_states.shape[1]

        # Image projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Text projections
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        txt_query = txt_query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        txt_key = txt_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        txt_value = txt_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Concatenate: [image; text]
        joint_query = torch.cat([query, txt_query], dim=2)
        joint_key = torch.cat([key, txt_key], dim=2)
        joint_value = torch.cat([value, txt_value], dim=2)

        # Compute attention weights manually to capture them
        # shape: (batch, heads, img+txt, head_dim)
        scale = head_dim ** -0.5
        # (batch, heads, img+txt, img+txt)
        attn_weights = torch.matmul(joint_query, joint_key.transpose(-2, -1)) * scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = attn_weights.softmax(dim=-1)

        # Extract image->text attention block:
        # rows = image tokens [0:img_seq_len], cols = text tokens [img_seq_len:]
        img_to_txt_attn = attn_weights[:, :, :img_seq_len, img_seq_len:]
        # shape: (batch, heads, img_tokens, txt_tokens)

        # Only store attention from the conditional (positive) branch
        # In CFG, batch = [negative, positive], we want the positive half
        if batch_size > 1 and batch_size % 2 == 0:
            # CFG: first half is negative, second half is positive
            positive_attn = img_to_txt_attn[batch_size // 2:]
            self.extractor.store_attention(self.layer_idx, positive_attn)
        else:
            self.extractor.store_attention(self.layer_idx, img_to_txt_attn)

        # Compute output using full attention
        hidden_out = torch.matmul(attn_weights, joint_value)
        hidden_out = hidden_out.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_out = hidden_out.to(joint_query.dtype)

        # Split back into image and text
        img_out = hidden_out[:, :img_seq_len]
        txt_out = hidden_out[:, img_seq_len:]

        # Output projections
        img_out = attn.to_out[0](img_out)
        img_out = attn.to_out[1](img_out)

        if not attn.context_pre_only:
            txt_out = attn.to_add_out(txt_out)

        if attn.residual_connection:
            img_out = img_out + residual

        return img_out, txt_out
