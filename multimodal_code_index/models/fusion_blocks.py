"""
models/fusion_blocks.py
=======================
Shared fusion blocks for multimodal channel prediction models.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.chiron_channel import GatedFFN


class GatedCrossModalFusion(nn.Module):
    """Channel tokens cross-attend to sensor tokens with gated residual."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_drop = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid(),
        )

        self.ffn = GatedFFN(embed_dim=embed_dim, mlp_ratio=4.0, dropout=dropout)

    def forward(
        self,
        channel_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        image_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        channel_tokens: (B, S, D)
        image_tokens:   (B, N, D), where N can be image, lidar, or concatenated sensor tokens

        Returns: (B, S, D)
        """
        residual = channel_tokens

        q = self.q_norm(channel_tokens)
        kv = self.kv_norm(image_tokens)
        attn_out, _ = self.cross_attn(
            q, kv, kv,
            key_padding_mask=image_key_padding_mask,
        )
        attn_out = self.attn_drop(attn_out)

        gate_input = torch.cat([residual, attn_out], dim=-1)
        g = self.gate(gate_input)
        channel_tokens = residual + g * attn_out

        return self.ffn(channel_tokens)


__all__ = ["GatedCrossModalFusion"]
