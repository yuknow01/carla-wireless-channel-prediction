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


class SensorFrameSummarizer(nn.Module):
    """Summarize per-frame spatial sensor tokens into one token per frame."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_time_offset: bool = True,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out_norm = nn.LayerNorm(embed_dim)
        self.time_proj = (
            nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
            if use_time_offset
            else None
        )

    def forward(
        self,
        frame_tokens: torch.Tensor,
        time_offsets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        frame_tokens: (B, T, N, D), where N is spatial tokens per frame.
        time_offsets: (B, T, 1), optional seconds since each frame.

        Returns: (B, T, D)
        """
        B, T, N, D = frame_tokens.shape
        flat = frame_tokens.reshape(B * T, N, D)
        query = self.query.expand(B * T, -1, -1)
        summary, _ = self.attn(
            self.q_norm(query),
            self.kv_norm(flat),
            self.kv_norm(flat),
        )
        summary = summary.squeeze(1).reshape(B, T, D)
        if time_offsets is not None and self.time_proj is not None:
            summary = summary + self.time_proj(
                time_offsets.to(device=summary.device, dtype=summary.dtype)
            )
        return self.out_norm(summary)


class PerTimeModalityFusion(nn.Module):
    """Fuse modalities independently at each time step with a learnable query."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_steps: int = 64,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.query = nn.Parameter(torch.zeros(1, max_steps, 1, embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = GatedFFN(embed_dim=embed_dim, mlp_ratio=4.0, dropout=dropout)

    def forward(
        self,
        channel_tokens: torch.Tensor,
        sensor_tokens: torch.Tensor,
        sensor_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        channel_tokens: (B, K, D)
        sensor_tokens:  (B, K, D) or (B, K, M, D), aligned to channel_tokens.
        sensor_valid_mask: (B, K) or (B, K, M) bool, True when sensor token is valid.

        Returns: (B, K, D)
        """
        B, K, D = channel_tokens.shape
        if K > self.max_steps:
            raise ValueError(f"K={K} exceeds max_steps={self.max_steps}")

        if sensor_tokens.dim() == 3:
            sensor_tokens = sensor_tokens.unsqueeze(2)
        elif sensor_tokens.dim() != 4:
            raise ValueError(f"sensor_tokens must be 3D or 4D, got {tuple(sensor_tokens.shape)}")

        modalities = torch.cat([channel_tokens.unsqueeze(2), sensor_tokens], dim=2)
        num_modalities = modalities.size(2)
        key_padding_mask = torch.zeros(
            B, K, num_modalities, dtype=torch.bool, device=channel_tokens.device
        )
        if sensor_valid_mask is not None:
            if sensor_valid_mask.dim() == 2:
                sensor_valid_mask = sensor_valid_mask.unsqueeze(-1)
            key_padding_mask[:, :, 1:] = ~sensor_valid_mask.bool()

        q = self.query[:, :K].expand(B, -1, -1, -1).reshape(B * K, 1, D)
        kv = modalities.reshape(B * K, num_modalities, D)
        mask = key_padding_mask.reshape(B * K, num_modalities)
        fused, _ = self.attn(
            self.q_norm(q),
            self.kv_norm(kv),
            self.kv_norm(kv),
            key_padding_mask=mask,
        )
        fused = fused.reshape(B, K, D)
        return self.ffn(fused)


__all__ = ["GatedCrossModalFusion", "SensorFrameSummarizer", "PerTimeModalityFusion"]
