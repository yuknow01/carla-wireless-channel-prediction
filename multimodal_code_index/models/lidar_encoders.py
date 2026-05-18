"""
models/lidar_encoders.py
========================
Shared LiDAR encoders for multimodal channel prediction models.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    """Encode sparse LiDAR point clouds into a fixed set of tokens.

    Inputs:
        points:     (B, max_points, 4), x/y/z/intensity with zero padding
        point_mask: (B, max_points) bool, True for valid points

    Output:
        tokens:     (B, num_tokens, D)
    """

    def __init__(
        self,
        point_dim: int = 4,
        embed_dim: int = 256,
        max_points: int = 64,
        num_tokens: int = 16,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_points = max_points
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        self.point_mlp = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        self.pool_queries = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.pool_norm = nn.LayerNorm(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pool_queries, std=0.02)

    def forward(
        self,
        points: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B = points.size(0)

        feat = self.point_mlp(points)  # (B, N, D)

        if point_mask is not None:
            key_pad = ~point_mask
            fully_masked = key_pad.all(dim=1)
            if fully_masked.any():
                key_pad = key_pad.clone()
                key_pad[fully_masked, 0] = False
        else:
            key_pad = None

        normed = self.attn_norm(feat)
        attn_out, _ = self.self_attn(
            normed, normed, normed,
            key_padding_mask=key_pad,
        )
        feat = feat + attn_out

        queries = self.pool_queries.expand(B, -1, -1)
        q_normed = self.pool_norm(queries)
        kv_normed = self.attn_norm(feat)
        pooled, _ = self.pool_attn(
            q_normed, kv_normed, kv_normed,
            key_padding_mask=key_pad,
        )
        tokens = queries + pooled
        return self.output_norm(tokens), None


__all__ = ["PointNetEncoder"]
