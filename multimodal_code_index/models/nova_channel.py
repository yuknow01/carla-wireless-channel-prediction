"""
models/nova_channel.py
======================
NOVA — Non-Autoregressive Optimised Variable-sequence Architecture

Designed to surpass LWM-Temporal (-48.14 dB) for the CARLA-Wireless dataset
(28 GHz, 16-ant ULA, 64 subcarriers, K=16 → P=4).

Root causes of Chiron's underperformance that NOVA fixes:
  1. Autoregressive rollout  → NOVA uses direct P-step prediction
  2. Coarse patches (pw=32)  → NOVA uses pw=8 (S=32 vs S=8)
  3. Single conv kernel      → NOVA uses multi-scale temporal conv (k=3,7)
  4. Single-round head       → NOVA uses 2-round cross-attn prediction head

Architecture:
    channel_history (B, K, Na, Nsc, 2)
        |
        v
    PatchEmbed2D [ph=4, pw=8] → (B*K, S, D)     S = (Na/ph)*(Nsc/pw) = 4*8 = 32
    + spatial_pos (1, 1, S, D) + temporal_pos (1, K, 1, D)
        |
        v  flatten → (B, K*S, D)   [16*32 = 512 tokens]
        |
        v
    [NOVABlock × depth]:
        NOVATemporalBlock  (multi-scale gated conv + bidirectional attn, over K axis)
        SpatialBlock       (self-attention over S patches per frame)
        GatedFFN           (SwiGLU)
        |
        v
    final_norm
        |
        v
    RefinedPredictionHead
        P learnable queries × 2-round cross-attend to K*S tokens
        → MLP → (B, P, Na, Nsc, 2)

References
----------
- Chiron (chiron_channel.py): bidirectional factorised ST-attention backbone
- LWM-Temporal (lwm_temporal.py): sparse ST-attention with masked future prediction
- CPMamba (arXiv 2407.14440): multi-step channel prediction benchmarks
- Flash-Attention / SwiGLU: standard efficiency components
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Patch Embedding (same as Chiron)
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    """Tokenise one (Na, Nsc, 2) channel frame into S patch vectors.

    Input : (B, Na, Nsc, 2)
    Output: (B, S, D)        S = (Na // ph) * (Nsc // pw)
    """

    def __init__(
        self,
        num_antennas: int = 16,
        num_subcarriers: int = 64,
        patch_h: int = 4,
        patch_w: int = 8,
        embed_dim: int = 256,
    ):
        super().__init__()
        assert num_antennas  % patch_h == 0, "patch_h must divide num_antennas"
        assert num_subcarriers % patch_w == 0, "patch_w must divide num_subcarriers"

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.grid_h = num_antennas // patch_h
        self.grid_w = num_subcarriers // patch_w
        self.num_patches = self.grid_h * self.grid_w  # S
        patch_dim = patch_h * patch_w * 2  # real + imag channels

        self.proj = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, Na, Nsc, 2) → (B, S, D)"""
        B = x.size(0)
        x = x.view(B, self.grid_h, self.patch_h, self.grid_w, self.patch_w, 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()   # (B, gh, gw, ph, pw, 2)
        x = x.view(B, self.num_patches, -1)              # (B, S, ph*pw*2)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Multi-Scale Gated Temporal Conv
# ---------------------------------------------------------------------------

class MultiScaleGatedTemporalConv(nn.Module):
    """Parallel depthwise convolutions at two temporal scales, fused by gating.

    Combines a short-range kernel (k_small) and a medium-range kernel (k_large)
    to capture both fast Doppler fluctuations and slower channel trends.

    Input:  (N, K, D)
    Output: (N, K, D)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        k_small: int = 3,
        k_large: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)

        # Gate projection: splits input into value + gate for each scale + blend
        # Output: [x_s, x_l, gate_s, gate_l, blend_gate] → 5 * D/1
        self.gate_proj = nn.Linear(embed_dim, embed_dim * 4)  # value_s, value_l, g_s, g_l

        # Two parallel symmetric depthwise convolutions
        self.conv_s = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=k_small,
            padding=(k_small - 1) // 2,
            groups=embed_dim,
        )
        self.conv_l = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=k_large,
            padding=(k_large - 1) // 2,
            groups=embed_dim,
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, K, D) → (N, K, D)"""
        residual = x
        x = self.norm(x)

        # Split into four streams
        gs = self.gate_proj(x)              # (N, K, 4D)
        vs, vl, g_s, g_l = gs.chunk(4, dim=-1)   # each (N, K, D)

        # Depthwise convolutions
        cs = self.conv_s(vs.transpose(1, 2)).transpose(1, 2)  # (N, K, D)
        cl = self.conv_l(vl.transpose(1, 2)).transpose(1, 2)  # (N, K, D)

        # Element-wise gating (GLU-style)
        cs = cs * torch.sigmoid(g_s)
        cl = cl * torch.sigmoid(g_l)

        # Fuse: learnable combination via output projection
        out = self.out_proj(cs + cl)
        return residual + self.drop(out)


# ---------------------------------------------------------------------------
# Temporal Block (multi-scale conv + bidirectional attention)
# ---------------------------------------------------------------------------

class NOVATemporalBlock(nn.Module):
    """Temporal modeling along K axis at each spatial position.

    Stage 1: multi-scale gated conv (fast, local)
    Stage 2: bidirectional self-attention (global, over all K frames)

    Operates on (B*S, K, D) internally.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        k_small: int = 3,
        k_large: int = 7,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention

        self.conv = MultiScaleGatedTemporalConv(
            embed_dim=embed_dim,
            k_small=k_small,
            k_large=k_large,
            dropout=dropout,
        )

        if use_attention:
            self.attn_norm = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, K: int, S: int) -> torch.Tensor:
        """x: (B, K*S, D) → (B, K*S, D)  [temporal axis]"""
        B, _, D = x.shape
        # Reshape to (B*S, K, D) for per-spatial temporal processing
        x = x.view(B, K, S, D).permute(0, 2, 1, 3).reshape(B * S, K, D)

        x = self.conv(x)                     # multi-scale conv
        if self.use_attention:
            residual = x
            attn_out, _ = self.attn(self.attn_norm(x), self.attn_norm(x), self.attn_norm(x))
            x = residual + self.attn_drop(attn_out)

        # Reshape back to (B, K*S, D)
        x = x.view(B, S, K, D).permute(0, 2, 1, 3).reshape(B, K * S, D)
        return x


# ---------------------------------------------------------------------------
# Spatial Block (unchanged from Chiron, works well)
# ---------------------------------------------------------------------------

class SpatialBlock(nn.Module):
    """Self-attention over S spatial patches within each time frame.

    Operates on (B*K, S, D) internally.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, K: int, S: int) -> torch.Tensor:
        """x: (B, K*S, D) → (B, K*S, D)  [spatial axis]"""
        B, _, D = x.shape
        residual = x
        x = x.view(B, K, S, D).reshape(B * K, S, D)
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.drop(attn_out)
        x = x.view(B, K, S, D).reshape(B, K * S, D)
        return residual + x


# ---------------------------------------------------------------------------
# Gated FFN (SwiGLU)
# ---------------------------------------------------------------------------

class GatedFFN(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, embed_dim: int = 256, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.w1 = nn.Linear(embed_dim, hidden)
        self.w2 = nn.Linear(embed_dim, hidden)
        self.w3 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        return residual + self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))


# ---------------------------------------------------------------------------
# NOVA Block
# ---------------------------------------------------------------------------

class NOVABlock(nn.Module):
    """One NOVA block: MultiScaleTemporal + Spatial + GatedFFN."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        k_small: int = 3,
        k_large: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_temporal_attention: bool = True,
    ):
        super().__init__()
        self.temporal = NOVATemporalBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            k_small=k_small,
            k_large=k_large,
            dropout=dropout,
            use_attention=use_temporal_attention,
        )
        self.spatial = SpatialBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ffn = GatedFFN(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, K: int, S: int) -> torch.Tensor:
        x = self.temporal(x, K, S)
        x = self.spatial(x, K, S)
        x = self.ffn(x)
        return x


# ---------------------------------------------------------------------------
# Refined Prediction Head (2-round cross-attention)
# ---------------------------------------------------------------------------

class RefinedPredictionHead(nn.Module):
    """Decode K*S encoder tokens → P future channel matrices.

    Two-round cross-attention for iterative refinement:
      Round 1: P queries cross-attend to K*S context tokens
      Round 2: refined queries cross-attend again (residual)
    Then: shared MLP decodes each step independently.

    This is strictly more expressive than Chiron's single-round head.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_antennas: int = 16,
        num_subcarriers: int = 64,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        prediction_horizon: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.prediction_horizon = prediction_horizon
        output_dim = num_antennas * num_subcarriers * 2

        # P learnable queries (one per future step)
        self.queries = nn.Parameter(torch.zeros(1, prediction_horizon, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        # Round 1: cross-attention (queries → context)
        self.ctx_norm1 = nn.LayerNorm(embed_dim)
        self.cross_attn1 = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_drop1 = nn.Dropout(dropout)

        # Self-attention among P queries (allow future steps to share information)
        self.self_norm = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.self_drop = nn.Dropout(dropout)

        # Round 2: second cross-attention for refinement
        self.ctx_norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn2 = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_drop2 = nn.Dropout(dropout)

        # Shared MLP applied per prediction step
        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, K*S, D) → (B, P, Na, Nsc, 2)"""
        B = tokens.size(0)
        q = self.queries.expand(B, -1, -1)        # (B, P, D)

        # Round 1: queries attend to all K*S context tokens
        ctx = self.ctx_norm1(tokens)
        q2, _ = self.cross_attn1(q, ctx, ctx)     # (B, P, D)
        q = q + self.cross_drop1(q2)

        # Self-attention: adjacent prediction steps share context
        q2, _ = self.self_attn(self.self_norm(q), self.self_norm(q), self.self_norm(q))
        q = q + self.self_drop(q2)

        # Round 2: refined queries attend to context again
        ctx = self.ctx_norm2(tokens)
        q2, _ = self.cross_attn2(q, ctx, ctx)     # (B, P, D)
        q = q + self.cross_drop2(q2)

        # Decode per step via shared MLP
        out = self.mlp(q)                          # (B, P, Na*Nsc*2)
        return out.view(B, self.prediction_horizon, self.num_antennas, self.num_subcarriers, 2)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class NOVAChannelPredictor(nn.Module):
    """
    NOVA Channel Predictor.

    Non-autoregressive factorised spatio-temporal transformer with
    multi-scale temporal convolution and refined cross-attention head.

    Key improvements over Chiron:
      1. Direct P-step prediction (no autoregressive rollout)
      2. Finer patches: pw=8 → S=32 (vs pw=32 → S=8 in Chiron)
      3. Multi-scale temporal conv (k=3, k=7 parallel)
      4. 2-round cross-attention prediction head

    Input:  (B, K, Na, Nsc, 2)
    Output: (B, P, Na, Nsc, 2)
    """

    def __init__(
        self,
        num_bs_antennas: int = 16,
        num_subcarriers: int = 64,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        history_len: int = 16,
        prediction_horizon: int = 4,
        patch_h: int = 4,
        patch_w: int = 8,
        k_small: int = 3,
        k_large: int = 7,
        mlp_ratio: float = 4.0,
        prediction_head_hidden: int = 1024,
        dropout: float = 0.1,
        use_temporal_attention: bool = True,
    ):
        super().__init__()
        self.num_bs_antennas = num_bs_antennas
        self.num_subcarriers = num_subcarriers
        self.embed_dim = embed_dim
        self.depth = depth
        self.history_len = history_len
        self.prediction_horizon = prediction_horizon

        # Patch embedding
        self.patch_embed = PatchEmbed2D(
            num_antennas=num_bs_antennas,
            num_subcarriers=num_subcarriers,
            patch_h=patch_h,
            patch_w=patch_w,
            embed_dim=embed_dim,
        )
        S = self.patch_embed.num_patches  # = 32 for default config

        # Separate temporal and spatial positional embeddings
        self.temporal_pos = nn.Parameter(torch.zeros(1, history_len, 1, embed_dim))
        self.spatial_pos  = nn.Parameter(torch.zeros(1, 1, S, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos,  std=0.02)

        # Backbone
        self.blocks = nn.ModuleList([
            NOVABlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                k_small=k_small,
                k_large=k_large,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_temporal_attention=use_temporal_attention,
            )
            for _ in range(depth)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

        # Prediction head
        self.head = RefinedPredictionHead(
            embed_dim=embed_dim,
            num_antennas=num_bs_antennas,
            num_subcarriers=num_subcarriers,
            hidden_dim=prediction_head_hidden,
            dropout=dropout,
            prediction_horizon=prediction_horizon,
            num_heads=num_heads,
        )

        self._S = S
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        channel_history: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        channel_history : (B, K, Na, Nsc, 2)

        Returns
        -------
        pred : (B, P, Na, Nsc, 2)
        """
        B, K, Na, Nsc, _ = channel_history.shape
        S = self._S

        # Patch-embed each frame
        x = channel_history.reshape(B * K, Na, Nsc, 2)
        tokens = self.patch_embed(x)           # (B*K, S, D)
        tokens = tokens.view(B, K, S, self.embed_dim)

        # Add positional embeddings
        tokens = tokens + self.temporal_pos[:, :K] + self.spatial_pos

        # Flatten to token sequence
        tokens = tokens.reshape(B, K * S, self.embed_dim)

        # Backbone
        for block in self.blocks:
            tokens = block(tokens, K, S)

        tokens = self.final_norm(tokens)

        # Refined prediction head: (B, K*S, D) → (B, P, Na, Nsc, 2)
        return self.head(tokens)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "nova_channel",
            "num_bs_antennas": self.num_bs_antennas,
            "num_subcarriers": self.num_subcarriers,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "history_len": self.history_len,
            "prediction_horizon": self.prediction_horizon,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NOVAChannelPredictor":
        config = {k: v for k, v in config.items() if k != "model_type"}
        return cls(**config)
