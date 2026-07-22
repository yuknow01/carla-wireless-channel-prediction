"""
models/chiron_channel.py
========================
CHIRON Channel-Only Predictor — Bidirectional Hybrid channel prediction model.

Designed to surpass LWM / LWM-Temporal for the CARLA-Wireless dataset:
  - 28 GHz, 16-element ULA, 512 subcarriers, 2 kHz sampling

Key improvements over LWM-Temporal:
  1. Factorized spatio-temporal processing (temporal O(K²), spatial O(S²))
  2. Bidirectional temporal blocks (symmetric conv + full self-attention)
  3. Gated convolution for efficient temporal sequence modeling
  4. 2D patch tokenization preserving antenna-frequency structure
  5. P learnable queries in prediction head for direct multi-step output

Architecture:
    channel_history (B, K, Na, Nsc, 2)
        |
        v
    PatchEmbed2D -> (B, K*S, D)          S = spatial patches per frame
    + spatial_pos_embed + temporal_pos_embed
        |
        v
    [ChironBlock x L]:
        TemporalBlock  (symmetric conv + bidirectional attention along time)
        SpatialBlock   (self-attention over spatial patches per frame)
        GatedFFN
        |
        v
    PredictionHead  P learnable queries cross-attend to all (K*S) tokens
    -> (B, P, Na, Nsc, 2)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class PatchEmbed2D(nn.Module):
    """Embed 2D channel matrix patches into token vectors.

    Input:  (B, Na, Nsc, 2)  per frame
    Output: (B, S, D)        S = (Na // ph) * (Nsc // pw)
    """

    def __init__(
        self,
        num_antennas: int = 16,
        num_subcarriers: int = 512,
        patch_h: int = 4,
        patch_w: int = 32,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.grid_h = num_antennas // patch_h
        self.grid_w = num_subcarriers // patch_w
        self.num_patches = self.grid_h * self.grid_w
        patch_dim = patch_h * patch_w * 2  # real + imag

        self.proj = nn.Sequential(
            nn.Linear(patch_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, Na, Nsc, 2) -> (B, S, D)"""
        B = x.size(0)
        # Reshape into patches: (B, gh, ph, gw, pw, 2) -> (B, gh*gw, ph*pw*2)
        x = x.view(B, self.grid_h, self.patch_h, self.grid_w, self.patch_w, 2)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, gh, gw, ph, pw, 2)
        x = x.view(B, self.num_patches, -1)  # (B, S, patch_dim)
        return self.proj(x)


# ---------------------------------------------------------------------------
# Temporal Block (bidirectional)
# ---------------------------------------------------------------------------

class TemporalBlock(nn.Module):
    """Process each spatial position across time with bidirectional attention.

    Combines gated symmetric convolution + full (non-causal) self-attention.
    Since all K input frames are observed history, there is no future leakage
    to guard against — bidirectional attention gives strictly more expressiveness.
    Operates on (B*S, K, D) internally.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        conv_kernel: int = 7,
        dropout: float = 0.1,
        use_attention: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention

        # Gated depthwise conv (symmetric padding — bidirectional)
        self.conv_norm = nn.LayerNorm(embed_dim)
        self.conv_gate = nn.Linear(embed_dim, embed_dim * 2)
        self.conv = nn.Conv1d(
            embed_dim, embed_dim,
            kernel_size=conv_kernel,
            padding=(conv_kernel - 1) // 2,  # symmetric, output length == input length
            groups=embed_dim,
        )
        self.conv_proj = nn.Linear(embed_dim, embed_dim)
        self.conv_drop = nn.Dropout(dropout)

        # Bidirectional self-attention (no mask)
        if use_attention:
            self.attn_norm = nn.LayerNorm(embed_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_drop = nn.Dropout(dropout)

    def _conv(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated symmetric depthwise conv. x: (N, K, D)"""
        residual = x
        x = self.conv_norm(x)

        gate_out = self.conv_gate(x)  # (N, K, 2D)
        x_g, gate = gate_out.chunk(2, dim=-1)
        gate = torch.sigmoid(gate)

        x_conv = x_g.transpose(1, 2)  # (N, D, K)
        x_conv = self.conv(x_conv)     # (N, D, K) — symmetric padding keeps length
        x_conv = x_conv.transpose(1, 2)  # (N, K, D)

        x_out = x_conv * gate
        x_out = self.conv_proj(x_out)
        return residual + self.conv_drop(x_out)

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply bidirectional self-attention. x: (N, K, D)"""
        residual = x
        x = self.attn_norm(x)
        attn_out, _ = self.attn(x, x, x)  # no attn_mask — full bidirectional
        return residual + self.attn_drop(attn_out)

    def forward(self, x: torch.Tensor, K: int, S: int) -> torch.Tensor:
        """x: (B, K*S, D) -> (B, K*S, D). Process temporal axis."""
        B, _, D = x.shape
        # Reshape: (B, K, S, D) -> (B*S, K, D) for temporal processing
        x = x.view(B, K, S, D).permute(0, 2, 1, 3).reshape(B * S, K, D)

        x = self._conv(x)
        if self.use_attention:
            x = self._attention(x)

        # Reshape back: (B*S, K, D) -> (B, K*S, D)
        x = x.view(B, S, K, D).permute(0, 2, 1, 3).reshape(B, K * S, D)
        return x


# ---------------------------------------------------------------------------
# Spatial Block
# ---------------------------------------------------------------------------

class SpatialBlock(nn.Module):
    """Self-attention over spatial patches within each time frame.

    Operates on (B*K, S, D) internally.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
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
        """x: (B, K*S, D) -> (B, K*S, D). Process spatial axis per frame."""
        B, _, D = x.shape
        residual = x

        # Reshape: (B, K, S, D) -> (B*K, S, D)
        x = x.view(B, K, S, D).reshape(B * K, S, D)
        x = self.norm(x)
        attn_out, _ = self.attn(x, x, x)
        x = self.drop(attn_out)

        # Reshape back: (B*K, S, D) -> (B, K*S, D)
        x = x.view(B, K, S, D).reshape(B, K * S, D)
        return residual + x


# ---------------------------------------------------------------------------
# Gated FFN
# ---------------------------------------------------------------------------

class GatedFFN(nn.Module):
    """SwiGLU-style gated feed-forward network."""

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
# CHIRON Block
# ---------------------------------------------------------------------------

class ChironBlock(nn.Module):
    """One CHIRON block = Temporal + Spatial + GatedFFN."""

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        conv_kernel: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_temporal_attention: bool = True,
    ):
        super().__init__()
        self.temporal = TemporalBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            conv_kernel=conv_kernel,
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
# Prediction Head (multi-resolution)
# ---------------------------------------------------------------------------

class ChannelPredictionHead(nn.Module):
    """Decode all encoder tokens to P future channel matrices.

    Uses P learnable queries — one per future step — that cross-attend to the
    full (K*S) token sequence.  Each query then independently decodes its step
    through a shared MLP, yielding (B, P, Na, Nsc, 2).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_patches: int = 64,
        num_antennas: int = 16,
        num_subcarriers: int = 512,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
        prediction_horizon: int = 1,
        delta_skip: bool = False,
    ):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.prediction_horizon = prediction_horizon
        self.delta_skip = delta_skip

        # P learnable queries, one per future step
        self.pool_query = nn.Parameter(torch.zeros(1, prediction_horizon, embed_dim))
        self.pool_norm = nn.LayerNorm(embed_dim)
        self.pool_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=4,
            dropout=dropout, batch_first=True,
        )

        # Shared MLP applied independently to each step's pooled vector
        output_dim = num_antennas * num_subcarriers * 2
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        nn.init.trunc_normal_(self.pool_query, std=0.02)

    def zero_init_output(self) -> None:
        """Zero the final projection so the head starts by predicting delta=0.

        With delta_skip=True this makes the model exactly reproduce the last
        input frame (copy-last) at initialization.
        """
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        if final.bias is not None:
            nn.init.zeros_(final.bias)

    def forward(self, tokens: torch.Tensor, last_frame: torch.Tensor | None = None) -> torch.Tensor:
        """tokens: (B, K*S, D) -> (B, P, Na, Nsc, 2)

        last_frame: (B, Na, Nsc, 2) — when delta_skip is enabled, the MLP
        output is treated as a residual on top of this frame.
        """
        B = tokens.size(0)
        P = self.prediction_horizon
        query = self.pool_query.expand(B, -1, -1)  # (B, P, D)
        ctx = self.pool_norm(tokens)
        pooled, _ = self.pool_attn(query, ctx, ctx)  # (B, P, D)
        out = self.mlp(pooled)  # (B, P, Na*Nsc*2)
        out = out.view(B, P, self.num_antennas, self.num_subcarriers, 2)
        if self.delta_skip and last_frame is not None:
            out = out + last_frame.unsqueeze(1)
        return out


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class ChironChannelPredictor(nn.Module):
    """
    CHIRON Channel-Only Predictor.

    Causal, factorized spatio-temporal transformer for channel prediction.
    Designed for: 28 GHz, 16-ant ULA, 512 subcarriers, 2 kHz sampling.

    Interface matches MultiModalPredictator for drop-in compatibility.
    """

    def __init__(
        self,
        num_bs_antennas: int = 16,
        num_subcarriers: int = 512,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        history_len: int = 32,
        prediction_horizon: int = 1,
        patch_h: int = 4,
        patch_w: int = 32,
        conv_kernel: int = 7,
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
        num_spatial = self.patch_embed.num_patches

        # Positional embeddings (separate temporal + spatial)
        self.temporal_pos = nn.Parameter(torch.zeros(1, history_len, 1, embed_dim))
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, num_spatial, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

        # Backbone blocks
        self.blocks = nn.ModuleList([
            ChironBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                conv_kernel=conv_kernel,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_temporal_attention=use_temporal_attention,
            )
            for _ in range(depth)
        ])

        self.final_norm = nn.LayerNorm(embed_dim)

        # Prediction head
        self.head = ChannelPredictionHead(
            embed_dim=embed_dim,
            num_patches=num_spatial,
            num_antennas=num_bs_antennas,
            num_subcarriers=num_subcarriers,
            hidden_dim=prediction_head_hidden,
            dropout=dropout,
            prediction_horizon=prediction_horizon,
        )

        self._num_spatial = num_spatial
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
            K past channel measurements in real representation.

        Returns
        -------
        pred : (B, P, Na, Nsc, 2)
            Predicted P future channels in real representation.
        """
        tokens = self.encode_tokens(channel_history)

        # Head: P learnable queries cross-attend to all K*S tokens
        # tokens: (B, K*S, D) -> pred: (B, P, Na, Nsc, 2)
        return self.head(tokens)

    def encode_tokens(self, channel_history: torch.Tensor) -> torch.Tensor:
        """Backbone encoding without the head: (B,K,Na,Nsc,2) -> (B, K*S, D)."""
        B, K, Na, Nsc, _ = channel_history.shape
        S = self._num_spatial

        x = channel_history  # (B, K, Na, Nsc, 2)

        # Patch embed each frame
        x = x.reshape(B * K, Na, Nsc, 2)
        tokens = self.patch_embed(x)  # (B*K, S, D)
        tokens = tokens.view(B, K, S, self.embed_dim)

        # Add positional embeddings
        tokens = tokens + self.temporal_pos[:, :K] + self.spatial_pos

        # Flatten to sequence: (B, K*S, D)
        tokens = tokens.reshape(B, K * S, self.embed_dim)

        # Backbone
        for block in self.blocks:
            tokens = block(tokens, K, S)

        return self.final_norm(tokens)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "chiron_channel",
            "num_bs_antennas": self.num_bs_antennas,
            "num_subcarriers": self.num_subcarriers,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "history_len": self.history_len,
            "prediction_horizon": self.prediction_horizon,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ChironChannelPredictor":
        config.pop("model_type", None)
        return cls(**config)
