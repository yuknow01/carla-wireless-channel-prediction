"""
models/nova_temporal_channel.py
================================
NOVA-Temporal — Masked future-frame injection for multi-step channel prediction

Key change from NOVA (nova_channel.py):
  [NOVA]:          K history frames → backbone → prediction head → P outputs
  [NOVA-Temporal]: (K + P) frames, P masked → ALL 8 blocks → decode P tokens

Why this closes the gap with LWM_Temporal (-48.14 dB):
  - LWM_Temporal: future tokens attend to history in ALL transformer layers
  - NOVA original: future tokens see history ONLY in the 2-round cross-attn head
  - NOVA-Temporal: depth=8 NOVABlocks × (temporal-conv + temporal-attn +
                   spatial-attn + FFN) give future tokens deep iterative refinement
                   through history context — identical philosophy to LWM_Temporal
                   but with NOVA's stronger multi-scale temporal conv + finer
                   S=32 spatial resolution.

Architecture:
    channel_history (B, K, Na, Nsc, 2)
        │
        ▼  patch_embed each frame
    (B, K, S, D)
        │  concat P learned [MASK] future tokens
        ▼
    (B, K+P, S, D)
        │  + temporal_pos (K+P positions) + spatial_pos
        ▼  flatten → (B, (K+P)*S, D)
        │
    [NOVABlock × depth]  ← future tokens ↔ history tokens at EVERY block
        │
        ▼  final_norm
        │  extract last P*S tokens → (B, P*S, D)
        ▼
    FrameDecoder (2-layer MLP per spatial patch)
        │
        ▼
    (B, P, Na, Nsc, 2)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.nova_channel import PatchEmbed2D, NOVABlock


# ---------------------------------------------------------------------------
# Frame Decoder — per-patch MLP reconstruction
# ---------------------------------------------------------------------------

class FrameDecoder(nn.Module):
    """Decode P*S future tokens → (B, P, Na, Nsc, 2).

    Each of the P*S tokens is independently mapped to its corresponding spatial
    patch via a 2-layer MLP, then the patches are reassembled into the full
    (Na, Nsc, 2) channel frame layout.
    """

    def __init__(
        self,
        embed_dim: int,
        patch_h: int,
        patch_w: int,
        grid_h: int,
        grid_w: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.grid_h  = grid_h
        self.grid_w  = grid_w
        patch_dim = patch_h * patch_w * 2

        self.mlp = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, patch_dim),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        P: int,
        Na: int,
        Nsc: int,
    ) -> torch.Tensor:
        """tokens: (B, P*S, D) → (B, P, Na, Nsc, 2)"""
        B = tokens.size(0)

        out = self.mlp(tokens)   # (B, P*S, ph*pw*2)

        # Reshape to spatial grid: (B, P, grid_h, grid_w, ph, pw, 2)
        out = out.view(B, P, self.grid_h, self.grid_w, self.patch_h, self.patch_w, 2)

        # Interleave patch and grid dims: (B, P, grid_h, ph, grid_w, pw, 2)
        out = out.permute(0, 1, 2, 4, 3, 5, 6).contiguous()

        # Merge: Na = grid_h * ph,  Nsc = grid_w * pw
        out = out.view(B, P, Na, Nsc, 2)
        return out


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class NOVATemporalPredictor(nn.Module):
    """
    NOVA-Temporal Channel Predictor.

    Extends NOVA by injecting P learnable [MASK] future-frame tokens alongside
    K history tokens before the transformer backbone. All (K+P)*S tokens are
    processed together through ``depth`` NOVABlocks so future tokens attend to
    history at every layer (not just in a single prediction head).

    This iterative multi-layer refinement is the key mechanism that makes
    LWM_Temporal outperform NOVA.  NOVA-Temporal adopts the same philosophy
    while keeping NOVA's stronger multi-scale temporal conv (k=3,7) and finer
    32-patch spatial grid (patch_h=4, patch_w=8).

    Input:  (B, K, Na, Nsc, 2)
    Output: (B, P, Na, Nsc, 2)
    """

    def __init__(
        self,
        num_bs_antennas: int = 16,
        num_subcarriers: int = 64,
        embed_dim: int = 256,
        depth: int = 8,           # deeper than NOVA (6) for more refinement rounds
        num_heads: int = 4,
        history_len: int = 16,
        prediction_horizon: int = 4,
        patch_h: int = 4,
        patch_w: int = 8,
        k_small: int = 3,
        k_large: int = 7,
        mlp_ratio: float = 4.0,
        decoder_hidden: int = 512,
        dropout: float = 0.1,
        use_temporal_attention: bool = True,
    ):
        super().__init__()
        self.num_bs_antennas  = num_bs_antennas
        self.num_subcarriers  = num_subcarriers
        self.embed_dim        = embed_dim
        self.depth            = depth
        self.history_len      = history_len
        self.prediction_horizon = prediction_horizon

        K_total = history_len + prediction_horizon   # 20 for default K=16, P=4

        # ── Patch embedding ────────────────────────────────────────────────
        self.patch_embed = PatchEmbed2D(
            num_antennas=num_bs_antennas,
            num_subcarriers=num_subcarriers,
            patch_h=patch_h,
            patch_w=patch_w,
            embed_dim=embed_dim,
        )
        S      = self.patch_embed.num_patches   # 32  (4 ant-patches × 8 sc-patches)
        grid_h = self.patch_embed.grid_h        # 4
        grid_w = self.patch_embed.grid_w        # 8

        # ── Positional embeddings (cover K+P temporal positions) ───────────
        self.temporal_pos = nn.Parameter(torch.zeros(1, K_total, 1, embed_dim))
        self.spatial_pos  = nn.Parameter(torch.zeros(1, 1,       S, embed_dim))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos,  std=0.02)

        # ── Learnable [MASK] future token ──────────────────────────────────
        # One shared embedding broadcast over all P steps and all S patches.
        # Temporal differentiation is provided by temporal_pos[K:K+P].
        # (Analogous to BERT's single [MASK] token — the model learns to ignore
        #  the content and use positional cues + history attention instead.)
        self.future_token = nn.Parameter(torch.zeros(1, 1, 1, embed_dim))
        nn.init.trunc_normal_(self.future_token, std=0.02)

        # ── Backbone ────────────────────────────────────────────────────────
        # Reuse NOVABlock as-is; forward(x, K_total, S) works for any K_total.
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

        # ── Per-patch frame decoder ─────────────────────────────────────────
        self.frame_decoder = FrameDecoder(
            embed_dim=embed_dim,
            patch_h=patch_h,
            patch_w=patch_w,
            grid_h=grid_h,
            grid_w=grid_w,
            hidden_dim=decoder_hidden,
            dropout=dropout,
        )

        # Cached shapes
        self._S       = S
        self._K_total = K_total

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
            K history channel frames (real/imag).

        Returns
        -------
        pred : (B, P, Na, Nsc, 2)
            Direct P-step prediction (non-autoregressive).
        """
        B, K, Na, Nsc, _ = channel_history.shape
        S = self._S
        P = self.prediction_horizon
        K_total = K + P

        # ── 1. Embed K history frames ───────────────────────────────────
        h = channel_history.reshape(B * K, Na, Nsc, 2)
        h_tokens = self.patch_embed(h)               # (B*K, S, D)
        h_tokens = h_tokens.view(B, K, S, self.embed_dim)

        # ── 2. Expand future [MASK] tokens ─────────────────────────────
        # Same learned embedding for all P future steps and S spatial patches.
        # Broadcast is memory-efficient and gradients accumulate back to
        # self.future_token via the sum over all B*P*S uses.
        f_tokens = self.future_token.expand(B, P, S, self.embed_dim)

        # ── 3. Concatenate history + future ────────────────────────────
        tokens = torch.cat([h_tokens, f_tokens], dim=1)   # (B, K+P, S, D)

        # ── 4. Add factorised positional embeddings ─────────────────────
        # temporal_pos[:, :K_total] covers positions 0 … K+P-1
        # positions K … K+P-1 differentiate the P future prediction steps
        tokens = tokens + self.temporal_pos[:, :K_total] + self.spatial_pos

        # ── 5. Flatten to token sequence ───────────────────────────────
        tokens = tokens.reshape(B, K_total * S, self.embed_dim)

        # ── 6. NOVA backbone ────────────────────────────────────────────
        # Future tokens attend to history at EVERY block via:
        #   (a) multi-scale temporal conv: mixes adjacent time steps including history
        #   (b) bidirectional temporal attention: global (K+P) context per spatial pos
        #   (c) spatial attention: cross-patch within each time frame
        #   (d) SwiGLU FFN: pointwise nonlinear refinement
        for block in self.blocks:
            tokens = block(tokens, K_total, S)

        tokens = self.final_norm(tokens)

        # ── 7. Extract future frame tokens (last P*S positions) ─────────
        future_tokens = tokens[:, K * S:, :]    # (B, P*S, D)

        # ── 8. Per-patch decode → channel predictions ───────────────────
        return self.frame_decoder(future_tokens, P, Na, Nsc)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "nova_temporal_channel",
            "num_bs_antennas":    self.num_bs_antennas,
            "num_subcarriers":    self.num_subcarriers,
            "embed_dim":          self.embed_dim,
            "depth":              self.depth,
            "history_len":        self.history_len,
            "prediction_horizon": self.prediction_horizon,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NOVATemporalPredictor":
        config = {k: v for k, v in config.items() if k != "model_type"}
        return cls(**config)
