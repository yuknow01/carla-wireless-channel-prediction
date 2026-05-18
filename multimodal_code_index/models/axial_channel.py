"""
Patch-based Factorized Axial Attention Channel Predictor
=========================================================
Input : (B, K, Na, Nsc, 2)  — K history frames, real/imag
Output: (B, P, Na, Nsc, 2)  — P predicted future frames

아키텍처 (v2 — Patch 기반):
  1. Patch Embedding: (Na, Nsc, 2) → S spatial tokens per time step
       patch_a=1, patch_s=4  →  S = (Na/1) × (Nsc/4) = 16 × 16 = 256 tokens
       각 patch: 1 antenna × 4 subcarriers × 2 (real/imag) = 8 features → D

  2. Factorized Axial Encoder on (B, K, S, D):
       Temporal attention : (B*S, K, D)  — effective batch = B*S  (훨씬 작음!)
       Spatial  attention : (B*K, S, D)  — effective batch = B*K
       FFN

  3. Prediction Head (per-patch temporal cross-attention):
       P learnable queries가 K개 시간 토큰에 cross-attend (patch별 독립)
       → patch decoder (D→8) → 패치 조합 → (B, P, Na, Nsc, 2)

메모리 분석 (D=192, depth=6, batch_size=128, 3 GPUs):
  GPU당 B=42  →  Temporal QKV: 42*256×16×576×2bytes = 199 MB
  6 blocks × 2 attentions × 2(for/back) = 4.8 GB  →  24GB GPU에서 OK
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class MHSA(nn.Module):
    """Flash Attention 기반 self-attention (F.scaled_dot_product_attention).
    N×N attention matrix를 명시적으로 저장하지 않아 O(N) 메모리."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = dropout

    def forward(self, x: Tensor) -> Tensor:
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)           # (3, B, H, N, hd)
        q, k, v = qkv.unbind(0)                     # each (B, H, N, hd)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
        )                                            # (B, H, N, hd)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.proj(out)


class PatchEmbed(nn.Module):
    """
    (B*K, Na, Nsc, 2) → (B*K, S, D) via spatial patches.

    patch_a antennas × patch_s subcarriers per token.
    S = (Na // patch_a) × (Nsc // patch_s)
    """

    def __init__(
        self,
        Na: int,
        Nsc: int,
        patch_a: int,
        patch_s: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        assert Na % patch_a == 0 and Nsc % patch_s == 0
        self.patch_a = patch_a
        self.patch_s = patch_s
        self.Pa = Na // patch_a    # # patches along antenna axis
        self.Ps = Nsc // patch_s   # # patches along subcarrier axis
        self.S = self.Pa * self.Ps
        self.proj = nn.Linear(patch_a * patch_s * 2, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        # x: (BK, Na, Nsc, 2)
        BK, Na, Nsc, C = x.shape
        x = x.reshape(BK, self.Pa, self.patch_a, self.Ps, self.patch_s, C)
        # (BK, Pa, Ps, pa, ps, C) → (BK, Pa*Ps, pa*ps*C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(BK, self.S, self.patch_a * self.patch_s * C)
        return self.proj(x)    # (BK, S, D)


class AxialBlock2D(nn.Module):
    """
    Factorized Temporal + Spatial attention on (B, K, S, D).

    Temporal : (B*S, K, D) — effective batch = B*S
    Spatial  : (B*K, S, D) — effective batch = B*K
    FFN
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)

        self.norm_t = nn.LayerNorm(embed_dim)
        self.attn_t = MHSA(embed_dim, num_heads, dropout)

        self.norm_s = nn.LayerNorm(embed_dim)
        self.attn_s = MHSA(embed_dim, num_heads, dropout)

        self.norm_f = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, K, S, D = x.shape

        # ── Temporal attention ───────────────────────────────────────────
        # 각 spatial token이 K개 시간 토큰끼리 attend
        xt = x.permute(0, 2, 1, 3).reshape(B * S, K, D)
        xt = xt + self.attn_t(self.norm_t(xt))
        x = xt.reshape(B, S, K, D).permute(0, 2, 1, 3)    # → (B, K, S, D)

        # ── Spatial attention ────────────────────────────────────────────
        # 각 time step의 S개 spatial token끼리 attend
        xs = x.reshape(B * K, S, D)
        xs = xs + self.attn_s(self.norm_s(xs))
        x = xs.reshape(B, K, S, D)

        # ── FFN ──────────────────────────────────────────────────────────
        x = x + self.ffn(self.norm_f(x))
        return x


class PatchExpandHead(nn.Module):
    """
    Per-patch temporal cross-attention + patch decoder.

    각 spatial patch마다 P learnable query가 K 시간 토큰에 cross-attend,
    이후 patch decoder로 원래 (patch_a, patch_s, 2) 공간을 복원.
    Output: (B, P, Na, Nsc, 2)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        prediction_horizon: int,
        Pa: int,
        Ps: int,
        patch_a: int,
        patch_s: int,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        self.P = prediction_horizon
        self.S = Pa * Ps
        self.Pa = Pa
        self.Ps = Ps
        self.patch_a = patch_a
        self.patch_s = patch_s
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.queries  = nn.Parameter(torch.zeros(1, prediction_horizon, embed_dim))
        self.norm_q   = nn.LayerNorm(embed_dim)
        self.norm_kv  = nn.LayerNorm(embed_dim)
        self.q_proj   = nn.Linear(embed_dim, embed_dim)
        self.kv_proj  = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.to_patch = nn.Linear(embed_dim, patch_a * patch_s * 2)

        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, K, S, D)
        B, K, S, D = x.shape
        BS = B * S

        # Per-patch temporal cross-attention
        kv_in = x.permute(0, 2, 1, 3).reshape(BS, K, D)   # (B*S, K, D)
        q_in  = self.queries.expand(BS, -1, -1)             # (B*S, P, D)

        q  = self.q_proj(self.norm_q(q_in))                 # (B*S, P, D)
        kv = self.kv_proj(self.norm_kv(kv_in))              # (B*S, K, 2D)
        k, v = kv.chunk(2, dim=-1)                          # each (B*S, K, D)

        H, hd = self.num_heads, self.head_dim
        q = q.reshape(BS, self.P, H, hd).permute(0, 2, 1, 3)  # (B*S, H, P, hd)
        k = k.reshape(BS, K, H, hd).permute(0, 2, 1, 3)
        v = v.reshape(BS, K, H, hd).permute(0, 2, 1, 3)

        out = F.scaled_dot_product_attention(q, k, v)          # (B*S, H, P, hd)
        out = out.transpose(1, 2).reshape(BS, self.P, D)

        out = self.out_proj(out)
        out = self.to_patch(out)                               # (B*S, P, pa*ps*2)

        # Unfold patches → (B, P, Na, Nsc, 2)
        pa, ps = self.patch_a, self.patch_s
        out = out.reshape(B, self.Pa, self.Ps, self.P, pa, ps, 2)
        out = out.permute(0, 3, 1, 4, 2, 5, 6).contiguous()
        out = out.reshape(B, self.P, self.Pa * pa, self.Ps * ps, 2)
        return out    # (B, P, Na, Nsc, 2)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class AxialChannelPredictor(nn.Module):
    """
    Patch-based Factorized Axial Attention Channel Predictor.

    Args:
        history_len        : K, 히스토리 프레임 수  (default 16)
        prediction_horizon : P, 예측 프레임 수       (default 4)
        num_antennas       : Na                      (default 16)
        num_subcarriers    : Nsc                     (default 64)
        embed_dim          : D, 임베딩 차원           (default 192)
        depth              : AxialBlock2D 수          (default 6)
        num_heads          : 어텐션 헤드 수            (default 8)
        mlp_ratio          : FFN 확장 비율             (default 4.0)
        dropout            : 드롭아웃 비율              (default 0.0)
        patch_a            : antenna patch size        (default 1)
        patch_s            : subcarrier patch size     (default 4)
    """

    def __init__(
        self,
        history_len: int = 16,
        prediction_horizon: int = 4,
        num_antennas: int = 16,
        num_subcarriers: int = 64,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        patch_a: int = 1,
        patch_s: int = 4,
    ) -> None:
        super().__init__()
        K, P = history_len, prediction_horizon
        Na, Nsc, D = num_antennas, num_subcarriers, embed_dim
        Pa = Na // patch_a
        Ps = Nsc // patch_s
        S  = Pa * Ps

        self.K = K;  self.P = P;  self.Na = Na;  self.Nsc = Nsc
        self.S = S;  self.Pa = Pa;  self.Ps = Ps
        self.patch_a = patch_a;  self.patch_s = patch_s

        # ── Patch embedding ───────────────────────────────────────────────
        self.patch_embed = PatchEmbed(Na, Nsc, patch_a, patch_s, D)

        # ── Positional embeddings ─────────────────────────────────────────
        self.temporal_pos = nn.Parameter(torch.zeros(K, D))
        self.spatial_pos  = nn.Parameter(torch.zeros(S, D))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos,  std=0.02)

        # ── Encoder ───────────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            AxialBlock2D(D, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(D)

        # ── Prediction head ───────────────────────────────────────────────
        self.head = PatchExpandHead(D, num_heads, P, Pa, Ps, patch_a, patch_s)

        self._init_weights()

    def _init_weights(self) -> None:
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
        x: Optional[Tensor] = None,
        *,
        channel_history: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x / channel_history: (B, K, Na, Nsc, 2)
        Returns:
            (B, P, Na, Nsc, 2)
        """
        if channel_history is not None:
            x = channel_history
        assert x is not None

        B, K, Na, Nsc, _ = x.shape

        # Patch embedding: (B, K, Na, Nsc, 2) → (B*K, Na, Nsc, 2) → (B*K, S, D)
        x = x.reshape(B * K, Na, Nsc, 2)
        x = self.patch_embed(x)             # (B*K, S, D)
        x = x.reshape(B, K, self.S, -1)    # (B, K, S, D)

        # Positional encoding
        x = x + self.temporal_pos[None, :, None, :]
        x = x + self.spatial_pos[None, None, :, :]

        # Axial encoder
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # Predict P future frames
        return self.head(x)    # (B, P, Na, Nsc, 2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> dict:
        return {
            "history_len":        self.K,
            "prediction_horizon": self.P,
            "num_antennas":       self.Na,
            "num_subcarriers":    self.Nsc,
            "embed_dim":          self.temporal_pos.shape[1],
            "depth":              len(self.blocks),
            "num_heads":          self.blocks[0].attn_t.num_heads,
            "mlp_ratio":          self.blocks[0].ffn[0].out_features / self.temporal_pos.shape[1],
            "patch_a":            self.patch_a,
            "patch_s":            self.patch_s,
            "spatial_tokens":     self.S,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: AxialChannelPredictor) -> None:
    total = count_parameters(model)
    D = model.temporal_pos.shape[1]
    sections = {
        "patch_embed":    count_parameters(model.patch_embed),
        "pos_embed":      model.temporal_pos.numel() + model.spatial_pos.numel(),
        "encoder_blocks": sum(count_parameters(b) for b in model.blocks),
        "output_norm":    count_parameters(model.norm),
        "pred_head":      count_parameters(model.head),
    }
    print(f"{'Module':<20} {'Params':>12}")
    print("-" * 33)
    for k, v in sections.items():
        print(f"  {k:<18} {v:>12,}")
    print("-" * 33)
    print(f"  {'Total':<18} {total:>12,}  ({total/1e6:.3f}M)")
    print(f"  Spatial tokens S = {model.S}  "
          f"(patch: {model.patch_a}×{model.patch_s}, "
          f"grid: {model.Pa}×{model.Ps})")
