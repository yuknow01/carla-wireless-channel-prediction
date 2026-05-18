"""
models/lwm_multimodal.py
========================
LWM 멀티모달 채널 예측기 — Late Fusion + Projection Adapter 방식.

Architecture
------------
channel_history (B, K, Na, Nsc, 2)
    │
    ▼  per-SC Transformer d=64  (B×Nsc, K, Na×2=32)  →  last hidden
    │
(B, Nsc, d_model=64)
    │
    ▼  Projection Adapter: Linear(64→256) + LayerNorm   [~17k params]
    │
(B, Nsc, D=256)  ← SC 토큰 (LWM d=64 → 공통 embed_dim=256)
    │
    ├─ ImageTokenEncoder over T image frames → (B, T*49, D=256)  ┐
    └─ PointNetEncoder                      → (B, 16, D=256)    ┘ concat
    │
    ▼  GatedCrossModalFusion × fusion_layers
    │
(B, Nsc, D=256)
    │
    ▼  MLP head
    │
(B, P, Na, Nsc, 2)

Projection Adapter 필요 이유
-----------------------------
LWM 내부 d_model=64, 센서 인코더 embed_dim=256.
차원 불일치 해소를 위해 Linear(64→256) 필수.
channel_only 모드에서도 동일한 head를 사용하기 위해 adapter를 항상 적용한다.

Modes
-----
"multimodal"   : channel + sensor(s)
"channel_only" : channel history only
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sensor_encoders import (
    ImageTokenEncoder,
    PointNetEncoder,
    GatedCrossModalFusion,
)


# ---------------------------------------------------------------------------
# Per-SC LWM Transformer (d_model=64, notebook 설정과 동일)
# ---------------------------------------------------------------------------

class _LayerNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + self.eps) + self.b


class _Embedding(nn.Module):
    def __init__(self, feat_dim: int, d_model: int, max_len: int = 64):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = _LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return self.norm(self.proj(x.float()) + self.pos_embed(pos))


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        B = Q.size(0)
        q = self.wq(Q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.wk(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.wv(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return Q + self.drop(self.out(ctx))


class _FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = _LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.drop(self.fc2(self.drop(F.relu(self.fc1(x))))))


class _EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = _MultiHeadAttention(d_model, n_heads, dropout)
        self.norm = _LayerNorm(d_model)
        self.ffn = _FFN(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.norm(self.attn(x, x, x)))


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class LWMMultiModalPredictor(nn.Module):
    """LWM-based multimodal channel predictor."""

    VALID_MODES = ("multimodal", "channel_only")

    def __init__(
        self,
        mode: str = "multimodal",
        # Channel backbone
        num_bs_antennas: int = 16,
        num_subcarriers: int = 64,
        history_len: int = 16,
        prediction_horizon: int = 4,
        d_model: int = 64,
        n_layers: int = 12,
        n_heads: int = 8,
        d_ff: int = 256,
        lwm_dropout: float = 0.1,
        # Projection adapter
        embed_dim: int = 256,
        # Sensor encoder
        use_image: bool = True,
        use_lidar: bool = True,
        pretrained_image: bool = True,
        lidar_max_points: int = 64,
        lidar_num_tokens: int = 16,
        # Fusion
        fusion_layers: int = 3,
        fusion_heads: int = 4,
        fusion_dropout: float = 0.1,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.mode = mode
        self.num_bs_antennas = num_bs_antennas
        self.num_subcarriers = num_subcarriers
        self.history_len = history_len
        self.prediction_horizon = prediction_horizon
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.use_image = use_image
        self.use_lidar = use_lidar

        feat_dim = num_bs_antennas * 2  # Na*2 = 32 per timestep

        # Per-SC LWM Transformer
        self.embedding = _Embedding(feat_dim, d_model, max_len=max(history_len + 10, 64))
        self.layers = nn.ModuleList([
            _EncoderLayer(d_model, n_heads, d_ff, lwm_dropout)
            for _ in range(n_layers)
        ])

        # Projection Adapter: d_model(64) → embed_dim(256)
        # Applied in both channel_only and multimodal modes for consistent head usage.
        self.proj_adapter = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Sensor encoders (multimodal only)
        _need_sensor = (mode == "multimodal") and (use_image or use_lidar)

        if use_image and _need_sensor:
            self.image_encoder = ImageTokenEncoder(
                embed_dim=embed_dim,
                pretrained=pretrained_image,
            )
            self.no_image_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.no_image_token, std=0.02)
        else:
            self.image_encoder = None
            self.no_image_token = None

        if use_lidar and _need_sensor:
            self.lidar_encoder = PointNetEncoder(
                embed_dim=embed_dim,
                max_points=lidar_max_points,
                num_tokens=lidar_num_tokens,
            )
            self.no_lidar_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.no_lidar_token, std=0.02)
        else:
            self.lidar_encoder = None
            self.no_lidar_token = None

        if _need_sensor:
            self.fusion_blocks = nn.ModuleList([
                GatedCrossModalFusion(
                    embed_dim=embed_dim,
                    num_heads=fusion_heads,
                    dropout=fusion_dropout,
                )
                for _ in range(fusion_layers)
            ])
        else:
            self.fusion_blocks = None

        # Prediction head: (B, Nsc, embed_dim=256) → (B, P, Na, Nsc, 2)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, prediction_horizon * num_bs_antennas * 2),
        )

    def _encode_channel(self, channel_history: torch.Tensor) -> torch.Tensor:
        """Per-SC LWM Transformer encoding.

        channel_history: (B, K, Na, Nsc, 2)
        Returns:         (B, Nsc, d_model=64)
        """
        B, K, Na, Nsc, _ = channel_history.shape
        # (B, K, Na, Nsc, 2) → (B*Nsc, K, Na*2)
        x = channel_history.permute(0, 3, 1, 2, 4).reshape(B * Nsc, K, Na * 2)
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        last = x[:, -1, :]                           # (B*Nsc, d_model)
        return last.reshape(B, Nsc, self.d_model)    # (B, Nsc, d_model)

    def forward(
        self,
        channel_history: torch.Tensor,
        image_seq: Optional[torch.Tensor] = None,
        image_valid_mask: Optional[torch.Tensor] = None,
        lidar_points: Optional[torch.Tensor] = None,
        lidar_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        channel_history : (B, K, Na, Nsc, 2)
        image_seq       : (B, 3, H, W) or (B, T, 3, H, W)  optional
        image_valid_mask: (B,) or (B, T) bool               optional
        lidar_points    : (B, max_points, 4)                 optional
        lidar_mask      : (B, max_points) bool               optional

        Returns
        -------
        (B, P, Na, Nsc, 2)
        """
        B = channel_history.size(0)
        Nsc = self.num_subcarriers
        Na = self.num_bs_antennas
        P = self.prediction_horizon

        # Step 1: channel encoding → (B, Nsc, d_model=64)
        ch_hidden = self._encode_channel(channel_history)

        # Step 2: projection adapter → (B, Nsc, embed_dim=256)
        ch_tokens = self.proj_adapter(ch_hidden)

        # Step 3: channel_only shortcut
        if self.mode == "channel_only":
            out = self.head(ch_tokens)                        # (B, Nsc, P*Na*2)
            out = out.reshape(B, Nsc, P, Na, 2)
            return out.permute(0, 2, 3, 1, 4).contiguous()   # (B, P, Na, Nsc, 2)

        # Step 4: sensor encoding
        sensor_list = []

        if self.use_image and self.image_encoder is not None:
            if image_seq is None:
                tokens_per_frame = getattr(self.image_encoder, "tokens_per_frame", 49)
                img_tokens = self.no_image_token.expand(B, tokens_per_frame, -1)
            else:
                if image_seq.dim() == 5:
                    _, T, C, H, W = image_seq.shape
                    flat_images = image_seq.reshape(B * T, C, H, W)
                    frame_tokens = self.image_encoder(flat_images)  # (B*T, 49, 256)
                    tokens_per_frame = frame_tokens.size(1)
                    real_img_tokens = frame_tokens.view(
                        B, T, tokens_per_frame, self.embed_dim,
                    ).reshape(B, T * tokens_per_frame, self.embed_dim)

                    if image_valid_mask is not None:
                        valid = image_valid_mask.bool()[:, :T].to(real_img_tokens.device)
                    else:
                        valid = torch.ones(B, T, dtype=torch.bool, device=real_img_tokens.device)
                    token_valid = valid.unsqueeze(-1).expand(
                        B, T, tokens_per_frame,
                    ).reshape(B, T * tokens_per_frame)
                    no_img_tokens = self.no_image_token.expand(B, real_img_tokens.size(1), -1)
                    img_tokens = torch.where(
                        token_valid.unsqueeze(-1),
                        real_img_tokens,
                        no_img_tokens,
                    )
                else:
                    if image_valid_mask is not None:
                        has_image = image_valid_mask.bool().reshape(B, -1).any(dim=1)
                    else:
                        has_image = torch.ones(B, dtype=torch.bool, device=image_seq.device)

                    real_img_tokens = self.image_encoder(image_seq)  # (B, 49, 256)
                    no_img_tokens = self.no_image_token.expand(B, real_img_tokens.size(1), -1)
                    img_tokens = torch.where(
                        has_image.to(real_img_tokens.device).view(B, 1, 1),
                        real_img_tokens,
                        no_img_tokens,
                    )
            sensor_list.append(img_tokens)

        if self.use_lidar and self.lidar_encoder is not None:
            if lidar_points is None:
                tokens_per_cloud = getattr(self.lidar_encoder, "num_tokens", 16)
                lid_tokens = self.no_lidar_token.expand(B, tokens_per_cloud, -1)
            else:
                lid_tokens, _ = self.lidar_encoder(lidar_points, lidar_mask)  # (B, 16, 256)
                if lidar_mask is not None:
                    has_lidar = lidar_mask.bool().reshape(B, -1).any(dim=1)
                    no_lid_tokens = self.no_lidar_token.expand(B, lid_tokens.size(1), -1)
                    lid_tokens = torch.where(
                        has_lidar.to(lid_tokens.device).view(B, 1, 1),
                        lid_tokens,
                        no_lid_tokens,
                    )
            sensor_list.append(lid_tokens)

        assert len(sensor_list) > 0, "multimodal mode requires at least one sensor input"
        sensor_tokens = torch.cat(sensor_list, dim=1)         # (B, 49|16|65, 256)

        # Step 5: GatedCrossModalFusion
        fused = ch_tokens
        for block in self.fusion_blocks:
            fused = block(fused, sensor_tokens)               # (B, Nsc, 256)

        # Step 6: head
        out = self.head(fused)                                # (B, Nsc, P*Na*2)
        out = out.reshape(B, Nsc, P, Na, 2)
        return out.permute(0, 2, 3, 1, 4).contiguous()       # (B, P, Na, Nsc, 2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict:
        return {
            "model_type": "lwm_multimodal",
            "mode": self.mode,
            "num_bs_antennas": self.num_bs_antennas,
            "num_subcarriers": self.num_subcarriers,
            "history_len": self.history_len,
            "prediction_horizon": self.prediction_horizon,
            "d_model": self.d_model,
            "embed_dim": self.embed_dim,
            "use_image": self.use_image,
            "use_lidar": self.use_lidar,
        }
