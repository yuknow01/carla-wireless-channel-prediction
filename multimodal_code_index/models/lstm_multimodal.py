"""
models/lstm_multimodal.py
=========================
LSTM 멀티모달 채널 예측기 — Late Fusion 방식.

Architecture
------------
channel_history (B, K, Na, Nsc, 2)
    │
    ▼  per-SC LSTM  (B×Nsc, K, Na×2=32)  →  last hidden
    │
(B, Nsc, lstm_hidden=256)  ← 64개 SC 토큰
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

Modes
-----
"multimodal"   : channel + sensor(s)
"channel_only" : channel history only (sensor encoders not used)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.sensor_encoders import (
    ImageTokenEncoder,
    PointNetEncoder,
    GatedCrossModalFusion,
)


class LSTMMultiModalPredictor(nn.Module):
    """LSTM-based multimodal channel predictor."""

    VALID_MODES = ("multimodal", "channel_only")

    def __init__(
        self,
        mode: str = "multimodal",
        # Channel backbone
        num_bs_antennas: int = 16,
        num_subcarriers: int = 64,
        history_len: int = 16,
        prediction_horizon: int = 4,
        lstm_hidden: int = 256,
        lstm_layers: int = 3,
        lstm_dropout: float = 0.1,
        # Sensor encoder
        embed_dim: int = 256,
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
        self.lstm_hidden = lstm_hidden
        self.embed_dim = embed_dim
        self.use_image = use_image
        self.use_lidar = use_lidar

        # Per-SC LSTM: input = Na*2 per timestep
        feat_dim = num_bs_antennas * 2
        self.lstm = nn.LSTM(
            feat_dim, lstm_hidden, num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
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

        # Prediction head
        # channel_only: (B, Nsc, lstm_hidden=256) → head
        # multimodal:   (B, Nsc, embed_dim=256)   → head
        # Both are 256-dim by default; same head used for both modes.
        head_in = embed_dim  # lstm_hidden == embed_dim == 256 by default
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, head_in),
            nn.GELU(),
            nn.Linear(head_in, prediction_horizon * num_bs_antennas * 2),
        )

    def _encode_channel(self, channel_history: torch.Tensor) -> torch.Tensor:
        """Per-SC LSTM encoding.

        channel_history: (B, K, Na, Nsc, 2)
        Returns:         (B, Nsc, lstm_hidden)
        """
        B, K, Na, Nsc, _ = channel_history.shape
        # (B, K, Na, Nsc, 2) → (B*Nsc, K, Na*2)
        x = channel_history.permute(0, 3, 1, 2, 4).reshape(B * Nsc, K, Na * 2)
        output, _ = self.lstm(x)          # (B*Nsc, K, hidden)
        last = output[:, -1, :]           # (B*Nsc, hidden)
        return last.reshape(B, Nsc, self.lstm_hidden)  # (B, Nsc, hidden)

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

        # Step 1: channel encoding → (B, Nsc, 256)
        ch_tokens = self._encode_channel(channel_history)

        # Step 2: channel_only shortcut
        if self.mode == "channel_only":
            out = self.head(ch_tokens)                        # (B, Nsc, P*Na*2)
            out = out.reshape(B, Nsc, P, Na, 2)
            return out.permute(0, 2, 3, 1, 4).contiguous()   # (B, P, Na, Nsc, 2)

        # Step 3: sensor encoding
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

        # Step 4: GatedCrossModalFusion
        fused = ch_tokens
        for block in self.fusion_blocks:
            fused = block(fused, sensor_tokens)               # (B, Nsc, 256)

        # Step 5: head
        out = self.head(fused)                                # (B, Nsc, P*Na*2)
        out = out.reshape(B, Nsc, P, Na, 2)
        return out.permute(0, 2, 3, 1, 4).contiguous()       # (B, P, Na, Nsc, 2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict:
        return {
            "model_type": "lstm_multimodal",
            "mode": self.mode,
            "num_bs_antennas": self.num_bs_antennas,
            "num_subcarriers": self.num_subcarriers,
            "history_len": self.history_len,
            "prediction_horizon": self.prediction_horizon,
            "lstm_hidden": self.lstm_hidden,
            "embed_dim": self.embed_dim,
            "use_image": self.use_image,
            "use_lidar": self.use_lidar,
        }
