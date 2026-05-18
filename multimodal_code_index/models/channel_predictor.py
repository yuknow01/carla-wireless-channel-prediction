"""
models/channel_predictor.py
===========================
Main multimodal channel prediction model.

Architecture:
    MultimodalChannelPredictor
    +-- ImageEncoder        (ResNet18 -> embed_dim)
    +-- ChannelEncoder      (1D CNN or Transformer -> embed_dim)
    +-- CrossAttentionFusion (image queries, channel keys/values)
    +-- PredictionHead       (MLP -> Na x Nsc x 2)

Supports three modes:
    - "multimodal":   uses both image + channel history
    - "channel_only": uses only channel history (ablation)
    - "image_only":   uses only image input (ablation)
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn

from models.components import (
    ImageEncoder,
    ChannelEncoder,
    CrossAttentionFusion,
    PredictionHead,
)


class MultimodalChannelPredictor(nn.Module):
    """
    Multimodal channel predictor combining visual observations and
    past channel measurements to predict future channel states.

    Parameters
    ----------
    mode : str
        One of "multimodal", "channel_only", "image_only".
    embed_dim : int
        Dimension of all internal embeddings (default: 256).
    num_bs_antennas : int
        Number of base station antennas (default: 16).
    num_subcarriers : int
        Number of OFDM subcarriers (default: 64).
    history_len : int
        Number of past channel measurements K (default: 32).
    channel_encoder_type : str
        "cnn" or "transformer" (default: "transformer").
    channel_encoder_layers : int
        Number of layers in the channel encoder (default: 2).
    channel_encoder_heads : int
        Number of attention heads in the channel encoder (default: 4).
    prediction_head_hidden : int
        Hidden dimension of the prediction MLP (default: 1024).
    prediction_head_layers : int
        Number of layers in the prediction MLP (default: 3).
    pretrained_image : bool
        Whether to use pretrained ImageNet weights for ResNet18 (default: True).
    dropout : float
        Dropout rate (default: 0.1).
    use_time_since_image : bool
        If True, concatenate time_since_image scalar to fusion input (default: True).
    """

    VALID_MODES = ("multimodal", "channel_only", "image_only")

    def __init__(
        self,
        mode: str = "multimodal",
        embed_dim: int = 256,
        num_bs_antennas: int = 16,
        num_subcarriers: int = 64,
        history_len: int = 32,
        channel_encoder_type: str = "transformer",
        channel_encoder_layers: int = 2,
        channel_encoder_heads: int = 4,
        prediction_head_hidden: int = 1024,
        prediction_head_layers: int = 3,
        pretrained_image: bool = True,
        dropout: float = 0.1,
        use_time_since_image: bool = True,
        image_encoder_type: str = "resnet",  # "resnet" or "depth"
    ):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.mode = mode
        self.embed_dim = embed_dim
        self.num_bs_antennas = num_bs_antennas
        self.num_subcarriers = num_subcarriers
        self.history_len = history_len
        self.use_time_since_image = use_time_since_image
        self.image_encoder_type = image_encoder_type

        # ----- Sub-modules -----

        # Image encoder (used in multimodal and image_only modes)
        if mode in ("multimodal", "image_only"):
            if image_encoder_type == "depth":
                from models.depth_encoder import DepthEncoderSimple
                self.image_encoder = DepthEncoderSimple(embed_dim=embed_dim)
            else:
                self.image_encoder = ImageEncoder(
                    embed_dim=embed_dim,
                    pretrained=pretrained_image,
                )
        else:
            self.image_encoder = None

        # Channel encoder (used in multimodal and channel_only modes)
        if mode in ("multimodal", "channel_only"):
            self.channel_encoder = ChannelEncoder(
                num_bs_antennas=num_bs_antennas,
                num_subcarriers=num_subcarriers,
                embed_dim=embed_dim,
                max_history=max(history_len, 64),
                encoder_type=channel_encoder_type,
                num_heads=channel_encoder_heads,
                num_layers=channel_encoder_layers,
                dropout=dropout,
            )
        else:
            self.channel_encoder = None

        # Fusion module (only for multimodal mode)
        if mode == "multimodal":
            self.fusion = CrossAttentionFusion(
                embed_dim=embed_dim,
                num_heads=4,
                dropout=dropout,
            )
            # Optional time-since-image feature
            if use_time_since_image:
                self.time_proj = nn.Sequential(
                    nn.Linear(1, embed_dim),
                    nn.GELU(),
                )
                self.combine = nn.Sequential(
                    nn.Linear(embed_dim * 2, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                )
            else:
                self.time_proj = None
                self.combine = None
        else:
            self.fusion = None
            self.time_proj = None
            self.combine = None

        # Prediction head (always present)
        self.prediction_head = PredictionHead(
            embed_dim=embed_dim,
            num_bs_antennas=num_bs_antennas,
            num_subcarriers=num_subcarriers,
            hidden_dim=prediction_head_hidden,
            num_layers=prediction_head_layers,
            dropout=dropout,
        )

    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        channel_history: Optional[torch.Tensor] = None,
        time_since_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        image : Tensor or None, shape (batch, 3, 224, 224)
            RGB image from the BS camera. Required for multimodal and image_only modes.
        channel_history : Tensor or None, shape (batch, K, Na, Nsc, 2)
            Past K channel measurements in real representation.
            Required for multimodal and channel_only modes.
        time_since_image : Tensor or None, shape (batch, 1)
            Time elapsed since the nearest image was captured (seconds).
            Used only in multimodal mode with use_time_since_image=True.

        Returns
        -------
        Tensor, shape (batch, num_bs_antennas, num_subcarriers, 2)
            Predicted future channel in real representation.
        """
        if self.mode == "multimodal":
            assert image is not None, "multimodal mode requires image input"
            assert channel_history is not None, "multimodal mode requires channel_history input"

            img_feat = self.image_encoder(image)          # (B, embed_dim)
            ch_feat = self.channel_encoder(channel_history)  # (B, embed_dim)
            fused = self.fusion(img_feat, ch_feat)        # (B, embed_dim)

            if self.use_time_since_image and time_since_image is not None:
                t_feat = self.time_proj(time_since_image)  # (B, embed_dim)
                fused = self.combine(torch.cat([fused, t_feat], dim=-1))  # (B, embed_dim)

            return self.prediction_head(fused)

        elif self.mode == "channel_only":
            assert channel_history is not None, "channel_only mode requires channel_history input"

            ch_feat = self.channel_encoder(channel_history)
            return self.prediction_head(ch_feat)

        elif self.mode == "image_only":
            assert image is not None, "image_only mode requires image input"

            img_feat = self.image_encoder(image)
            return self.prediction_head(img_feat)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        """Return a dictionary of configuration for serialization."""
        return {
            "mode": self.mode,
            "embed_dim": self.embed_dim,
            "num_bs_antennas": self.num_bs_antennas,
            "num_subcarriers": self.num_subcarriers,
            "history_len": self.history_len,
            "use_time_since_image": self.use_time_since_image,
            "image_encoder_type": self.image_encoder_type,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultimodalChannelPredictor":
        """Reconstruct model from a config dictionary."""
        return cls(**config)
