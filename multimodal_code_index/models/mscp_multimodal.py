"""
models/mscp_multimodal.py
=========================
MSCP-inspired multimodal channel predictor.

This module is not a bit-exact reimplementation of the MSCP paper. The paper
uses vision sensing to extract physical scene attributes such as user/object
positions, orientations, and materials, then combines those attributes with RF
information for channel prediction. This implementation provides the same
modeling interface inside the CARLA-Wireless codebase:

    RF channel history -> temporal channel tokens
    RGB image sequence -> spatial image tokens
    scene geometry/state -> explicit scene tokens
    channel tokens attend to sensing tokens
    learned future query decodes the predicted channel

The scene branch can consume oracle CARLA metadata first (UE position/velocity,
vehicles_all, or precomputed scene/material features). A later perception
pipeline can replace those oracle features with detector/segmenter outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from models.components import PredictionHead
from models.multi_modal_predictator import (
    ChannelTokenEncoder,
    CrossModalFusionBlock,
    DepthImageTokenEncoder,
    FeedForwardBlock,
    FutureQueryDecoder,
    ResNetImageTokenEncoder,
)


class MSCPSceneEncoder(nn.Module):
    """Encode explicit geometry/semantic scene attributes into tokens."""

    def __init__(
        self,
        embed_dim: int = 256,
        ego_state_dim: int = 6,
        object_feature_dim: int = 4,
        scene_feature_dim: int = 0,
        object_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ego_state_dim = ego_state_dim
        self.object_feature_dim = object_feature_dim
        self.scene_feature_dim = scene_feature_dim

        hidden_dim = object_hidden_dim or embed_dim

        self.ego_encoder = (
            nn.Sequential(
                nn.Linear(ego_state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            if ego_state_dim > 0
            else None
        )
        self.object_encoder = (
            nn.Sequential(
                nn.Linear(object_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            if object_feature_dim > 0
            else None
        )
        self.scene_feature_encoder = (
            nn.Sequential(
                nn.Linear(scene_feature_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.LayerNorm(embed_dim),
            )
            if scene_feature_dim > 0
            else None
        )

        self.ego_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.object_summary_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.object_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.scene_feature_type = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.output_norm = nn.LayerNorm(embed_dim)

        for param in (
            self.ego_type,
            self.object_summary_type,
            self.object_type,
            self.scene_feature_type,
        ):
            nn.init.trunc_normal_(param, std=0.02)

    def _validate_last_dim(self, name: str, tensor: torch.Tensor, expected: int) -> None:
        if tensor.size(-1) != expected:
            raise ValueError(
                f"{name} last dimension must be {expected}, got {tensor.size(-1)}"
            )

    def _encode_ego(self, ego_state: torch.Tensor) -> torch.Tensor:
        if self.ego_encoder is None:
            raise ValueError("ego_state was provided but ego_state_dim is 0")
        self._validate_last_dim("ego_state", ego_state, self.ego_state_dim)
        return self.ego_encoder(ego_state).unsqueeze(1) + self.ego_type

    def _encode_scene_features(self, scene_features: torch.Tensor) -> torch.Tensor:
        if self.scene_feature_encoder is None:
            raise ValueError(
                "scene_features was provided but scene_feature_dim is 0"
            )
        self._validate_last_dim(
            "scene_features",
            scene_features,
            self.scene_feature_dim,
        )
        return self.scene_feature_encoder(scene_features).unsqueeze(1) + self.scene_feature_type

    def _encode_objects(
        self,
        object_features: torch.Tensor,
        object_valid_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.object_encoder is None:
            raise ValueError("object_features was provided but object_feature_dim is 0")
        if object_features.dim() == 2:
            object_features = object_features.unsqueeze(1)
        if object_features.dim() != 3:
            raise ValueError(
                "object_features must have shape (B, N, F) or (B, F), "
                f"got {tuple(object_features.shape)}"
            )
        self._validate_last_dim(
            "object_features",
            object_features,
            self.object_feature_dim,
        )

        batch_size, num_objects, _ = object_features.shape
        if object_valid_mask is None:
            object_valid_mask = torch.isfinite(object_features).all(dim=-1)
        else:
            object_valid_mask = object_valid_mask.to(
                dtype=torch.bool,
                device=object_features.device,
            )
            if object_valid_mask.shape != (batch_size, num_objects):
                raise ValueError(
                    "object_valid_mask must have shape "
                    f"{(batch_size, num_objects)}, got {tuple(object_valid_mask.shape)}"
                )

        safe_features = torch.nan_to_num(object_features, nan=0.0, posinf=0.0, neginf=0.0)
        object_tokens = self.object_encoder(safe_features) + self.object_type

        weights = object_valid_mask.to(dtype=object_tokens.dtype).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        summary = (object_tokens * weights).sum(dim=1, keepdim=True) / denom.unsqueeze(1)
        summary = summary + self.object_summary_type

        tokens = torch.cat([summary, object_tokens], dim=1)
        key_padding_mask = torch.cat(
            [
                torch.zeros(
                    batch_size,
                    1,
                    dtype=torch.bool,
                    device=object_features.device,
                ),
                ~object_valid_mask,
            ],
            dim=1,
        )
        return tokens, key_padding_mask

    def forward(
        self,
        *,
        ego_state: Optional[torch.Tensor] = None,
        object_features: Optional[torch.Tensor] = None,
        object_valid_mask: Optional[torch.Tensor] = None,
        scene_features: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        token_chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []

        if ego_state is not None:
            ego_tokens = self._encode_ego(ego_state)
            token_chunks.append(ego_tokens)
            mask_chunks.append(
                torch.zeros(
                    ego_tokens.size(0),
                    ego_tokens.size(1),
                    dtype=torch.bool,
                    device=ego_tokens.device,
                )
            )

        if object_features is not None:
            object_tokens, object_mask = self._encode_objects(
                object_features,
                object_valid_mask,
            )
            token_chunks.append(object_tokens)
            mask_chunks.append(object_mask)

        if scene_features is not None:
            scene_tokens = self._encode_scene_features(scene_features)
            token_chunks.append(scene_tokens)
            mask_chunks.append(
                torch.zeros(
                    scene_tokens.size(0),
                    scene_tokens.size(1),
                    dtype=torch.bool,
                    device=scene_tokens.device,
                )
            )

        if not token_chunks:
            return None, None

        tokens = self.output_norm(torch.cat(token_chunks, dim=1))
        key_padding_mask = torch.cat(mask_chunks, dim=1)
        return tokens, key_padding_mask


class MSCPMultiModalPredictor(nn.Module):
    """
    MSCP-inspired predictor with explicit scene-token support.

    Supported modes:
    - "multimodal": channel + image and/or scene tokens
    - "rf_scene": channel + scene tokens
    - "channel_only": channel history only
    - "image_only": image sequence only
    - "scene_only": explicit scene tokens only
    """

    VALID_MODES = ("multimodal", "rf_scene", "channel_only", "image_only", "scene_only")

    def __init__(
        self,
        mode: str = "multimodal",
        embed_dim: int = 256,
        num_bs_antennas: int = 16,
        num_subcarriers: int = 64,
        history_len: int = 32,
        max_image_frames: int = 1,
        image_temporal_encoder_layers: int = 1,
        channel_encoder_type: str = "transformer",
        channel_encoder_layers: int = 3,
        channel_encoder_heads: int = 4,
        fusion_layers: int = 2,
        fusion_heads: int = 4,
        decoder_heads: int = 4,
        prediction_head_hidden: int = 1024,
        prediction_head_layers: int = 3,
        pretrained_image: bool = True,
        dropout: float = 0.1,
        use_time_since_image: bool = True,
        image_encoder_type: str = "resnet",
        use_scene: bool = True,
        ego_state_dim: int = 6,
        object_feature_dim: int = 4,
        scene_feature_dim: int = 0,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")

        self.mode = mode
        self.embed_dim = embed_dim
        self.num_bs_antennas = num_bs_antennas
        self.num_subcarriers = num_subcarriers
        self.history_len = history_len
        self.max_image_frames = max_image_frames
        self.image_temporal_encoder_layers = image_temporal_encoder_layers
        self.channel_encoder_type = channel_encoder_type
        self.channel_encoder_layers = channel_encoder_layers
        self.channel_encoder_heads = channel_encoder_heads
        self.fusion_layers_count = fusion_layers
        self.fusion_heads = fusion_heads
        self.decoder_heads = decoder_heads
        self.prediction_head_hidden = prediction_head_hidden
        self.prediction_head_layers = prediction_head_layers
        self.pretrained_image = pretrained_image
        self.dropout = dropout
        self.use_time_since_image = use_time_since_image
        self.image_encoder_type = image_encoder_type
        self.use_scene = use_scene
        self.ego_state_dim = ego_state_dim
        self.object_feature_dim = object_feature_dim
        self.scene_feature_dim = scene_feature_dim

        channel_input_dim = num_bs_antennas * num_subcarriers * 2
        if mode in ("multimodal", "rf_scene", "channel_only"):
            self.channel_encoder = ChannelTokenEncoder(
                input_dim=channel_input_dim,
                embed_dim=embed_dim,
                max_history=max(history_len, 64),
                encoder_type=channel_encoder_type,
                num_heads=channel_encoder_heads,
                num_layers=channel_encoder_layers,
                dropout=dropout,
            )
        else:
            self.channel_encoder = None

        if mode in ("multimodal", "image_only"):
            if image_encoder_type == "resnet":
                self.image_encoder = ResNetImageTokenEncoder(
                    embed_dim=embed_dim,
                    pretrained=pretrained_image,
                )
            elif image_encoder_type == "depth":
                self.image_encoder = DepthImageTokenEncoder(embed_dim=embed_dim)
            else:
                raise ValueError(
                    f"Unknown image_encoder_type: {image_encoder_type}. "
                    "Use 'resnet' or 'depth'."
                )
            self.image_tokens_per_frame = self.image_encoder.num_tokens
            self.image_frame_pos_embed = nn.Parameter(
                torch.zeros(1, max_image_frames, embed_dim)
            )
            self.image_frame_norm = nn.LayerNorm(embed_dim)
            self.image_token_norm = nn.LayerNorm(embed_dim)
            if image_temporal_encoder_layers > 0:
                image_temporal_layer = nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=fusion_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.image_temporal_encoder = nn.TransformerEncoder(
                    image_temporal_layer,
                    num_layers=image_temporal_encoder_layers,
                )
            else:
                self.image_temporal_encoder = None
            nn.init.trunc_normal_(self.image_frame_pos_embed, std=0.02)
        else:
            self.image_encoder = None
            self.image_tokens_per_frame = 0
            self.image_frame_pos_embed = None
            self.image_frame_norm = None
            self.image_token_norm = None
            self.image_temporal_encoder = None

        if use_scene and mode in ("multimodal", "rf_scene", "scene_only"):
            self.scene_encoder = MSCPSceneEncoder(
                embed_dim=embed_dim,
                ego_state_dim=ego_state_dim,
                object_feature_dim=object_feature_dim,
                scene_feature_dim=scene_feature_dim,
                dropout=dropout,
            )
        else:
            self.scene_encoder = None

        if mode in ("multimodal", "rf_scene"):
            self.fusion = nn.ModuleList(
                [
                    CrossModalFusionBlock(
                        embed_dim=embed_dim,
                        num_heads=fusion_heads,
                        dropout=dropout,
                    )
                    for _ in range(fusion_layers)
                ]
            )
        else:
            self.fusion = None

        if use_time_since_image:
            self.time_proj = nn.Sequential(
                nn.Linear(1, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
            )
        else:
            self.time_proj = None

        self.decoder = FutureQueryDecoder(
            embed_dim=embed_dim,
            num_heads=decoder_heads,
            dropout=dropout,
        )
        self.prediction_head = PredictionHead(
            embed_dim=embed_dim,
            num_bs_antennas=num_bs_antennas,
            num_subcarriers=num_subcarriers,
            hidden_dim=prediction_head_hidden,
            num_layers=prediction_head_layers,
            dropout=dropout,
        )

    def _sanitize_key_padding_mask(
        self,
        key_padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if key_padding_mask is None:
            return None
        sanitized = key_padding_mask.bool().clone()
        fully_masked = sanitized.all(dim=1)
        if fully_masked.any():
            sanitized[fully_masked, 0] = False
        return sanitized

    def _build_decoder_condition(
        self,
        image_time_offsets: Optional[torch.Tensor],
        image_valid_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.use_time_since_image or self.time_proj is None or image_time_offsets is None:
            return None

        offsets = image_time_offsets.squeeze(-1)
        if offsets.dim() == 1:
            offsets = offsets.unsqueeze(1)

        if image_valid_mask is not None:
            valid_mask = image_valid_mask.bool()
            safe_offsets = offsets.masked_fill(~valid_mask, float("inf"))
            min_offsets = safe_offsets.min(dim=1).values
            min_offsets = torch.where(
                torch.isfinite(min_offsets),
                min_offsets,
                torch.zeros_like(min_offsets),
            )
        else:
            min_offsets = offsets.min(dim=1).values

        return self.time_proj(min_offsets.unsqueeze(-1))

    def _encode_channel_history(self, channel_history: torch.Tensor) -> torch.Tensor:
        if self.channel_encoder is None:
            raise ValueError(f"mode {self.mode!r} does not use channel_history")
        if channel_history.dim() == 5:
            batch_size, hist_len, num_ant, num_sc, complex_dim = channel_history.shape
            expected = (self.num_bs_antennas, self.num_subcarriers, 2)
            actual = (num_ant, num_sc, complex_dim)
            if actual != expected:
                raise ValueError(
                    f"channel_history trailing shape must be {expected}, got {actual}"
                )
            channel_history = channel_history.reshape(batch_size, hist_len, -1)
        elif channel_history.dim() != 3:
            raise ValueError(
                "channel_history must have shape (B, K, Na, Nsc, 2) or "
                f"(B, K, Na*Nsc*2), got {tuple(channel_history.shape)}"
            )
        return self.channel_encoder(channel_history)

    def _encode_image_sequence(
        self,
        image_seq: torch.Tensor,
        image_time_offsets: Optional[torch.Tensor] = None,
        image_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.image_encoder is None:
            raise ValueError(f"mode {self.mode!r} does not use image input")
        if image_seq.dim() == 4:
            image_seq = image_seq.unsqueeze(1)
        if image_seq.dim() != 5:
            raise ValueError(
                "image_seq must have shape (B, T, C, H, W) or (B, C, H, W), "
                f"got {tuple(image_seq.shape)}"
            )

        batch_size, num_frames, channels, height, width = image_seq.shape
        if num_frames > self.max_image_frames:
            raise ValueError(
                f"image sequence length {num_frames} exceeds "
                f"max_image_frames={self.max_image_frames}"
            )

        if image_valid_mask is None:
            image_valid_mask = torch.ones(
                batch_size,
                num_frames,
                dtype=torch.bool,
                device=image_seq.device,
            )
        else:
            image_valid_mask = image_valid_mask.to(dtype=torch.bool, device=image_seq.device)

        flat_images = image_seq.reshape(batch_size * num_frames, channels, height, width)
        frame_tokens = self.image_encoder(flat_images)
        frame_tokens = frame_tokens.reshape(
            batch_size,
            num_frames,
            self.image_tokens_per_frame,
            self.embed_dim,
        )

        frame_context = self.image_frame_pos_embed[:, :num_frames].expand(
            batch_size,
            -1,
            -1,
        )
        if image_time_offsets is not None and self.use_time_since_image and self.time_proj is not None:
            flat_offsets = image_time_offsets.reshape(batch_size * num_frames, 1)
            time_context = self.time_proj(flat_offsets).view(
                batch_size,
                num_frames,
                self.embed_dim,
            )
            frame_context = frame_context + time_context
        frame_context = self.image_frame_norm(frame_context)

        frame_tokens = frame_tokens + frame_context.unsqueeze(2)
        frame_summary = frame_tokens.mean(dim=2)

        frame_key_padding_mask = self._sanitize_key_padding_mask(~image_valid_mask)
        if self.image_temporal_encoder is not None:
            frame_summary = self.image_temporal_encoder(
                frame_summary,
                src_key_padding_mask=frame_key_padding_mask,
            )

        frame_tokens = frame_tokens + frame_summary.unsqueeze(2)
        image_tokens = frame_tokens.reshape(
            batch_size,
            num_frames * self.image_tokens_per_frame,
            self.embed_dim,
        )
        image_tokens = self.image_token_norm(image_tokens)

        token_key_padding_mask = (~image_valid_mask).unsqueeze(-1).expand(
            batch_size,
            num_frames,
            self.image_tokens_per_frame,
        ).reshape(batch_size, num_frames * self.image_tokens_per_frame)
        token_key_padding_mask = self._sanitize_key_padding_mask(token_key_padding_mask)

        decoder_condition = self._build_decoder_condition(
            image_time_offsets,
            image_valid_mask,
        )
        return image_tokens, token_key_padding_mask, decoder_condition

    def _make_ego_state(
        self,
        ego_state: Optional[torch.Tensor],
        ue_position: Optional[torch.Tensor],
        ue_velocity: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if ego_state is not None:
            return ego_state
        if ue_position is None and ue_velocity is None:
            return None
        if self.ego_state_dim != 6:
            raise ValueError(
                "ue_position/ue_velocity shorthand requires ego_state_dim=6; "
                "pass ego_state directly for other dimensions"
            )

        base = ue_position if ue_position is not None else ue_velocity
        assert base is not None
        batch_size = base.size(0)
        device = base.device
        dtype = base.dtype
        if ue_position is None:
            ue_position = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        if ue_velocity is None:
            ue_velocity = torch.zeros(batch_size, 3, device=device, dtype=dtype)
        return torch.cat([ue_position, ue_velocity], dim=-1)

    def _encode_scene(
        self,
        *,
        ego_state: Optional[torch.Tensor],
        ue_position: Optional[torch.Tensor],
        ue_velocity: Optional[torch.Tensor],
        object_features: Optional[torch.Tensor],
        vehicles_all: Optional[torch.Tensor],
        object_valid_mask: Optional[torch.Tensor],
        scene_features: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.scene_encoder is None:
            if any(
                value is not None
                for value in (
                    ego_state,
                    ue_position,
                    ue_velocity,
                    object_features,
                    vehicles_all,
                    object_valid_mask,
                    scene_features,
                )
            ):
                raise ValueError(f"mode {self.mode!r} does not use scene input")
            return None, None

        ego = self._make_ego_state(ego_state, ue_position, ue_velocity)
        objects = object_features if object_features is not None else vehicles_all
        scene_tokens, scene_mask = self.scene_encoder(
            ego_state=ego,
            object_features=objects,
            object_valid_mask=object_valid_mask,
            scene_features=scene_features,
        )
        scene_mask = self._sanitize_key_padding_mask(scene_mask)
        return scene_tokens, scene_mask

    def _concat_tokens_and_masks(
        self,
        chunks: List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        token_chunks: List[torch.Tensor] = []
        mask_chunks: List[torch.Tensor] = []
        for tokens, mask in chunks:
            if tokens is None:
                continue
            token_chunks.append(tokens)
            if mask is None:
                mask = torch.zeros(
                    tokens.size(0),
                    tokens.size(1),
                    dtype=torch.bool,
                    device=tokens.device,
                )
            mask_chunks.append(mask)

        if not token_chunks:
            return None, None
        tokens = torch.cat(token_chunks, dim=1)
        mask = torch.cat(mask_chunks, dim=1)
        return tokens, self._sanitize_key_padding_mask(mask)

    def forward(
        self,
        channel_history: Optional[torch.Tensor] = None,
        image_seq: Optional[torch.Tensor] = None,
        image_time_offsets: Optional[torch.Tensor] = None,
        image_valid_mask: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        time_since_image: Optional[torch.Tensor] = None,
        ego_state: Optional[torch.Tensor] = None,
        ue_position: Optional[torch.Tensor] = None,
        ue_velocity: Optional[torch.Tensor] = None,
        object_features: Optional[torch.Tensor] = None,
        vehicles_all: Optional[torch.Tensor] = None,
        object_valid_mask: Optional[torch.Tensor] = None,
        scene_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image_seq is None and image is not None:
            image_seq = image.unsqueeze(1)
        if image_time_offsets is None and time_since_image is not None:
            if time_since_image.dim() == 1:
                time_since_image = time_since_image.unsqueeze(-1)
            image_time_offsets = time_since_image.unsqueeze(1)

        channel_tokens = None
        if channel_history is not None:
            channel_tokens = self._encode_channel_history(channel_history)

        image_tokens = None
        image_key_padding_mask = None
        decoder_condition = None
        if image_seq is not None:
            image_tokens, image_key_padding_mask, decoder_condition = (
                self._encode_image_sequence(
                    image_seq=image_seq,
                    image_time_offsets=image_time_offsets,
                    image_valid_mask=image_valid_mask,
                )
            )

        scene_tokens, scene_key_padding_mask = self._encode_scene(
            ego_state=ego_state,
            ue_position=ue_position,
            ue_velocity=ue_velocity,
            object_features=object_features,
            vehicles_all=vehicles_all,
            object_valid_mask=object_valid_mask,
            scene_features=scene_features,
        )

        sensing_tokens, sensing_key_padding_mask = self._concat_tokens_and_masks(
            [
                (image_tokens, image_key_padding_mask),
                (scene_tokens, scene_key_padding_mask),
            ]
        )

        if self.mode in ("multimodal", "rf_scene"):
            if channel_tokens is None:
                raise ValueError(f"{self.mode} mode requires channel_history")
            if self.mode == "rf_scene" and scene_tokens is None:
                raise ValueError("rf_scene mode requires scene input")

            if sensing_tokens is None:
                latent = self.decoder(channel_tokens)
                return self.prediction_head(latent)

            fused_tokens = channel_tokens
            assert self.fusion is not None
            for fusion_block in self.fusion:
                fused_tokens = fusion_block(
                    fused_tokens,
                    sensing_tokens,
                    image_key_padding_mask=sensing_key_padding_mask,
                )

            context_tokens = torch.cat([fused_tokens, sensing_tokens], dim=1)
            channel_mask = torch.zeros(
                fused_tokens.size(0),
                fused_tokens.size(1),
                dtype=torch.bool,
                device=fused_tokens.device,
            )
            context_key_padding_mask = torch.cat(
                [channel_mask, sensing_key_padding_mask],
                dim=1,
            )
            context_key_padding_mask = self._sanitize_key_padding_mask(
                context_key_padding_mask
            )
            latent = self.decoder(
                context_tokens,
                condition=decoder_condition,
                context_key_padding_mask=context_key_padding_mask,
            )
            return self.prediction_head(latent)

        if self.mode == "channel_only":
            if channel_tokens is None:
                raise ValueError("channel_only mode requires channel_history")
            latent = self.decoder(channel_tokens)
            return self.prediction_head(latent)

        if self.mode == "image_only":
            if image_tokens is None:
                raise ValueError("image_only mode requires image input")
            latent = self.decoder(
                image_tokens,
                condition=decoder_condition,
                context_key_padding_mask=image_key_padding_mask,
            )
            return self.prediction_head(latent)

        if self.mode == "scene_only":
            if scene_tokens is None:
                raise ValueError("scene_only mode requires scene input")
            latent = self.decoder(
                scene_tokens,
                context_key_padding_mask=scene_key_padding_mask,
            )
            return self.prediction_head(latent)

        raise RuntimeError(f"Unhandled mode: {self.mode}")

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "embed_dim": self.embed_dim,
            "num_bs_antennas": self.num_bs_antennas,
            "num_subcarriers": self.num_subcarriers,
            "history_len": self.history_len,
            "max_image_frames": self.max_image_frames,
            "image_temporal_encoder_layers": self.image_temporal_encoder_layers,
            "channel_encoder_type": self.channel_encoder_type,
            "channel_encoder_layers": self.channel_encoder_layers,
            "channel_encoder_heads": self.channel_encoder_heads,
            "fusion_layers": self.fusion_layers_count,
            "fusion_heads": self.fusion_heads,
            "decoder_heads": self.decoder_heads,
            "prediction_head_hidden": self.prediction_head_hidden,
            "prediction_head_layers": self.prediction_head_layers,
            "pretrained_image": self.pretrained_image,
            "dropout": self.dropout,
            "use_time_since_image": self.use_time_since_image,
            "image_encoder_type": self.image_encoder_type,
            "use_scene": self.use_scene,
            "ego_state_dim": self.ego_state_dim,
            "object_feature_dim": self.object_feature_dim,
            "scene_feature_dim": self.scene_feature_dim,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MSCPMultiModalPredictor":
        return cls(**config)


__all__ = ["MSCPSceneEncoder", "MSCPMultiModalPredictor"]
