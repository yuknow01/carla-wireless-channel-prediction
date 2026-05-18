"""
models/multi_modal_predictator.py
=================================
Token-level multimodal architecture for channel prediction.

This model keeps the same external interface as the existing
``MultimodalChannelPredictor`` while using a richer internal design:

    image -> spatial image tokens
    channel history -> temporal channel tokens
    channel tokens attend to image tokens
    learned future query token reads fused context
    prediction head reconstructs future channel

Supported modes:
    - "multimodal"
    - "channel_only"
    - "image_only"
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision.models as tv_models

from models.components import PredictionHead


class FeedForwardBlock(nn.Module):
    """Transformer-style feed-forward block."""

    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalConvBlock(nn.Module):
    """Residual temporal convolution block for channel token encoding."""

    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x).transpose(1, 2)
        y = self.conv(y).transpose(1, 2)
        return residual + y


class ChannelTokenEncoder(nn.Module):
    """
    Encode channel history into temporal tokens.

    Input:  (B, K, Na * Nsc * 2)
    Output: (B, K, D)
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        max_history: int = 64,
        encoder_type: str = "transformer",
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.max_history = max_history

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, max_history, embed_dim))
        self.output_norm = nn.LayerNorm(embed_dim)

        if encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        elif encoder_type == "cnn":
            self.encoder = nn.ModuleList(
                [TemporalConvBlock(embed_dim=embed_dim, dropout=dropout) for _ in range(num_layers)]
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.max_history:
            raise ValueError(
                f"channel history length {x.size(1)} exceeds max_history={self.max_history}"
            )

        tokens = self.input_proj(x)
        tokens = tokens + self.pos_embed[:, : x.size(1)]

        if self.encoder_type == "transformer":
            tokens = self.encoder(tokens)
        else:
            for block in self.encoder:
                tokens = block(tokens)

        return self.output_norm(tokens)


class ResNetImageTokenEncoder(nn.Module):
    """
    Convert an RGB image into spatial tokens using a ResNet18 feature map.

    Output: (B, 49, D)
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True, grid_size: int = 7):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        self.num_tokens = grid_size * grid_size

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.stages = nn.Sequential(
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.stages(feat)
        feat = self.pool(feat)
        feat = self.proj(feat)
        tokens = feat.flatten(2).transpose(1, 2)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        return self.norm(tokens)


class DepthImageTokenEncoder(nn.Module):
    """Fallback depth-based image encoder that emits a single token."""

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        from models.depth_encoder import DepthEncoderSimple

        self.encoder = DepthEncoderSimple(embed_dim=embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.num_tokens = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token = self.norm(self.encoder(x))
        return token.unsqueeze(1)


class CrossModalFusionBlock(nn.Module):
    """Channel tokens read image tokens through cross-attention."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardBlock(embed_dim=embed_dim, dropout=dropout)

    def forward(
        self,
        channel_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        image_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(
            self.query_norm(channel_tokens),
            self.context_norm(image_tokens),
            self.context_norm(image_tokens),
            key_padding_mask=image_key_padding_mask,
        )
        channel_tokens = channel_tokens + self.dropout(attn_out)
        channel_tokens = channel_tokens + self.ffn(self.ffn_norm(channel_tokens))
        return channel_tokens


class FutureQueryDecoder(nn.Module):
    """Decode a single future-state representation from context tokens."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = FeedForwardBlock(embed_dim=embed_dim, dropout=dropout)

        nn.init.trunc_normal_(self.query_token, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        context_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self.query_token.expand(context_tokens.size(0), -1, -1)
        if condition is not None:
            query = query + condition.unsqueeze(1)

        attn_out, _ = self.attn(
            self.query_norm(query),
            self.context_norm(context_tokens),
            self.context_norm(context_tokens),
            key_padding_mask=context_key_padding_mask,
        )
        query = query + attn_out
        query = query + self.ffn(self.ffn_norm(query))
        return query.squeeze(1)


class MultiModalPredictator(nn.Module):
    """
    Token-level multimodal architecture for future channel prediction.

    The public interface intentionally mirrors the existing
    ``MultimodalChannelPredictor`` so it can be integrated incrementally.
    """

    VALID_MODES = ("multimodal", "channel_only", "image_only")

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
    ):
        super().__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

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

        channel_input_dim = num_bs_antennas * num_subcarriers * 2

        if mode in ("multimodal", "channel_only"):
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
                    f"Unknown image_encoder_type: {image_encoder_type}. Use 'resnet' or 'depth'."
                )
            self.image_tokens_per_frame = self.image_encoder.num_tokens
            self.image_frame_pos_embed = nn.Parameter(torch.zeros(1, max_image_frames, embed_dim))
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

        if mode == "multimodal":
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

    def _encode_image_sequence(
        self,
        image_seq: torch.Tensor,
        image_time_offsets: Optional[torch.Tensor] = None,
        image_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        if image_seq.dim() == 4:
            image_seq = image_seq.unsqueeze(1)
        if image_seq.dim() != 5:
            raise ValueError(
                f"image_seq must have shape (B, T, C, H, W) or (B, C, H, W), got {tuple(image_seq.shape)}"
            )

        batch_size, num_frames, channels, height, width = image_seq.shape
        if num_frames > self.max_image_frames:
            raise ValueError(
                f"image sequence length {num_frames} exceeds max_image_frames={self.max_image_frames}"
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

        frame_context = self.image_frame_pos_embed[:, :num_frames].expand(batch_size, -1, -1)
        if image_time_offsets is not None and self.use_time_since_image and self.time_proj is not None:
            flat_offsets = image_time_offsets.reshape(batch_size * num_frames, 1)
            time_context = self.time_proj(flat_offsets).view(batch_size, num_frames, self.embed_dim)
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

        decoder_condition = self._build_decoder_condition(image_time_offsets, image_valid_mask)
        return image_tokens, token_key_padding_mask, decoder_condition

    def forward(
        self,
        channel_history: Optional[torch.Tensor] = None,
        image_seq: Optional[torch.Tensor] = None,
        image_time_offsets: Optional[torch.Tensor] = None,
        image_valid_mask: Optional[torch.Tensor] = None,
        image: Optional[torch.Tensor] = None,
        time_since_image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if image_seq is None and image is not None:
            image_seq = image.unsqueeze(1)

        if image_time_offsets is None and time_since_image is not None:
            if time_since_image.dim() == 1:
                time_since_image = time_since_image.unsqueeze(-1)
            image_time_offsets = time_since_image.unsqueeze(1)

        image_tokens = None
        image_key_padding_mask = None
        decoder_condition = None
        if image_seq is not None:
            image_tokens, image_key_padding_mask, decoder_condition = self._encode_image_sequence(
                image_seq=image_seq,
                image_time_offsets=image_time_offsets,
                image_valid_mask=image_valid_mask,
            )

        if self.mode == "multimodal":
            assert image_tokens is not None, "multimodal mode requires image input"
            assert channel_history is not None, "multimodal mode requires channel_history input"

            channel_tokens = self.channel_encoder(channel_history)

            fused_tokens = channel_tokens
            for fusion_block in self.fusion:
                fused_tokens = fusion_block(
                    fused_tokens,
                    image_tokens,
                    image_key_padding_mask=image_key_padding_mask,
                )

            context_tokens = torch.cat([fused_tokens, image_tokens], dim=1)
            channel_mask = torch.zeros(
                fused_tokens.size(0),
                fused_tokens.size(1),
                dtype=torch.bool,
                device=fused_tokens.device,
            )
            context_key_padding_mask = (
                torch.cat([channel_mask, image_key_padding_mask], dim=1)
                if image_key_padding_mask is not None
                else None
            )
            latent = self.decoder(
                context_tokens,
                condition=decoder_condition,
                context_key_padding_mask=context_key_padding_mask,
            )
            return self.prediction_head(latent)

        if self.mode == "channel_only":
            assert channel_history is not None, "channel_only mode requires channel_history input"
            channel_tokens = self.channel_encoder(channel_history)
            latent = self.decoder(channel_tokens)
            return self.prediction_head(latent)

        assert image_tokens is not None, "image_only mode requires image input"
        latent = self.decoder(
            image_tokens,
            condition=decoder_condition,
            context_key_padding_mask=image_key_padding_mask,
        )
        return self.prediction_head(latent)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

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
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultiModalPredictator":
        return cls(**config)
