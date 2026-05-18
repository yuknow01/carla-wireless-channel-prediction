"""
models/chiron_multimodal.py
===========================
CHIRON Multi-Modal Predictor -- extends the CHIRON channel backbone
with image, LiDAR, and ego-state fusion for enhanced channel prediction.

Designed for the CARLA-Wireless dataset:
  - 28 GHz, 16-element ULA, 512 subcarriers, 2 kHz sampling
  - RGB camera images (224x224)
  - LiDAR point clouds (N, 4) — x, y, z, intensity
  - UE position / velocity metadata

Sensor modalities are controlled by `use_image` and `use_lidar` flags:
  - use_image=True,  use_lidar=False  → channel + image
  - use_image=False, use_lidar=True   → channel + lidar
  - use_image=True,  use_lidar=True   → channel + image + lidar
  - Both False with mode=channel_only → channel only

Supported modes:
    - "multimodal"   : channel + sensor(s) (image / lidar / both)
    - "channel_only" : channel history only
    - "image_only"   : image only
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.chiron_channel import (
    PatchEmbed2D,
    ChironBlock,
    ChannelPredictionHead,
    GatedFFN,
)
from models.fusion_blocks import GatedCrossModalFusion
from models.image_encoders import ImageTokenEncoder
from models.lidar_encoders import PointNetEncoder


# ---------------------------------------------------------------------------
# Ego State Encoder (position / velocity)
# ---------------------------------------------------------------------------

class EgoStateEncoder(nn.Module):
    """Encode UE ego-state (position, velocity, etc.) into a conditioning vector.

    Default input: [x, y, z, vx, vy, vz] -> (B, D)
    """

    def __init__(self, state_dim: int = 6, embed_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, state_dim) -> (B, D)"""
        return self.net(x)


# ---------------------------------------------------------------------------
# Multi-Image Temporal Encoder
# ---------------------------------------------------------------------------

class ImageSequenceEncoder(nn.Module):
    """Encode a sequence of images into a single set of tokens.

    Handles temporal position encoding and optional valid masks for
    asynchronous image/channel sampling rates.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        max_image_frames: int = 4,
        tokens_per_frame: int = 49,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_image_frames = max_image_frames
        self.tokens_per_frame = tokens_per_frame
        self.embed_dim = embed_dim

        self.frame_pos = nn.Parameter(torch.zeros(1, max_image_frames, embed_dim))
        self.frame_norm = nn.LayerNorm(embed_dim)

        # Temporal self-attention across frame summaries
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.frame_pos, std=0.02)

    def forward(
        self,
        image_encoder: ImageTokenEncoder,
        image_seq: torch.Tensor,
        image_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        image_seq: (B, T, 3, H, W) or (B, 3, H, W) for single image
        image_valid_mask: (B, T) bool mask, True = valid

        Returns:
            image_tokens: (B, T*tokens_per_frame, D)
            token_padding_mask: (B, T*tokens_per_frame) bool, True = masked
        """
        if image_seq.dim() == 4:
            image_seq = image_seq.unsqueeze(1)

        B, T, C, H, W = image_seq.shape

        if image_valid_mask is None:
            image_valid_mask = torch.ones(B, T, dtype=torch.bool, device=image_seq.device)
        else:
            image_valid_mask = image_valid_mask.bool()

        # Encode each frame
        flat = image_seq.reshape(B * T, C, H, W)
        frame_tokens = image_encoder(flat)  # (B*T, tokens_per_frame, D)
        frame_tokens = frame_tokens.view(B, T, self.tokens_per_frame, self.embed_dim)

        # Add temporal position to each frame
        frame_pos = self.frame_norm(self.frame_pos[:, :T])  # (1, T, D)
        frame_tokens = frame_tokens + frame_pos.unsqueeze(2)  # broadcast to tokens

        # Temporal attention across frame summaries
        frame_summary = frame_tokens.mean(dim=2)  # (B, T, D)
        frame_summary_normed = self.temporal_norm(frame_summary)

        # Mask for temporal attention: True = do not attend
        frame_key_mask = ~image_valid_mask
        # Sanitize: ensure at least one position is valid per batch
        fully_masked = frame_key_mask.all(dim=1)
        if fully_masked.any():
            frame_key_mask = frame_key_mask.clone()
            frame_key_mask[fully_masked, 0] = False

        attn_out, _ = self.temporal_attn(
            frame_summary_normed, frame_summary_normed, frame_summary_normed,
            key_padding_mask=frame_key_mask,
        )
        frame_summary = frame_summary + attn_out

        # Inject temporal context back into per-token representations
        frame_tokens = frame_tokens + frame_summary.unsqueeze(2)

        # Flatten all tokens
        image_tokens = frame_tokens.reshape(B, T * self.tokens_per_frame, self.embed_dim)
        image_tokens = self.output_norm(image_tokens)

        # Build token-level padding mask
        token_padding_mask = (~image_valid_mask).unsqueeze(-1).expand(
            B, T, self.tokens_per_frame,
        ).reshape(B, T * self.tokens_per_frame)

        # Sanitize token mask
        fully_masked_tokens = token_padding_mask.all(dim=1)
        if fully_masked_tokens.any():
            token_padding_mask = token_padding_mask.clone()
            token_padding_mask[fully_masked_tokens, 0] = False

        return image_tokens, token_padding_mask


# ---------------------------------------------------------------------------
# Future Query Decoder
# ---------------------------------------------------------------------------

class FutureQueryDecoder(nn.Module):
    """Decode a future-state representation from fused context tokens.

    Uses a learned query token that attends to the full context,
    optionally conditioned by ego-state or time offset.
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.q_norm = nn.LayerNorm(embed_dim)
        self.kv_norm = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = GatedFFN(embed_dim=embed_dim, mlp_ratio=4.0, dropout=dropout)

        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(
        self,
        context_tokens: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        context_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        context_tokens: (B, N, D)
        condition: (B, D) optional additive conditioning

        Returns: (B, D) -- decoded future representation
        """
        B = context_tokens.size(0)
        q = self.query.expand(B, -1, -1)  # (B, 1, D)
        if condition is not None:
            q = q + condition.unsqueeze(1)

        q_normed = self.q_norm(q)
        kv_normed = self.kv_norm(context_tokens)
        attn_out, _ = self.attn(
            q_normed, kv_normed, kv_normed,
            key_padding_mask=context_key_padding_mask,
        )
        q = q + attn_out

        # FFN operates on (B, 1, D) -- GatedFFN handles it fine
        q = self.ffn(q)

        return q.squeeze(1)  # (B, D)


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class ChironMultiModalPredictor(nn.Module):
    """
    CHIRON Multi-Modal Predictor.

    Extends the CHIRON channel backbone with image encoder, ego-state
    conditioning, and gated cross-modal fusion for channel prediction.

    Interface is compatible with MultiModalPredictator for drop-in use.
    """

    VALID_MODES = ("multimodal", "channel_only", "image_only")

    def __init__(
        self,
        mode: str = "multimodal",
        # Channel backbone
        num_bs_antennas: int = 16,
        num_subcarriers: int = 512,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 4,
        history_len: int = 32,
        patch_h: int = 4,
        patch_w: int = 32,
        conv_kernel: int = 7,
        mlp_ratio: float = 4.0,
        use_temporal_attention: bool = True,
        # Image encoder
        pretrained_image: bool = True,
        image_grid_size: int = 7,
        max_image_frames: int = 4,
        # LiDAR encoder
        use_lidar: bool = False,
        lidar_max_points: int = 64,
        lidar_num_tokens: int = 16,
        # Sensor flags
        use_image: bool = True,
        # Fusion
        fusion_layers: int = 3,
        fusion_heads: int = 4,
        # Ego state
        ego_state_dim: int = 6,
        use_ego_state: bool = True,
        # Prediction head
        prediction_head_hidden: int = 1024,
        prediction_horizon: int = 1,
        # General
        dropout: float = 0.1,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.mode = mode
        self.num_bs_antennas = num_bs_antennas
        self.num_subcarriers = num_subcarriers
        self.embed_dim = embed_dim
        self.depth = depth
        self.history_len = history_len
        self.prediction_horizon = prediction_horizon
        self.max_image_frames = max_image_frames
        self.use_ego_state = use_ego_state
        self.use_image = use_image
        self.use_lidar = use_lidar

        # ---- Channel backbone (shared with ChironChannelPredictor) ----
        if mode in ("multimodal", "channel_only"):
            self.patch_embed = PatchEmbed2D(
                num_antennas=num_bs_antennas,
                num_subcarriers=num_subcarriers,
                patch_h=patch_h,
                patch_w=patch_w,
                embed_dim=embed_dim,
            )
            self._num_spatial = self.patch_embed.num_patches

            self.temporal_pos = nn.Parameter(
                torch.zeros(1, history_len, 1, embed_dim)
            )
            self.spatial_pos = nn.Parameter(
                torch.zeros(1, 1, self._num_spatial, embed_dim)
            )
            nn.init.trunc_normal_(self.temporal_pos, std=0.02)
            nn.init.trunc_normal_(self.spatial_pos, std=0.02)

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
            self.channel_norm = nn.LayerNorm(embed_dim)
        else:
            self.patch_embed = None
            self._num_spatial = 0
            self.temporal_pos = None
            self.spatial_pos = None
            self.blocks = None
            self.channel_norm = None

        # ---- Image encoder ----
        _need_image = (mode in ("multimodal", "image_only")) and use_image
        if _need_image:
            self.image_encoder = ImageTokenEncoder(
                embed_dim=embed_dim,
                pretrained=pretrained_image,
                grid_size=image_grid_size,
            )
            self.image_seq_encoder = ImageSequenceEncoder(
                embed_dim=embed_dim,
                max_image_frames=max_image_frames,
                tokens_per_frame=image_grid_size * image_grid_size,
                num_heads=fusion_heads,
                dropout=dropout,
            )
            self.no_image_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.no_image_token, std=0.02)
        else:
            self.image_encoder = None
            self.image_seq_encoder = None
            self.no_image_token = None

        # ---- LiDAR encoder ----
        if use_lidar and mode in ("multimodal",):
            self.lidar_encoder = PointNetEncoder(
                point_dim=4,
                embed_dim=embed_dim,
                max_points=lidar_max_points,
                num_tokens=lidar_num_tokens,
                num_heads=fusion_heads,
                dropout=dropout,
            )
            self.no_lidar_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.no_lidar_token, std=0.02)
        else:
            self.lidar_encoder = None
            self.no_lidar_token = None

        # ---- Cross-modal fusion ----
        _need_fusion = mode == "multimodal" and (use_image or use_lidar)
        if _need_fusion:
            self.fusion_blocks = nn.ModuleList([
                GatedCrossModalFusion(
                    embed_dim=embed_dim,
                    num_heads=fusion_heads,
                    dropout=dropout,
                )
                for _ in range(fusion_layers)
            ])
        else:
            self.fusion_blocks = None

        # ---- Ego state encoder ----
        if use_ego_state:
            self.ego_encoder = EgoStateEncoder(
                state_dim=ego_state_dim,
                embed_dim=embed_dim,
            )
        else:
            self.ego_encoder = None

        # ---- Future query decoder ----
        self.decoder = FutureQueryDecoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # ---- Prediction head ----
        self.head = ChannelPredictionHead(
            embed_dim=embed_dim,
            num_patches=max(self._num_spatial, 1),
            num_antennas=num_bs_antennas,
            num_subcarriers=num_subcarriers,
            hidden_dim=prediction_head_hidden,
            dropout=dropout,
            prediction_horizon=prediction_horizon,
        )

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

    # ---- Channel backbone forward ----

    def _encode_channel(self, channel_history: torch.Tensor) -> torch.Tensor:
        """Run CHIRON backbone on channel history.

        channel_history: (B, K, Na, Nsc, 2)
        Returns: (B, K*S, D) -- all temporal-spatial tokens
        """
        B, K, Na, Nsc, _ = channel_history.shape
        S = self._num_spatial

        x = channel_history

        # Patch embed each frame
        x = x.reshape(B * K, Na, Nsc, 2)
        tokens = self.patch_embed(x)  # (B*K, S, D)
        tokens = tokens.view(B, K, S, self.embed_dim)

        # Add positional embeddings
        tokens = tokens + self.temporal_pos[:, :K] + self.spatial_pos

        # Flatten to sequence
        tokens = tokens.reshape(B, K * S, self.embed_dim)

        # Backbone
        for block in self.blocks:
            tokens = block(tokens, K, S)

        tokens = self.channel_norm(tokens)

        return tokens  # (B, K*S, D)

    # ---- Image encoding forward ----

    def _encode_images(
        self,
        image_seq: torch.Tensor,
        image_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode image sequence into tokens.

        image_seq: (B, T, 3, H, W) or (B, 3, H, W)
        Returns: (image_tokens, token_padding_mask)
        """
        return self.image_seq_encoder(
            self.image_encoder,
            image_seq,
            image_valid_mask=image_valid_mask,
        )

    # ---- Main forward ----

    def forward(
        self,
        channel_history: Optional[torch.Tensor] = None,
        image_seq: Optional[torch.Tensor] = None,
        image_valid_mask: Optional[torch.Tensor] = None,
        lidar_points: Optional[torch.Tensor] = None,
        lidar_mask: Optional[torch.Tensor] = None,
        ego_state: Optional[torch.Tensor] = None,
        # Legacy single-image interface compatibility
        image: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        channel_history : (B, K, Na, Nsc, 2), optional
        image_seq : (B, T, 3, H, W) or (B, 3, H, W), optional
        image_valid_mask : (B, T) bool, optional
        lidar_points : (B, max_points, 4), optional
            Zero-padded LiDAR point cloud (x, y, z, intensity).
        lidar_mask : (B, max_points) bool, optional
            True = valid point in lidar_points.
        ego_state : (B, ego_state_dim), optional
        image : (B, 3, H, W), optional — legacy single-image input.

        Returns
        -------
        pred : (B, P, Na, Nsc, 2)
        """
        # Handle legacy single-image input
        if image_seq is None and image is not None:
            image_seq = image.unsqueeze(1)

        # Ego-state conditioning
        condition = None
        if self.use_ego_state and self.ego_encoder is not None and ego_state is not None:
            condition = self.ego_encoder(ego_state)

        # ---- Multimodal ----
        if self.mode == "multimodal":
            assert channel_history is not None, "multimodal mode requires channel_history"

            channel_tokens = self._encode_channel(channel_history)  # (B, K*S, D)
            B = channel_tokens.size(0)

            # Collect sensor tokens + masks
            sensor_tokens_list = []
            sensor_masks_list = []

            if self.use_image and self.image_encoder is not None:
                if image_seq is None:
                    tokens_per_frame = getattr(self.image_encoder, "tokens_per_frame", 49)
                    img_tokens = self.no_image_token.expand(B, tokens_per_frame, -1)
                else:
                    if image_valid_mask is not None:
                        img_valid = image_valid_mask.bool()
                        if img_valid.dim() == 1:
                            img_valid = img_valid.unsqueeze(1)
                    else:
                        img_valid = None

                    img_tokens, _ = self._encode_images(
                        image_seq, img_valid,
                    )
                    if img_valid is None:
                        token_valid = torch.ones(
                            B, img_tokens.size(1), dtype=torch.bool, device=img_tokens.device,
                        )
                    else:
                        tokens_per_frame = getattr(self.image_encoder, "tokens_per_frame", 49)
                        num_frames = img_tokens.size(1) // tokens_per_frame
                        token_valid = img_valid[:, :num_frames].to(img_tokens.device)
                        token_valid = token_valid.unsqueeze(-1).expand(
                            B, num_frames, tokens_per_frame,
                        ).reshape(B, img_tokens.size(1))

                    no_img_tokens = self.no_image_token.expand(B, img_tokens.size(1), -1)
                    img_tokens = torch.where(
                        token_valid.view(B, -1, 1),
                        img_tokens,
                        no_img_tokens,
                    )
                sensor_tokens_list.append(img_tokens)
                sensor_masks_list.append(
                    torch.zeros(B, img_tokens.size(1), dtype=torch.bool, device=img_tokens.device)
                )

            if self.use_lidar and self.lidar_encoder is not None:
                if lidar_points is None:
                    tokens_per_cloud = getattr(self.lidar_encoder, "num_tokens", 16)
                    lidar_tokens = self.no_lidar_token.expand(B, tokens_per_cloud, -1)
                else:
                    lidar_tokens, _ = self.lidar_encoder(lidar_points, lidar_mask)
                    if lidar_mask is not None:
                        has_lidar = lidar_mask.bool().reshape(B, -1).any(dim=1)
                        no_lid_tokens = self.no_lidar_token.expand(B, lidar_tokens.size(1), -1)
                        lidar_tokens = torch.where(
                            has_lidar.to(lidar_tokens.device).view(B, 1, 1),
                            lidar_tokens,
                            no_lid_tokens,
                        )
                sensor_tokens_list.append(lidar_tokens)
                sensor_masks_list.append(
                    torch.zeros(B, lidar_tokens.size(1), dtype=torch.bool, device=lidar_tokens.device)
                )

            assert len(sensor_tokens_list) > 0, "multimodal mode requires at least one sensor input"

            # Concatenate all sensor tokens
            all_sensor_tokens = torch.cat(sensor_tokens_list, dim=1)
            all_sensor_masks = torch.cat(sensor_masks_list, dim=1)

            # Cross-modal fusion: channel tokens attend to sensor tokens
            fused = channel_tokens
            for fusion_block in self.fusion_blocks:
                fused = fusion_block(
                    fused, all_sensor_tokens,
                    image_key_padding_mask=all_sensor_masks,
                )

            return self.head(fused)

        # ---- Channel-only ----
        if self.mode == "channel_only":
            assert channel_history is not None, "channel_only mode requires channel_history"
            channel_tokens = self._encode_channel(channel_history)
            return self.head(channel_tokens)

        # ---- Image-only ----
        assert image_seq is not None, "image_only mode requires image input"
        image_tokens, img_pad_mask = self._encode_images(
            image_seq, image_valid_mask,
        )
        latent = self.decoder(
            image_tokens, condition=condition,
            context_key_padding_mask=img_pad_mask,
        )
        return self.head(latent.unsqueeze(1))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict[str, Any]:
        return {
            "model_type": "chiron_multimodal",
            "mode": self.mode,
            "num_bs_antennas": self.num_bs_antennas,
            "num_subcarriers": self.num_subcarriers,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "history_len": self.history_len,
            "prediction_horizon": self.prediction_horizon,
            "max_image_frames": self.max_image_frames,
            "use_ego_state": self.use_ego_state,
            "use_image": self.use_image,
            "use_lidar": self.use_lidar,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ChironMultiModalPredictor":
        config.pop("model_type", None)
        return cls(**config)
