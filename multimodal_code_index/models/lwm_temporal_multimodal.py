"""
models/lwm_temporal_multimodal.py
==================================
LWM_Temporal 멀티모달 채널 예측기 — CLS Token Scene Context Injection 방식.

Architecture
------------
channel_history (B, K, Na, Nsc, 2)
    │
    ▼  Complex 변환 + P 빈 미래 프레임 append
    │
(B, K+P, Na, Nsc) complex
    │
    ├─ [multimodal only]
    │   ImageTokenEncoder over T image frames → (B, T*49, D)  ┐
    │   PointNetEncoder                      → (B, 16, D)    ┘ sensor_tokens
    │       │
    │       ▼  scene_query cross-attn → scene_ctx (B, 1, D)
    │                                   = cls_inject
    │
    ▼  _LWMModelCLSInject (global_cls=True)
    │   patch_embed → (B, T*H*W, D)
    │   cls_token + scene_ctx → append at end → (B, T*H*W+1, D)
    │   SparseSpatioTemporalAttention × depth
    │     ↑ 모든 채널 토큰이 CLS를 neighbor로 포함 → scene context 전파
    │   reconstruction head (CLS 제외) → (B, T*H*W, ph*pw*2)
    │
    ▼  마지막 P 프레임 토큰 추출 → reshape
    │
(B, P, Na, Nsc, 2)

CLS Injection 방식 선택 이유 (보고서 12 참조)
---------------------------------------------
sensor 토큰을 채널 토큰 앞에 concat하면 NeighborIndexer가 T, H, W를
기준으로 neighbor를 계산하므로 재계산이 필요하다.
CLS 초기화 방식은 NeighborIndexer 변경 없이 global scene context 주입 가능.
NeighborIndexer: include_cls=True일 때 모든 채널 토큰이 CLS를 neighbor로
포함하고, CLS는 모든 채널 토큰에 attend → global access 자동 보장.

Modes
-----
"multimodal"   : channel + sensor(s) — scene_ctx를 CLS에 주입
"channel_only" : channel history only — CLS는 global summary로만 동작
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from models.lwm_temporal import LWMConfig, LWMModel
from models.sensor_encoders import (
    ImageTokenEncoder,
    PointNetEncoder,
)


# ---------------------------------------------------------------------------
# LWMModel with CLS Injection support
# ---------------------------------------------------------------------------

class _LWMModelCLSInject(LWMModel):
    """LWMModel subclass that accepts an optional cls_inject tensor.

    cls_inject: (B, 1, D) — added to the CLS token BEFORE encoder pass.
    When None, behaves identically to the original LWMModel.
    """

    def forward_tokens(
        self,
        tokens: Tensor,
        mask: Tensor,
        T: int,
        H: int,
        W: int,
        *,
        return_cls: bool = False,
        cls_inject: Optional[Tensor] = None,
    ) -> Dict:
        embeddings = self.patch_embed(tokens)
        include_cls = self.global_cls
        if include_cls:
            cls_tokens = self.cls_token.expand(embeddings.size(0), -1, -1)  # (B, 1, D)
            if cls_inject is not None:
                cls_tokens = cls_tokens + cls_inject   # scene context 주입
            embeddings = torch.cat([embeddings, cls_tokens], dim=1)
            cls_mask = torch.zeros(
                (embeddings.size(0), 1), dtype=torch.bool, device=embeddings.device
            )
            mask = torch.cat([mask, cls_mask], dim=1)

        embeddings = self._add_positional(embeddings)
        embeddings = embeddings.masked_fill(mask.unsqueeze(-1), 0.0)
        encoded = self.encoder(embeddings, T, H, W, include_cls)

        if include_cls:
            reconstruction = self.head(encoded[:, :-1, :])   # CLS 제외
            cls = encoded[:, -1, :]
        else:
            reconstruction = self.head(encoded)
            cls = None

        return {"reconstruction": reconstruction, "cls": cls if return_cls else None}

    def forward(
        self,
        seq: Tensor,
        mask: Optional[Tensor] = None,
        *,
        return_cls: bool = False,
        cls_inject: Optional[Tensor] = None,
    ) -> Dict:
        tokens, base_mask = self.tokenizer(seq, self.config.patch_size)
        total_mask = base_mask if mask is None else mask
        ph, pw = self.config.patch_size
        T = seq.size(1)
        H = seq.size(2) // ph
        W = seq.size(3) // pw
        return self.forward_tokens(
            tokens, total_mask, T, H, W,
            return_cls=return_cls,
            cls_inject=cls_inject,
        )


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class LWMTemporalMultiModalPredictor(nn.Module):
    """LWM_Temporal-based multimodal channel predictor."""

    VALID_MODES = ("multimodal", "channel_only")

    def __init__(
        self,
        mode: str = "multimodal",
        # Channel backbone (LWMConfig 직접 받거나 아래 개별 파라미터 사용)
        num_bs_antennas: int = 16,
        num_subcarriers: int = 64,
        history_len: int = 16,
        prediction_horizon: int = 4,
        patch_h: int = 4,
        patch_w: int = 16,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        # Sensor encoder
        use_image: bool = True,
        use_lidar: bool = True,
        pretrained_image: bool = True,
        lidar_max_points: int = 64,
        lidar_num_tokens: int = 16,
        # Scene context compression
        scene_attn_heads: int = 4,
        # Sparse attention 설정 (LWMConfig 기본값 오버라이드)
        same_frame_window: int = -1,
        temporal_offsets: tuple = (-4, -3, -2, -1, 1, 2, 3),
        temporal_spatial_window: int = 2,
        temporal_drift_h: int = 1,
        temporal_drift_w: int = 1,
        routing_topk_enable: bool = True,
        routing_topk_fraction: float = 0.3,
        routing_topk_min: int = 8,
        routing_topk_max: int = 48,
        routing_topk_per_head: bool = True,
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")

        self.mode = mode
        self.num_bs_antennas = num_bs_antennas
        self.num_subcarriers = num_subcarriers
        self.history_len = history_len
        self.prediction_horizon = prediction_horizon
        self.embed_dim = embed_dim
        self.use_image = use_image
        self.use_lidar = use_lidar

        ph, pw = patch_h, patch_w
        self.ph, self.pw = ph, pw
        self.H = num_bs_antennas // ph    # spatial height
        self.W = num_subcarriers // pw    # spatial width
        self.tokens_per_frame = self.H * self.W

        # LWMConfig — global_cls=True (CLS token 항상 사용)
        cfg = LWMConfig(
            patch_size=(ph, pw),
            phase_mode="real_imag",
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=4.0,
            same_frame_window=same_frame_window,
            temporal_offsets=temporal_offsets,
            temporal_spatial_window=temporal_spatial_window,
            temporal_drift_h=temporal_drift_h,
            temporal_drift_w=temporal_drift_w,
            routing_topk_enable=routing_topk_enable,
            routing_topk_fraction=routing_topk_fraction,
            routing_topk_min=routing_topk_min,
            routing_topk_max=routing_topk_max,
            routing_topk_per_head=routing_topk_per_head,
            global_cls=True,         # CLS injection을 위해 반드시 True
            posenc="learned",
            max_seq_len=(history_len + prediction_horizon) * self.tokens_per_frame,
        )
        self.lwm = _LWMModelCLSInject(cfg)

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

        # Scene context compression: sensor_tokens → (B, 1, D)
        if _need_sensor:
            self.scene_query = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.scene_query, std=0.02)
            self.scene_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=scene_attn_heads,
                batch_first=True,
            )
            self.scene_norm = nn.LayerNorm(embed_dim)
        else:
            self.scene_query = None
            self.scene_attn = None
            self.scene_norm = None

    def _compute_scene_ctx(
        self,
        image_seq: Optional[Tensor],
        image_valid_mask: Optional[Tensor],
        lidar_points: Optional[Tensor],
        lidar_mask: Optional[Tensor],
        B: int,
    ) -> Optional[Tensor]:
        """Sensor 토큰 → scene context (B, 1, D).

        image_seq:    (B, 3, H, W) or (B, T, 3, H, W)
        image_valid_mask: (B,) or (B, T) bool
        lidar_points: (B, max_points, 4)
        lidar_mask:   (B, max_points) bool

        Returns None if no sensor input available.
        """
        sensor_list = []

        if self.use_image and self.image_encoder is not None:
            if image_seq is None:
                tokens_per_frame = getattr(self.image_encoder, "tokens_per_frame", 49)
                img_tokens = self.no_image_token.expand(B, tokens_per_frame, -1)
            else:
                if image_seq.dim() == 5:
                    _, T, C, H, W = image_seq.shape
                    flat_images = image_seq.reshape(B * T, C, H, W)
                    frame_tokens = self.image_encoder(flat_images)  # (B*T, 49, D)
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

                    real_img_tokens = self.image_encoder(image_seq)  # (B, 49, D)
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
                lid_tokens, _ = self.lidar_encoder(lidar_points, lidar_mask)  # (B, 16, D)
                if lidar_mask is not None:
                    has_lidar = lidar_mask.bool().reshape(B, -1).any(dim=1)
                    no_lid_tokens = self.no_lidar_token.expand(B, lid_tokens.size(1), -1)
                    lid_tokens = torch.where(
                        has_lidar.to(lid_tokens.device).view(B, 1, 1),
                        lid_tokens,
                        no_lid_tokens,
                    )
            sensor_list.append(lid_tokens)

        if not sensor_list:
            return None

        sensor_tokens = torch.cat(sensor_list, dim=1)      # (B, Ns, D)

        # Cross-attention: scene_query attends to sensor_tokens → (B, 1, D)
        q = self.scene_query.expand(B, -1, -1)
        scene_ctx, _ = self.scene_attn(q, sensor_tokens, sensor_tokens)
        return self.scene_norm(scene_ctx)                  # (B, 1, D)

    def forward(
        self,
        channel_history: Tensor,
        image_seq: Optional[Tensor] = None,
        image_valid_mask: Optional[Tensor] = None,
        lidar_points: Optional[Tensor] = None,
        lidar_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
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
        B, K, Na, Nsc, _ = channel_history.shape
        ph, pw = self.ph, self.pw
        P = self.prediction_horizon

        # Step 1: real representation → complex
        x = channel_history.float()
        x_c = torch.complex(x[..., 0], x[..., 1])           # (B, K, Na, Nsc)

        # Step 2: append P blank future frames (masked target)
        future = torch.zeros(B, P, Na, Nsc, dtype=x_c.dtype, device=x_c.device)
        seq = torch.cat([x_c, future], dim=1)                # (B, K+P, Na, Nsc)
        T = K + P

        # Step 3: mask — future frames masked, history unmasked
        mask = torch.zeros(B, T * self.tokens_per_frame, dtype=torch.bool, device=x.device)
        mask[:, -P * self.tokens_per_frame:] = True

        # Step 4: scene context (multimodal only)
        cls_inject = None
        if self.mode == "multimodal":
            cls_inject = self._compute_scene_ctx(
                image_seq, image_valid_mask, lidar_points, lidar_mask, B,
            )

        # Step 5: LWM forward with CLS injection
        recon = self.lwm(seq, mask, cls_inject=cls_inject)["reconstruction"]
        # recon: (B, T*H*W, ph*pw*2)  — CLS already excluded in forward_tokens

        # Step 6: extract predicted future tokens
        pred_tokens = recon[:, -P * self.tokens_per_frame:, :]
        # (B, P*H*W, ph*pw*2) → (B, P, H, ph, W, pw, 2)
        pred_tokens = pred_tokens.view(B, P, self.H, self.W, ph, pw, 2)
        pred = pred_tokens.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        return pred.view(B, P, Na, Nsc, 2)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self) -> Dict:
        return {
            "model_type": "lwm_temporal_multimodal",
            "mode": self.mode,
            "num_bs_antennas": self.num_bs_antennas,
            "num_subcarriers": self.num_subcarriers,
            "history_len": self.history_len,
            "prediction_horizon": self.prediction_horizon,
            "embed_dim": self.embed_dim,
            "use_image": self.use_image,
            "use_lidar": self.use_lidar,
        }
