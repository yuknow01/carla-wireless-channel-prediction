"""
models/image_encoders.py
========================
Shared image encoders for multimodal channel prediction models.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ImageTokenEncoder(nn.Module):
    """Encode RGB images into spatial tokens via a ResNet18 feature map.

    Input:  (B, 3, H, W)
    Output: (B, grid_size^2, D)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        pretrained: bool = True,
        grid_size: int = 7,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.tokens_per_frame = grid_size * grid_size

        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
        )
        self.stages = nn.Sequential(
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.proj = nn.Conv2d(512, embed_dim, kernel_size=1)
        self.spatial_pos = nn.Parameter(torch.zeros(1, grid_size * grid_size, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)

        nn.init.trunc_normal_(self.spatial_pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        feat = self.stages(feat)
        feat = self.pool(feat)
        feat = self.proj(feat)                      # (B, D, gs, gs)
        tokens = feat.flatten(2).transpose(1, 2)    # (B, gs^2, D)
        tokens = tokens + self.spatial_pos
        return self.norm(tokens)


__all__ = ["ImageTokenEncoder"]
