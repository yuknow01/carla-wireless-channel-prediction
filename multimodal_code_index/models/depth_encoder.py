"""
models/depth_encoder.py
=======================
Depth-pretrained image encoder for multimodal channel prediction.

Uses MiDaS Small (EfficientNet-Lite3 backbone) pretrained on depth estimation,
which gives spatial/geometric features directly relevant to channel prediction:
  - Object distances (BS-UE range)
  - Vehicle positions (AoD estimation)
  - Obstacle detection (LOS/NLOS)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthEncoderSimple(nn.Module):
    """
    Depth encoder: runs MiDaS to get a depth map,
    then extracts spatial features via a small CNN.

    The model explicitly sees depth information, which is directly
    relevant to wireless channel prediction (path loss, LOS/NLOS).
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()

        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.midas.requires_grad_(False)

        # Depth map -> features via small CNN
        self.depth_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 7, stride=4, padding=3),
            nn.GELU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
        )

        self.proj = nn.Sequential(
            nn.Linear(128 * 4 * 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224)
        Returns: (B, embed_dim)
        """
        self.midas.eval()
        with torch.no_grad():
            depth = self.midas(x)

        depth = depth - depth.amin(dim=(1, 2), keepdim=True)
        depth = depth / (depth.amax(dim=(1, 2), keepdim=True) + 1e-6)

        depth = depth.unsqueeze(1)
        features = self.depth_cnn(depth)
        features = features.flatten(1)
        return self.proj(features)
