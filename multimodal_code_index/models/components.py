"""
models/components.py
====================
Sub-modules for the multimodal channel prediction model.

- ImageEncoder:          ResNet18 backbone with projection to 256-dim
- ChannelEncoder:        1D CNN or Transformer over K past channel measurements
- CrossAttentionFusion:  Image features as queries, channel features as keys/values
- PredictionHead:        MLP outputting predicted channel (Na, Nsc, 2)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


# ---------------------------------------------------------------------------
# ImageEncoder
# ---------------------------------------------------------------------------

class ImageEncoder(nn.Module):
    """
    Encodes an RGB image (224x224) into a feature vector using a pretrained
    ResNet18 backbone with the final FC layer replaced by a projection head.

    Output: (batch, embed_dim)
    """

    def __init__(self, embed_dim: int = 256, pretrained: bool = True):
        super().__init__()
        weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = tv_models.resnet18(weights=weights)
        # Remove the final fully-connected layer
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # -> (B, 512, 1, 1)
        self.proj = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, 3, 224, 224)

        Returns
        -------
        Tensor, shape (batch, embed_dim)
        """
        feat = self.features(x)         # (B, 512, 1, 1)
        feat = feat.flatten(start_dim=1)  # (B, 512)
        return self.proj(feat)            # (B, embed_dim)


# ---------------------------------------------------------------------------
# Positional Encoding (for Transformer variant)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ---------------------------------------------------------------------------
# ChannelEncoder
# ---------------------------------------------------------------------------

class ChannelEncoder(nn.Module):
    """
    Encodes a sequence of K past channel measurements into a fixed-size vector.

    Input:  (batch, K, Na, Nsc, 2) — real channel history
    Output: (batch, embed_dim)

    Uses Conv2D to spatially encode each frame, then processes the temporal
    sequence with either a Transformer or 1D CNN.

    Parameters
    ----------
    num_bs_antennas : int
        Number of base station antennas (default: 16).
    num_subcarriers : int
        Number of OFDM subcarriers (default: 512).
    embed_dim : int
        Output embedding dimension (default: 256).
    max_history : int
        Maximum history length K (default: 64).
    encoder_type : str
        "cnn" or "transformer".
    num_heads : int
        Number of attention heads (transformer mode).
    num_layers : int
        Number of encoder layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        num_bs_antennas: int = 16,
        num_subcarriers: int = 512,
        embed_dim: int = 256,
        max_history: int = 64,
        encoder_type: str = "transformer",
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        # Legacy parameter — ignored, kept for backwards compatibility
        input_dim: Optional[int] = None,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.embed_dim = embed_dim

        hidden_dim = embed_dim

        # Spatial encoder: (B*K, 2, Na, Nsc) → (B*K, hidden_dim)
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(4, 16), stride=(2, 8), padding=(1, 4)),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=(4, 8), stride=(2, 4), padding=(1, 2)),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        if encoder_type == "transformer":
            self.pos_enc = SinusoidalPositionalEncoding(hidden_dim, max_len=max_history, dropout=dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.pool = nn.Linear(hidden_dim, embed_dim)

        elif encoder_type == "cnn":
            self.cnn = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.pool = nn.Linear(hidden_dim, embed_dim)

        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Use 'cnn' or 'transformer'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, K, Na, Nsc, 2)

        Returns
        -------
        Tensor, shape (batch, embed_dim)
        """
        B, K, Na, Nsc, _ = x.shape

        # Spatial encoding per frame
        x = x.permute(0, 1, 4, 2, 3)          # (B, K, 2, Na, Nsc)
        x = x.reshape(B * K, 2, Na, Nsc)
        h = self.spatial_encoder(x)             # (B*K, hidden_dim)
        h = h.view(B, K, -1)                    # (B, K, hidden_dim)

        if self.encoder_type == "transformer":
            h = self.pos_enc(h)
            h = self.encoder(h)
            h = h.mean(dim=1)
            return self.pool(h)

        elif self.encoder_type == "cnn":
            h = h.transpose(1, 2)
            h = self.cnn(h)
            h = h.squeeze(-1)
            return self.pool(h)


# ---------------------------------------------------------------------------
# CrossAttentionFusion
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """
    Fuses image and channel embeddings using cross-attention.

    Image features serve as queries; channel features serve as keys and values.
    This allows the model to attend to channel information conditioned on what
    the image shows (e.g., vehicle positions visible in the image).

    Parameters
    ----------
    embed_dim : int
        Dimension of both image and channel embeddings.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout rate.
    """

    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Expand single-vector embeddings to short sequences for multi-head attention
        self.img_expand = nn.Linear(embed_dim, embed_dim * 4)
        self.ch_expand = nn.Linear(embed_dim, embed_dim * 4)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

    def forward(
        self,
        img_feat: torch.Tensor,
        ch_feat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        img_feat : Tensor, shape (batch, embed_dim)
        ch_feat  : Tensor, shape (batch, embed_dim)

        Returns
        -------
        Tensor, shape (batch, embed_dim)
            Fused representation.
        """
        B, D = img_feat.shape

        # Expand to sequences of length 4 for richer attention
        q = self.img_expand(img_feat).reshape(B, 4, D)  # (B, 4, D)
        kv = self.ch_expand(ch_feat).reshape(B, 4, D)   # (B, 4, D)

        # Cross-attention: image queries attend to channel keys/values
        attn_out, _ = self.cross_attn(q, kv, kv)  # (B, 4, D)
        attn_out = self.norm1(q + attn_out)        # residual + norm

        # Feed-forward
        ffn_out = self.ffn(attn_out)
        ffn_out = self.norm2(attn_out + ffn_out)   # residual + norm

        # Pool back to single vector
        fused = ffn_out.mean(dim=1)  # (B, D)
        return self.output_proj(fused)


# ---------------------------------------------------------------------------
# PredictionHead
# ---------------------------------------------------------------------------

class PredictionHead(nn.Module):
    """
    MLP that maps the fused feature vector to a predicted channel matrix.

    Output shape: (batch, num_bs_antennas, num_subcarriers, 2)

    Parameters
    ----------
    embed_dim : int
        Input feature dimension.
    num_bs_antennas : int
    num_subcarriers : int
    hidden_dim : int
        Width of hidden layers.
    num_layers : int
        Number of hidden layers (minimum 1).
    dropout : float
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_bs_antennas: int = 16,
        num_subcarriers: int = 64,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_bs_antennas = num_bs_antennas
        self.num_subcarriers = num_subcarriers
        output_dim = num_bs_antennas * num_subcarriers * 2

        layers = []
        in_dim = embed_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, embed_dim)

        Returns
        -------
        Tensor, shape (batch, num_bs_antennas, num_subcarriers, 2)
        """
        out = self.mlp(x)  # (B, Na * Nsc * 2)
        return out.reshape(-1, self.num_bs_antennas, self.num_subcarriers, 2)
