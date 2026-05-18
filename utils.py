"""
utils.py
========
Utility functions for channel prediction: metrics, conversions, normalization.

Channel format convention:
  - Complex channel: np.ndarray, shape (num_bs_antennas, num_subcarriers), complex128
  - Real representation: torch.Tensor, shape (num_bs_antennas, num_subcarriers, 2)
    where [..., 0] = real part, [..., 1] = imaginary part
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """
    Normalized Mean Squared Error in dB.

    NMSE = 10 * log10( E[ ||pred - target||^2_F / ||target||^2_F ] )

    Per-sample linear ratio averaged first, then converted to dB.
    Follows CPMamba (Eq.5a), CSI-4CAST (Eq.7), LLM4CP (Eq.7a) convention.

    Parameters
    ----------
    pred : Tensor, shape (batch, ...) or (...)
        Predicted channel in real representation.
    target : Tensor, same shape as pred
        Ground-truth channel in real representation.
    eps : float
        Small constant to avoid log(0).

    Returns
    -------
    Tensor, scalar
        NMSE in dB (linear ratio averaged, then dB).
    """
    if pred.dim() > 1 and pred.shape[0] > 1:
        p = pred.reshape(pred.shape[0], -1)
        t = target.reshape(target.shape[0], -1)
    else:
        p = pred.reshape(1, -1)
        t = target.reshape(1, -1)

    error_power = torch.sum((p - t) ** 2, dim=-1)
    target_power = torch.sum(t ** 2, dim=-1)
    ratio = error_power / (target_power + eps)
    return 10.0 * torch.log10(ratio.mean() + eps)


def cosine_similarity(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity of flattened channel vectors, averaged over batch.

    Parameters
    ----------
    pred : Tensor, shape (batch, ...) or (...)
    target : Tensor, same shape as pred

    Returns
    -------
    Tensor, scalar
        Mean cosine similarity in [0, 1].
    """
    if pred.dim() > 1 and pred.shape[0] > 1:
        p = pred.reshape(pred.shape[0], -1)
        t = target.reshape(target.shape[0], -1)
    else:
        p = pred.reshape(1, -1)
        t = target.reshape(1, -1)

    dot = torch.sum(p * t, dim=-1)
    norm_p = torch.norm(p, dim=-1)
    norm_t = torch.norm(t, dim=-1)
    sim = dot / (norm_p * norm_t + eps)
    return sim.mean()


# ---------------------------------------------------------------------------
# Channel format conversions
# ---------------------------------------------------------------------------

def channel_to_real(H_complex: np.ndarray) -> np.ndarray:
    """
    Convert a complex channel matrix to real representation.

    Parameters
    ----------
    H_complex : np.ndarray, shape (..., Na, Nsc), complex
        Complex-valued channel matrix.

    Returns
    -------
    np.ndarray, shape (..., Na, Nsc, 2), float64
        Real representation: [..., 0] = real, [..., 1] = imag.
    """
    return np.stack([H_complex.real, H_complex.imag], axis=-1)


def real_to_channel(H_real: np.ndarray) -> np.ndarray:
    """
    Convert a real representation back to complex channel matrix.

    Parameters
    ----------
    H_real : np.ndarray, shape (..., Na, Nsc, 2), float
        Real representation.

    Returns
    -------
    np.ndarray, shape (..., Na, Nsc), complex128
        Complex channel matrix.
    """
    return H_real[..., 0] + 1j * H_real[..., 1]


def channel_to_real_tensor(H_complex: np.ndarray) -> torch.Tensor:
    """
    Convert a complex numpy channel matrix to a real-valued torch Tensor.

    Parameters
    ----------
    H_complex : np.ndarray, shape (..., Na, Nsc), complex

    Returns
    -------
    torch.Tensor, shape (..., Na, Nsc, 2), float32
    """
    H_real = channel_to_real(H_complex)
    return torch.from_numpy(H_real.astype(np.float32))


def real_tensor_to_channel(H_real: torch.Tensor) -> np.ndarray:
    """
    Convert a real-valued torch Tensor back to complex numpy array.

    Parameters
    ----------
    H_real : torch.Tensor, shape (..., Na, Nsc, 2)

    Returns
    -------
    np.ndarray, shape (..., Na, Nsc), complex128
    """
    h = H_real.detach().cpu().numpy()
    return real_to_channel(h)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_channel(
    H_real: torch.Tensor,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Min-Max normalize a real-representation channel tensor to [0, 1].

    Parameters
    ----------
    H_real : Tensor, shape (..., Na, Nsc, 2)
    min_val : Tensor, shape (Na, Nsc, 2) or broadcastable
    max_val : Tensor, shape (Na, Nsc, 2) or broadcastable
    eps : float

    Returns
    -------
    Tensor, same shape as H_real
    """
    return (H_real - min_val) / (max_val - min_val + eps)


def denormalize_channel(
    H_norm: torch.Tensor,
    min_val: torch.Tensor,
    max_val: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Reverse Min-Max normalization of a channel tensor.
    """
    return H_norm * (max_val - min_val + eps) + min_val


def compute_dataset_stats(
    dataset_path: str,
    num_bs_antennas: int = 16,
    num_subcarriers: int = 64,
    max_samples: Optional[int] = None,
    sc_indices: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per-element min and max over all channel files in a dataset directory.

    Scans for `channel_*.npy` files under `dataset_path` (recursively)
    and computes statistics in real representation.

    Parameters
    ----------
    dataset_path : str
        Root path of a scenario or dataset directory.
    num_bs_antennas : int
    num_subcarriers : int
    max_samples : int
        Maximum number of channel files to read (for speed).
    sc_indices : np.ndarray or None
        Optional subcarrier indices to keep before accumulating statistics.

    Returns
    -------
    min_val : np.ndarray, shape (Na, Nsc, 2)
    max_val : np.ndarray, shape (Na, Nsc, 2)
    """
    channel_files = sorted(Path(dataset_path).rglob("channel_*.npy"))
    if len(channel_files) == 0:
        raise FileNotFoundError(f"No channel_*.npy files found under {dataset_path}")

    if max_samples is not None and max_samples > 0 and len(channel_files) > max_samples:
        # Evenly subsample
        indices = np.linspace(0, len(channel_files) - 1, max_samples, dtype=int)
        channel_files = [channel_files[i] for i in indices]

    min_acc = np.full((num_bs_antennas, num_subcarriers, 2), np.inf, dtype=np.float64)
    max_acc = np.full((num_bs_antennas, num_subcarriers, 2), -np.inf, dtype=np.float64)

    for cf in channel_files:
        H = np.load(cf)
        H_real = channel_to_real(H).astype(np.float64)
        if sc_indices is not None:
            H_real = H_real[:, sc_indices, :]
        elif H_real.shape[1] != num_subcarriers:
            if H_real.shape[1] < num_subcarriers:
                raise ValueError(
                    f"{cf} has only {H_real.shape[1]} subcarriers, "
                    f"but num_subcarriers={num_subcarriers}"
                )
            H_real = H_real[:, :num_subcarriers, :]
        min_acc = np.minimum(min_acc, H_real)
        max_acc = np.maximum(max_acc, H_real)

    return min_acc, max_acc


def save_stats(min_val: np.ndarray, max_val: np.ndarray, path: str):
    """Save Min-Max normalization statistics to a .npz file."""
    np.savez(path, min_val=min_val, max_val=max_val)


def load_stats(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load Min-Max normalization statistics from a .npz file."""
    data = np.load(path)
    return data["min_val"], data["max_val"]
