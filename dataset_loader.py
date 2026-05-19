"""
dataset_loader.py
=================
Dataset loader for channel prediction: loads synchronized image + channel data.

Given:
  - Past K channel measurements: H[t-K+1], ..., H[t]
  - One or more latest-past images up to time t

Predict:
  - Future channel: H[t+P] (P steps ahead)

Channel shape: (num_bs_antennas, num_subcarriers) = (16, 64), complex
Image: RGB, resized to 224x224

Supports:
  - Temporal train/val/test split (70/15/15, chronological, no leakage)
  - Normalization of channels (separate real/imag)
  - Configurable history length K and prediction horizon P
  - Handling of mismatched temporal resolution between images and channels
  - Synthetic data generation mode for testing without CARLA
"""

from __future__ import annotations

from bisect import bisect_right
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from utils import channel_to_real, normalize_channel


# ---------------------------------------------------------------------------
# Image loading and transforms
# ---------------------------------------------------------------------------

def _load_and_preprocess_image(path: str, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Load an image from disk, resize to (size), and convert to float32 [0, 1].

    Returns shape (3, H, W) for direct use as a torch tensor.
    """
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(path).convert("RGB")
    img = img.resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    return arr


# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization to a (3, H, W) float32 image in [0, 1]."""
    return (img - IMAGENET_MEAN) / IMAGENET_STD


def _discover_image_cache(sensor_dir_path: Path) -> Optional[Dict[str, str]]:
    cache_dir = sensor_dir_path / "image_cache"
    images_path = cache_dir / "images_224_uint8.npy"
    indices_path = cache_dir / "image_indices.npy"
    metadata_path = cache_dir / "metadata.json"

    if not images_path.exists() or not indices_path.exists():
        return None

    return {
        "images_path": str(images_path),
        "indices_path": str(indices_path),
        "metadata_path": str(metadata_path) if metadata_path.exists() else "",
    }


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _discover_scenario_data(scenario_dir: str, sensor_dir: Optional[str] = None) -> Dict:
    """
    Scan a scenario directory and return sorted lists of file paths
    with their indices.

    Returns
    -------
    dict with keys:
        "channel_indices": sorted list of int
        "channel_paths": dict int -> str
        "image_indices": sorted list of int
        "image_paths": dict int -> str
        "image_cache": dict or None
        "position_paths": dict int -> str
        "velocity_paths": dict int -> str
        "metadata": dict or None
    """
    scenario_dir = Path(scenario_dir)
    sensor_dir_path = Path(sensor_dir) if sensor_dir is not None else scenario_dir

    # Channels
    ch_dir = scenario_dir / "channels"
    ch_paths = {}
    if ch_dir.exists():
        for f in sorted(ch_dir.glob("channel_*.npy")):
            idx = int(f.stem.split("_")[-1])
            ch_paths[idx] = str(f)

    # Images can live in a separate scenario directory, e.g. channels from
    # dataset_final/scXX and 1ms camera frames from dataset_1ms/scXX.
    img_dir = sensor_dir_path / "images"
    img_paths = {}
    image_cache = _discover_image_cache(sensor_dir_path)
    img_indices_from_cache = []
    if image_cache is not None:
        img_indices_from_cache = [
            int(idx)
            for idx in np.load(image_cache["indices_path"], mmap_mode="r")
        ]
    elif img_dir.exists():
        for f in sorted(img_dir.glob("frame_*.png")):
            idx = int(f.stem.split("_")[-1])
            img_paths[idx] = str(f)

    # Positions
    pos_dir = scenario_dir / "positions"
    pos_paths = {}
    if pos_dir.exists():
        for f in sorted(pos_dir.glob("positions_*.npy")):
            idx = int(f.stem.split("_")[-1])
            pos_paths[idx] = str(f)

    # Velocities
    vel_dir = scenario_dir / "velocities"
    vel_paths = {}
    if vel_dir.exists():
        for f in sorted(vel_dir.glob("velocities_*.npy")):
            idx = int(f.stem.split("_")[-1])
            vel_paths[idx] = str(f)

    # Metadata
    meta_path = scenario_dir / "metadata.json"
    metadata = None
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    ch_indices = sorted(ch_paths.keys())
    img_indices = img_indices_from_cache or sorted(img_paths.keys())

    sensor_metadata = None
    sensor_meta_path = sensor_dir_path / "metadata.json"
    if sensor_meta_path.exists():
        with open(sensor_meta_path) as f:
            sensor_metadata = json.load(f)

    return {
        "channel_indices": ch_indices,
        "channel_paths": ch_paths,
        "image_indices": img_indices,
        "image_paths": img_paths,
        "image_cache": image_cache,
        "position_paths": pos_paths,
        "velocity_paths": vel_paths,
        "metadata": metadata,
        "sensor_metadata": sensor_metadata,
    }


def _find_latest_past_image_index(target_idx: int, image_indices: List[int]) -> Tuple[int, int]:
    """
    Find the latest image index less than or equal to target_idx.

    Returns
    -------
    (image_idx, target_idx - image_idx)
    """
    if not image_indices:
        return -1, -1

    pos = bisect_right(image_indices, target_idx) - 1
    if pos < 0:
        return -1, -1

    img_idx = image_indices[pos]
    return img_idx, target_idx - img_idx


def _find_latest_past_image_indices(
    target_idx: int,
    image_indices: List[int],
    num_frames: int,
    stride: int = 1,
) -> Tuple[List[int], List[int], List[bool]]:
    """
    Collect the latest valid past image indices in chronological order.

    Missing frames are padded on the left with invalid entries so the newest
    valid image remains aligned at the end of the returned sequence.
    """
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    if stride <= 0:
        raise ValueError(f"stride must be positive, got {stride}")

    if not image_indices:
        return [-1] * num_frames, [0] * num_frames, [False] * num_frames

    pos = bisect_right(image_indices, target_idx) - 1
    if pos < 0:
        return [-1] * num_frames, [0] * num_frames, [False] * num_frames

    collected_indices: List[int] = []
    collected_dists: List[int] = []
    cursor = pos
    while cursor >= 0 and len(collected_indices) < num_frames:
        img_idx = image_indices[cursor]
        collected_indices.append(img_idx)
        collected_dists.append(target_idx - img_idx)
        cursor -= stride

    collected_indices.reverse()
    collected_dists.reverse()

    pad = num_frames - len(collected_indices)
    return (
        [-1] * pad + collected_indices,
        [0] * pad + collected_dists,
        [False] * pad + [True] * len(collected_indices),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChannelPredictionDataset(Dataset):
    """
    Loads synchronized image + channel data for prediction.

    Given:
      - Past K channel measurements: H[t-K+1], ..., H[t]
      - One or more latest-past images up to time t

    Predict:
      - Future channel: H[t+P] (P steps ahead)

    Parameters
    ----------
    data_dir : str
        Path to the channel scenario directory (e.g., dataset_final/sc01/).
    sensor_data_dir : str or None
        Optional scenario directory for sensor files. If set, images are read
        from this directory while channels/positions/velocities still come from
        data_dir. This is intended for dataset_final channels + dataset_1ms
        sensors aligned by the same sim-step index.
    sc_indices : np.ndarray or None
        Optional subcarrier indices to keep from the loaded channel tensor.
    history_len : int
        Number of past channel measurements K (default: 32).
    prediction_horizon : int
        Number of steps P into the future to predict (default: 1).
    split : str
        "train", "val", or "test". Temporal split based on train_ratio/val_ratio.
    channel_mean : np.ndarray or None
        Min for channel normalization, shape (Na, Nsc, 2). If None, no normalization.
    channel_std : np.ndarray or None
        Max for channel normalization.
    delta_t : float
        Time between consecutive channel samples in seconds (default: 0.002).
    image_size : tuple
        Target image size (default: (224, 224)).
    use_image : bool
        Whether to load images (set False for channel_only mode for speed).
    num_image_frames : int
        Number of latest-past images to provide per sample (default: 1).
    image_stride : int
        Step size in the image index list when sampling image sequences.
    image_policy : str
        Image selection policy. Currently only "latest_past" is supported.
    pad_image_sequence : bool
        Whether to left-pad image sequences if insufficient past images exist.
    train_ratio : float
        Fraction of data for training (default: 0.75).
    val_ratio : float
        Fraction of data for validation (default: 0.25).
        Remainder (1 - train_ratio - val_ratio) goes to test. Set to 0 for no test set.
    """

    def __init__(
        self,
        data_dir: str,
        sensor_data_dir: Optional[str] = None,
        sc_indices: Optional[np.ndarray] = None,
        history_len: int = 16,
        prediction_horizon: int = 4,
        split: str = "train",
        channel_mean: Optional[np.ndarray] = None,
        channel_std: Optional[np.ndarray] = None,
        delta_t: float = 0.0005,
        image_size: Tuple[int, int] = (224, 224),
        use_image: bool = True,
        min_speed: float = 0.0,
        num_image_frames: int = 1,
        image_stride: int = 1,
        image_policy: str = "latest_past",
        pad_image_sequence: bool = True,
        train_ratio: float = 0.75,
        val_ratio: float = 0.25,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.sensor_data_dir = sensor_data_dir
        self.sc_indices = sc_indices
        self.history_len = history_len
        self.prediction_horizon = prediction_horizon
        self.split = split
        self.delta_t = delta_t
        self.image_size = image_size
        self.use_image = use_image
        self.min_speed = min_speed
        self.num_image_frames = num_image_frames
        self.image_stride = image_stride
        self.image_policy = image_policy
        self.pad_image_sequence = pad_image_sequence

        if self.num_image_frames <= 0:
            raise ValueError(f"num_image_frames must be positive, got {self.num_image_frames}")
        if self.image_stride <= 0:
            raise ValueError(f"image_stride must be positive, got {self.image_stride}")
        if self.image_policy != "latest_past":
            raise ValueError(
                f"Unsupported image_policy '{self.image_policy}'. Only 'latest_past' is supported."
            )

        # Normalization stats
        self.channel_mean = channel_mean
        self.channel_std = channel_std

        # Discover data files
        data_info = _discover_scenario_data(data_dir, sensor_dir=sensor_data_dir)
        self.channel_indices = data_info["channel_indices"]
        self.channel_paths = data_info["channel_paths"]
        self.image_indices = data_info["image_indices"]
        self.image_paths = data_info["image_paths"]
        self.image_cache = data_info["image_cache"]
        self.metadata = data_info["metadata"]
        self.sensor_metadata = data_info["sensor_metadata"]

        if len(self.channel_indices) == 0:
            raise FileNotFoundError(f"No channel data found in {data_dir}")
        if self.use_image and self.sensor_data_dir is not None and len(self.image_indices) == 0:
            raise FileNotFoundError(f"No image data found in sensor_data_dir={self.sensor_data_dir}")

        # Pre-load velocities to filter out stationary UE
        vel_dir = Path(data_dir) / "velocities"
        self._speed_cache = {}
        if vel_dir.exists() and min_speed > 0:
            for f in vel_dir.glob("velocities_*.npy"):
                idx = int(f.stem.split("_")[-1])
                vel = np.load(f)
                self._speed_cache[idx] = float(np.linalg.norm(vel))

        # Build valid sample indices:
        # A valid sample needs indices [t-K+1, ..., t, t+1, ..., t+P] all present.
        min_idx = self.channel_indices[0]
        max_idx = self.channel_indices[-1]

        # Set of available indices for O(1) lookup
        available = set(self.channel_indices)

        all_valid = []
        for t in self.channel_indices:
            # Need K past indices ending at t, and all P future targets t+1..t+P
            start = t - history_len + 1
            last_target = t + prediction_horizon
            if start < min_idx or last_target > max_idx:
                continue
            # Filter out stationary samples
            if self._speed_cache and min_speed > 0:
                speed_t = self._speed_cache.get(t, 0)
                if speed_t < min_speed:
                    continue
            # Check all required indices are present
            valid = True
            for k in range(start, t + 1):
                if k not in available:
                    valid = False
                    break
            if valid and all(t + p in available for p in range(1, prediction_horizon + 1)):
                all_valid.append(t)

        if len(all_valid) == 0:
            raise ValueError(
                f"No valid samples found in {data_dir} with K={history_len}, P={prediction_horizon}. "
                f"Channel index range: [{min_idx}, {max_idx}], total channels: {len(self.channel_indices)}"
            )

        # Temporal split (chronological)
        n = len(all_valid)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        if split == "train":
            self.valid_indices = all_valid[:n_train]
        elif split == "val":
            self.valid_indices = all_valid[n_train:n_train + n_val]
        elif split == "test":
            self.valid_indices = all_valid[n_train + n_val:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        # Pre-convert normalization stats to torch
        if self.channel_mean is not None:
            ch_min = self.channel_mean
            ch_max = self.channel_std
            if self.sc_indices is not None:
                ch_min = ch_min[:, self.sc_indices, :]
                ch_max = ch_max[:, self.sc_indices, :]
            self._min_t = torch.from_numpy(ch_min.astype(np.float32))
            self._max_t = torch.from_numpy(ch_max.astype(np.float32))
        else:
            self._min_t = None
            self._max_t = None

        self._zero_image = torch.zeros(
            3,
            self.image_size[0],
            self.image_size[1],
            dtype=torch.float32,
        )
        self._image_cache_array = None
        self._image_cache_rows = {}
        if self.use_image and self.image_cache is not None:
            self._image_cache_array = np.load(
                self.image_cache["images_path"],
                mmap_mode="r",
            )
            cache_indices = np.load(self.image_cache["indices_path"], mmap_mode="r")
            self._image_cache_rows = {
                int(img_idx): row for row, img_idx in enumerate(cache_indices)
            }

    def __len__(self) -> int:
        return len(self.valid_indices)

    def _load_cached_image(self, img_idx: int) -> np.ndarray:
        row = self._image_cache_rows[img_idx]
        img = np.asarray(self._image_cache_array[row], dtype=np.float32) / 255.0
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError(f"Cached image has invalid shape: {img.shape}")
        if img.shape[0] != self.image_size[0] or img.shape[1] != self.image_size[1]:
            raise ValueError(
                f"Cached image shape {img.shape[:2]} does not match requested {self.image_size}"
            )
        return img.transpose(2, 0, 1)

    def _load_image_sequence(self, t: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_seq: List[torch.Tensor] = []
        time_offsets: List[torch.Tensor] = []
        valid_mask: List[bool] = []

        if not self.use_image:
            num_frames = self.num_image_frames if self.pad_image_sequence else 1
            for _ in range(num_frames):
                image_seq.append(self._zero_image.clone())
                time_offsets.append(torch.tensor([0.0], dtype=torch.float32))
                valid_mask.append(False)
            return (
                torch.stack(image_seq, dim=0),
                torch.stack(time_offsets, dim=0),
                torch.tensor(valid_mask, dtype=torch.bool),
            )

        indices, dists, validity = _find_latest_past_image_indices(
            target_idx=t,
            image_indices=self.image_indices,
            num_frames=self.num_image_frames,
            stride=self.image_stride,
        )

        if not self.pad_image_sequence:
            filtered = [(idx, dist, valid) for idx, dist, valid in zip(indices, dists, validity) if valid]
            if not filtered:
                filtered = [(-1, 0, False)]
            indices = [item[0] for item in filtered]
            dists = [item[1] for item in filtered]
            validity = [item[2] for item in filtered]

        for img_idx, dist, is_valid in zip(indices, dists, validity):
            if is_valid and (img_idx in self._image_cache_rows or img_idx in self.image_paths):
                try:
                    if img_idx in self._image_cache_rows:
                        img = self._load_cached_image(img_idx)
                    else:
                        img = _load_and_preprocess_image(self.image_paths[img_idx], self.image_size)
                    img = _normalize_image(img)
                    image_seq.append(torch.from_numpy(img))
                    time_offsets.append(torch.tensor([dist * self.delta_t], dtype=torch.float32))
                    valid_mask.append(True)
                except (OSError, ValueError):
                    image_seq.append(self._zero_image.clone())
                    time_offsets.append(torch.tensor([0.0], dtype=torch.float32))
                    valid_mask.append(False)
            else:
                image_seq.append(self._zero_image.clone())
                time_offsets.append(torch.tensor([0.0], dtype=torch.float32))
                valid_mask.append(False)

        return (
            torch.stack(image_seq, dim=0),
            torch.stack(time_offsets, dim=0),
            torch.tensor(valid_mask, dtype=torch.bool),
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            "channel_history": Tensor (K, Na, Nsc, 2) — real channel history
            "target":          Tensor (P, Na, Nsc, 2) — P consecutive future channels
            "image_seq":       Tensor (T_img, 3, 224, 224) — latest-past image sequence
            "image_time_offsets": Tensor (T_img, 1) — seconds since each image
            "image_valid_mask": Tensor (T_img,) — valid frames in image_seq
            "sample_index":    int — the time index t for this sample
        """
        t = self.valid_indices[idx]
        K = self.history_len
        P = self.prediction_horizon

        # --- Load channel history ---
        history = []
        for k in range(t - K + 1, t + 1):
            H = np.load(self.channel_paths[k])  # complex128, (Na, Nsc)
            H_real = channel_to_real(H).astype(np.float32)  # (Na, Nsc, 2)
            history.append(H_real)
        history = np.stack(history, axis=0)  # (K, Na, Nsc, 2)

        # --- Load P consecutive target frames ---
        target_frames = []
        for p in range(1, P + 1):
            H_t = np.load(self.channel_paths[t + p])
            target_frames.append(channel_to_real(H_t).astype(np.float32))  # (Na, Nsc, 2)
        target = np.stack(target_frames, axis=0)  # (P, Na, Nsc, 2)

        if self.sc_indices is not None:
            history = history[:, :, self.sc_indices, :]
            target = target[:, :, self.sc_indices, :]

        # --- Normalize channels ---
        if self._min_t is not None:
            history_t = torch.from_numpy(history)
            target_t = torch.from_numpy(target)
            history_t = normalize_channel(history_t, self._min_t, self._max_t)
            # normalize each of the P target frames independently
            target_t = torch.stack(
                [normalize_channel(target_t[p], self._min_t, self._max_t) for p in range(P)],
                dim=0,
            )
        else:
            history_t = torch.from_numpy(history)
            target_t = torch.from_numpy(target)

        image_seq_t, image_dt_t, image_valid_t = self._load_image_sequence(t)
        image_t = image_seq_t[-1]
        time_since_image = image_dt_t[-1]

        return {
            "channel_history": history_t,  # (K, Na, Nsc, 2)
            "target": target_t,
            "image_seq": image_seq_t,
            "image_time_offsets": image_dt_t,
            "image_valid_mask": image_valid_t,
            "image": image_t,
            "time_since_image": time_since_image,
            "sample_index": t,
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    data_dir: str,
    sensor_data_dir: Optional[str] = None,
    sc_indices: Optional[np.ndarray] = None,
    history_len: int = 16,
    prediction_horizon: int = 4,
    batch_size: int = 32,
    num_workers: int = 4,
    channel_mean: Optional[np.ndarray] = None,
    channel_std: Optional[np.ndarray] = None,
    delta_t: float = 0.0005,
    use_image: bool = True,
    min_speed: float = 0.0,
    num_image_frames: int = 1,
    image_stride: int = 1,
    image_policy: str = "latest_past",
    pad_image_sequence: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test DataLoaders for a scenario directory.

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    loaders = []
    for split in ("train", "val", "test"):
        ds = ChannelPredictionDataset(
            data_dir=data_dir,
            sensor_data_dir=sensor_data_dir,
            sc_indices=sc_indices,
            history_len=history_len,
            prediction_horizon=prediction_horizon,
            split=split,
            channel_mean=channel_mean,
            channel_std=channel_std,
            delta_t=delta_t,
            use_image=use_image,
            min_speed=min_speed,
            num_image_frames=num_image_frames,
            image_stride=image_stride,
            image_policy=image_policy,
            pad_image_sequence=pad_image_sequence,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
        loaders.append(loader)

    return tuple(loaders)


# ---------------------------------------------------------------------------
# Multi-scenario dataset
# ---------------------------------------------------------------------------

class MultiScenarioDataset(Dataset):
    """
    Concatenation of multiple ChannelPredictionDataset instances for
    training across multiple scenarios.
    """

    def __init__(self, datasets: List[ChannelPredictionDataset]):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Find which sub-dataset this index belongs to
        for i, cs in enumerate(self.cumulative_sizes):
            if idx < cs:
                if i == 0:
                    return self.datasets[i][idx]
                return self.datasets[i][idx - self.cumulative_sizes[i - 1]]
        raise IndexError(f"Index {idx} out of range")


# ---------------------------------------------------------------------------
# Per-Subcarrier Multi-Step Channel Prediction Dataset
# ---------------------------------------------------------------------------

class PerSubcarrierDataset(Dataset):
    """
    Per-subcarrier channel prediction dataset with multi-step output.

    Each subcarrier is treated as an independent sample.
    Based on CPMamba configuration: 16 past frames -> 4 future frames.

    Input:  (history_len, num_antennas, 2)  — per subcarrier, real repr
    Target: (prediction_steps, num_antennas, 2) — per subcarrier, real repr

    Total samples = num_valid_windows * num_subcarriers

    Parameters
    ----------
    data_dir : str
        Path to a scenario directory (e.g., dataset_final/sc01/).
    history_len : int
        Number of past frames (default: 16).
    prediction_steps : int
        Number of future frames to predict (default: 4).
    split : str
        "train", "val", or "test".
    train_ratio : float
        Fraction for training (default: 0.7).
    val_ratio : float
        Fraction for validation (default: 0.15).
    min_speed : float
        Minimum UE speed in m/s to include a sample (default: 0.0, no filter).
    channel_subdir : str
        Subdirectory name for channel files (default: "ofdm").
    channel_prefix : str
        File prefix for channel files (default: "ofdm").
    num_subcarriers : int
        Number of subcarriers (default: 512). Set to None to auto-detect.
    """

    def __init__(
        self,
        data_dir: str,
        history_len: int = 16,
        prediction_steps: int = 4,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        min_speed: float = 0.0,
        channel_subdir: str = "ofdm",
        channel_prefix: str = "ofdm",
        num_subcarriers: Optional[int] = 512,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.history_len = history_len
        self.prediction_steps = prediction_steps
        self.split = split
        self.window_len = history_len + prediction_steps  # total frames needed

        # Discover channel files
        ch_dir = self.data_dir / channel_subdir
        if not ch_dir.exists():
            raise FileNotFoundError(f"Channel directory not found: {ch_dir}")

        self.channel_paths: Dict[int, str] = {}
        for f in sorted(ch_dir.glob(f"{channel_prefix}_*.npy")):
            idx = int(f.stem.split("_")[-1])
            self.channel_paths[idx] = str(f)

        if not self.channel_paths:
            raise FileNotFoundError(f"No channel files found in {ch_dir}")

        self.channel_indices = sorted(self.channel_paths.keys())
        available = set(self.channel_indices)

        # Auto-detect num_subcarriers from first file
        sample = np.load(self.channel_paths[self.channel_indices[0]])
        self.num_antennas = sample.shape[0]
        self.num_subcarriers = num_subcarriers or sample.shape[1]
        assert sample.shape[1] == self.num_subcarriers, (
            f"Expected {self.num_subcarriers} subcarriers, got {sample.shape[1]}"
        )

        # Load velocities for speed filtering
        speed_cache: Dict[int, float] = {}
        if min_speed > 0:
            vel_dir = self.data_dir / "velocities"
            if vel_dir.exists():
                for f in vel_dir.glob("velocities_*.npy"):
                    idx = int(f.stem.split("_")[-1])
                    vel = np.load(f)
                    speed_cache[idx] = float(np.linalg.norm(vel))

        # Build valid window start indices
        # Window [t, t+1, ..., t+window_len-1] needs all indices present
        min_idx = self.channel_indices[0]
        max_idx = self.channel_indices[-1]

        all_valid_starts: List[int] = []
        for t in self.channel_indices:
            end = t + self.window_len - 1
            if end > max_idx:
                break
            # Speed filter: check at the prediction boundary
            if speed_cache and min_speed > 0:
                speed = speed_cache.get(t + history_len - 1, 0)
                if speed < min_speed:
                    continue
            # Check contiguous block exists
            valid = all((t + k) in available for k in range(self.window_len))
            if valid:
                all_valid_starts.append(t)

        if not all_valid_starts:
            raise ValueError(
                f"No valid windows in {data_dir} with history={history_len}, "
                f"pred={prediction_steps}. Range: [{min_idx}, {max_idx}]"
            )

        # Temporal split (chronological)
        n = len(all_valid_starts)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        if split == "train":
            self.valid_starts = all_valid_starts[:n_train]
        elif split == "val":
            self.valid_starts = all_valid_starts[n_train:n_train + n_val]
        elif split == "test":
            self.valid_starts = all_valid_starts[n_train + n_val:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        self.n_windows = len(self.valid_starts)

    def __len__(self) -> int:
        return self.n_windows * self.num_subcarriers

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
            "input":  Tensor (history_len, num_antennas, 2) float32
            "target": Tensor (prediction_steps, num_antennas, 2) float32
            "sc_idx": int — subcarrier index
            "time_idx": int — window start time index
        """
        window_idx = idx // self.num_subcarriers
        sc_idx = idx % self.num_subcarriers

        t_start = self.valid_starts[window_idx]

        # Load history: [t_start, ..., t_start + history_len - 1]
        history = np.empty((self.history_len, self.num_antennas, 2), dtype=np.float32)
        for i in range(self.history_len):
            H = np.load(self.channel_paths[t_start + i])  # (Na, Nsc) complex
            h_sc = H[:, sc_idx]  # (Na,) complex
            history[i, :, 0] = h_sc.real.astype(np.float32)
            history[i, :, 1] = h_sc.imag.astype(np.float32)

        # Load target: [t_start + history_len, ..., t_start + window_len - 1]
        target = np.empty((self.prediction_steps, self.num_antennas, 2), dtype=np.float32)
        for i in range(self.prediction_steps):
            H = np.load(self.channel_paths[t_start + self.history_len + i])
            h_sc = H[:, sc_idx]
            target[i, :, 0] = h_sc.real.astype(np.float32)
            target[i, :, 1] = h_sc.imag.astype(np.float32)

        return {
            "input": torch.from_numpy(history),
            "target": torch.from_numpy(target),
            "sc_idx": sc_idx,
            "time_idx": t_start,
        }


class FullBandMultiStepDataset(Dataset):
    """
    Full-band (all subcarriers) multi-step channel prediction dataset.

    Same as PerSubcarrierDataset but keeps all subcarriers together.
    Suitable for models that can handle the full subcarrier dimension
    (e.g., Transformer, CHIRON).

    Input:  (history_len, num_antennas, num_subcarriers, 2)
    Target: (prediction_steps, num_antennas, num_subcarriers, 2)

    Multimodal keys (only present when use_image/use_lidar=True):
        "image_seq"        : Tensor (3, 224, 224) — latest-past RGB image
        "lidar_points"     : Tensor (lidar_max_points, 4) — padded point cloud
        "lidar_valid_mask" : Tensor (lidar_max_points,) bool — True for real pts
    """

    def __init__(
        self,
        data_dir: str,
        history_len: int = 16,
        prediction_steps: int = 4,
        split: str = "train",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        min_speed: float = 0.0,
        channel_subdir: str = "ofdm",
        channel_prefix: str = "ofdm",
        channel_min: Optional[np.ndarray] = None,
        channel_max: Optional[np.ndarray] = None,
        sc_indices: Optional[np.ndarray] = None,
        # Multimodal sensor inputs
        use_image: bool = False,
        use_lidar: bool = False,
        image_size: Tuple[int, int] = (224, 224),
        lidar_max_points: int = 64,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.history_len = history_len
        self.prediction_steps = prediction_steps
        self.window_len = history_len + prediction_steps
        self.channel_min = channel_min
        self.channel_max = channel_max
        self.sc_indices = sc_indices
        self.use_image = use_image
        self.use_lidar = use_lidar
        self.image_size = image_size
        self.lidar_max_points = lidar_max_points

        ch_dir = self.data_dir / channel_subdir
        if not ch_dir.exists():
            raise FileNotFoundError(f"Channel directory not found: {ch_dir}")

        self.channel_paths: Dict[int, str] = {}
        for f in sorted(ch_dir.glob(f"{channel_prefix}_*.npy")):
            idx = int(f.stem.split("_")[-1])
            self.channel_paths[idx] = str(f)

        if not self.channel_paths:
            raise FileNotFoundError(f"No channel files found in {ch_dir}")

        self.channel_indices = sorted(self.channel_paths.keys())
        available = set(self.channel_indices)

        sample = np.load(self.channel_paths[self.channel_indices[0]])
        self.num_antennas = sample.shape[0]
        self.num_subcarriers = sample.shape[1]

        # Discover image files (stride ~20, same as CARLA sensor rate)
        self.image_paths: Dict[int, str] = {}
        self.image_indices: List[int] = []
        if use_image:
            img_dir = self.data_dir / "images"
            if img_dir.exists():
                for f in sorted(img_dir.glob("frame_*.png")):
                    idx = int(f.stem.split("_")[-1])
                    self.image_paths[idx] = str(f)
                self.image_indices = sorted(self.image_paths.keys())

        # Discover lidar files (stride ~20, same as CARLA sensor rate)
        self.lidar_paths: Dict[int, str] = {}
        self.lidar_indices: List[int] = []
        if use_lidar:
            lid_dir = self.data_dir / "lidar"
            if lid_dir.exists():
                for f in sorted(lid_dir.glob("lidar_*.npy")):
                    idx = int(f.stem.split("_")[-1])
                    self.lidar_paths[idx] = str(f)
                self.lidar_indices = sorted(self.lidar_paths.keys())

        speed_cache: Dict[int, float] = {}
        if min_speed > 0:
            vel_dir = self.data_dir / "velocities"
            if vel_dir.exists():
                for f in vel_dir.glob("velocities_*.npy"):
                    idx = int(f.stem.split("_")[-1])
                    speed_cache[idx] = float(np.linalg.norm(np.load(f)))

        min_idx = self.channel_indices[0]
        max_idx = self.channel_indices[-1]

        all_valid_starts: List[int] = []
        for t in self.channel_indices:
            end = t + self.window_len - 1
            if end > max_idx:
                break
            if speed_cache and min_speed > 0:
                speed = speed_cache.get(t + history_len - 1, 0)
                if speed < min_speed:
                    continue
            valid = all((t + k) in available for k in range(self.window_len))
            if valid:
                all_valid_starts.append(t)

        if not all_valid_starts:
            raise ValueError(
                f"No valid windows in {data_dir} with history={history_len}, "
                f"pred={prediction_steps}."
            )

        n = len(all_valid_starts)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        if split == "train":
            self.valid_starts = all_valid_starts[:n_train]
        elif split == "val":
            train_last_end = all_valid_starts[n_train - 1] + self.window_len
            self.valid_starts = [s for s in all_valid_starts[n_train:n_train + n_val]
                                 if s >= train_last_end]
        elif split == "test":
            val_end_idx = n_train + n_val
            if val_end_idx > 0 and val_end_idx <= len(all_valid_starts):
                val_last_end = all_valid_starts[min(val_end_idx - 1, len(all_valid_starts) - 1)] + self.window_len
                self.valid_starts = [s for s in all_valid_starts[val_end_idx:]
                                     if s >= val_last_end]
            else:
                self.valid_starts = all_valid_starts[val_end_idx:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test'")

    def __len__(self) -> int:
        return len(self.valid_starts)

    def _load_latest_image(self, t: int) -> torch.Tensor:
        """Return ImageNet-normalized (3, H, W) tensor for the latest image at or before t."""
        if not self.image_indices:
            return torch.zeros(3, *self.image_size, dtype=torch.float32)
        pos = bisect_right(self.image_indices, t) - 1
        if pos < 0:
            return torch.zeros(3, *self.image_size, dtype=torch.float32)
        img_idx = self.image_indices[pos]
        try:
            img = _load_and_preprocess_image(self.image_paths[img_idx], self.image_size)
            img = _normalize_image(img)
            return torch.from_numpy(img)
        except (OSError, ValueError):
            return torch.zeros(3, *self.image_size, dtype=torch.float32)

    def _load_latest_lidar(self, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (lidar_max_points, 4) padded point cloud and (lidar_max_points,) valid mask."""
        max_p = self.lidar_max_points
        zero_pts = torch.zeros(max_p, 4, dtype=torch.float32)
        zero_mask = torch.zeros(max_p, dtype=torch.bool)
        if not self.lidar_indices:
            return zero_pts, zero_mask
        pos = bisect_right(self.lidar_indices, t) - 1
        if pos < 0:
            return zero_pts, zero_mask
        lid_idx = self.lidar_indices[pos]
        pts = np.load(self.lidar_paths[lid_idx]).astype(np.float32)  # (N, 4)
        N = pts.shape[0]
        padded = np.zeros((max_p, 4), dtype=np.float32)
        valid = np.zeros(max_p, dtype=bool)
        if N >= max_p:
            padded[:] = pts[:max_p]
            valid[:] = True
        else:
            padded[:N] = pts
            valid[:N] = True
        return torch.from_numpy(padded), torch.from_numpy(valid)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t_start = self.valid_starts[idx]

        history = np.empty(
            (self.history_len, self.num_antennas, self.num_subcarriers, 2),
            dtype=np.float32,
        )
        for i in range(self.history_len):
            H = np.load(self.channel_paths[t_start + i])
            history[i, :, :, 0] = H.real.astype(np.float32)
            history[i, :, :, 1] = H.imag.astype(np.float32)

        target = np.empty(
            (self.prediction_steps, self.num_antennas, self.num_subcarriers, 2),
            dtype=np.float32,
        )
        for i in range(self.prediction_steps):
            H = np.load(self.channel_paths[t_start + self.history_len + i])
            target[i, :, :, 0] = H.real.astype(np.float32)
            target[i, :, :, 1] = H.imag.astype(np.float32)

        # Subcarrier subsampling
        if self.sc_indices is not None:
            history = history[:, :, self.sc_indices, :]
            target = target[:, :, self.sc_indices, :]

        # Min-max normalization: (x - min) / (max - min) -> [0, 1]
        if self.channel_min is not None and self.channel_max is not None:
            ch_min = self.channel_min
            ch_max = self.channel_max
            if self.sc_indices is not None:
                ch_min = ch_min[:, self.sc_indices, :]
                ch_max = ch_max[:, self.sc_indices, :]
            denom = ch_max - ch_min
            denom[denom == 0] = 1.0
            for i in range(self.history_len):
                history[i] = (history[i] - ch_min) / denom
            for i in range(self.prediction_steps):
                target[i] = (target[i] - ch_min) / denom

        sample = {
            "input": torch.from_numpy(history),
            "target": torch.from_numpy(target),
            "time_idx": t_start,
        }

        # Sensor data — only included when requested (keeps backward compatibility)
        t_ref = t_start + self.history_len - 1   # last history frame = reference time
        if self.use_image:
            sample["image_seq"] = self._load_latest_image(t_ref)      # (3, H, W)
        if self.use_lidar:
            pts, mask = self._load_latest_lidar(t_ref)
            sample["lidar_points"] = pts                               # (max_pts, 4)
            sample["lidar_valid_mask"] = mask                          # (max_pts,) bool

        return sample
