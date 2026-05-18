#!/usr/bin/env python3
"""
One-batch shape smoke test for the 16->4 multimodal predictors.

The important assertion is that the model predicts all four future channel
frames at once: pred.shape == target.shape == (B, 4, 16, Nsc, 2).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CODE_INDEX = REPO_ROOT / "multimodal_code_index"

for path in (str(REPO_ROOT), str(CODE_INDEX)):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CODE_INDEX))

from dataset_loader import ChannelPredictionDataset  # noqa: E402
from train_multimodal4 import build_model  # noqa: E402
from utils import cosine_similarity, nmse  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Smoke-test one-shot 16->4 multimodal channel prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", choices=("lstm", "lwm", "lwm_temporal", "chiron"), default="lstm")
    p.add_argument("--mode", choices=("multimodal", "channel_only"), default="multimodal")
    p.add_argument("--data-root", default=str(REPO_ROOT / "wireless-dataset"))
    p.add_argument("--scenario", default="sc01")
    p.add_argument("--history-len", type=int, default=16)
    p.add_argument("--prediction-horizon", type=int, default=4)
    p.add_argument("--num-bs-antennas", type=int, default=16)
    p.add_argument("--num-subcarriers", type=int, default=64)
    p.add_argument("--subcarrier-start", type=int, default=0)
    p.add_argument("--subcarrier-stride", type=int, default=1)
    p.add_argument("--num-image-frames", type=int, default=8)
    p.add_argument("--image-stride", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default=None)
    p.add_argument("--pretrained-image", action="store_true")
    return p.parse_args()


def model_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        mode=args.mode,
        num_bs_antennas=args.num_bs_antennas,
        num_subcarriers=args.num_subcarriers,
        history_len=args.history_len,
        prediction_horizon=args.prediction_horizon,
        no_pretrained_image=not args.pretrained_image,
        embed_dim=256,
        fusion_layers=3,
        fusion_heads=4,
        dropout=0.1,
        chiron_depth=6,
        chiron_patch_w=32,
        lwm_temporal_depth=6,
        lwm_temporal_patch_w=16,
        num_image_frames=args.num_image_frames,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_image = args.mode != "channel_only"
    sc_indices = (
        args.subcarrier_start
        + args.subcarrier_stride * np.arange(args.num_subcarriers, dtype=np.int64)
    )

    data_dir = Path(args.data_root) / args.scenario
    dataset = ChannelPredictionDataset(
        data_dir=str(data_dir),
        sc_indices=sc_indices,
        history_len=args.history_len,
        prediction_horizon=args.prediction_horizon,
        split="train",
        delta_t=0.0005,
        use_image=use_image,
        num_image_frames=args.num_image_frames,
        image_stride=args.image_stride,
        image_policy="latest_past",
        pad_image_sequence=True,
        train_ratio=0.75,
        val_ratio=0.25,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    model = build_model(args.model, model_args(args)).to(device)
    model.eval()

    channel_history = batch["channel_history"].to(device)
    target = batch["target"].to(device)
    image_seq = batch["image_seq"].to(device) if use_image else None
    image_valid_mask = batch["image_valid_mask"].to(device) if use_image else None

    with torch.no_grad():
        pred = model(
            channel_history=channel_history,
            image_seq=image_seq,
            image_valid_mask=image_valid_mask,
        )
        loss = F.mse_loss(pred, target)

    expected = (args.batch_size, args.prediction_horizon, args.num_bs_antennas, args.num_subcarriers, 2)
    assert tuple(target.shape) == expected, f"target shape {tuple(target.shape)} != {expected}"
    assert tuple(pred.shape) == expected, f"pred shape {tuple(pred.shape)} != {expected}"

    report = {
        "status": "ok",
        "data_dir": str(data_dir),
        "model": args.model,
        "mode": args.mode,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "dataset_samples": len(dataset),
        "channel_history_shape": list(channel_history.shape),
        "image_seq_shape": list(image_seq.shape) if image_seq is not None else None,
        "target_shape": list(target.shape),
        "pred_shape": list(pred.shape),
        "mse": float(loss.item()),
        "nmse_db": float(nmse(pred, target).item()),
        "cosine_similarity": float(cosine_similarity(pred, target).item()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
