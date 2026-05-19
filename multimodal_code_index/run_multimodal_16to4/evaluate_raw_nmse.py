#!/usr/bin/env python3
"""Evaluate 16->4 checkpoints with denormalized raw-channel NMSE."""

from __future__ import annotations

import argparse
import json
import math
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


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
from utils import load_stats  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate trained 16->4 checkpoints on raw denormalized channels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=str(HERE / "outputs" / "checkpoints"),
        help="Directory used when --checkpoint is omitted.",
    )
    parser.add_argument(
        "--pattern",
        default="*_best.pt",
        help="Glob pattern under --checkpoint-dir.",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.checkpoint:
        return [Path(path) for path in args.checkpoint]
    return sorted(Path(args.checkpoint_dir).glob(args.pattern))


def infer_model_name(ckpt: dict, path: Path) -> str:
    if "model_name" in ckpt:
        return str(ckpt["model_name"])
    model_type = ckpt.get("model_config", {}).get("model_type", "")
    for name in ("lstm", "lwm_temporal", "lwm", "chiron"):
        if name in model_type:
            return name
    stem = path.stem
    for name in ("lwm_temporal", "chiron", "lstm", "lwm"):
        if stem.endswith(name) or f"_{name}_" in stem:
            return name
    raise ValueError(f"Could not infer model name for {path}")


def resolve_data_dir(args: Namespace) -> Path:
    if getattr(args, "data_dir", None):
        data_dir = args.data_dir
        if isinstance(data_dir, list):
            data_dir = data_dir[0]
        return Path(data_dir)
    data_root = Path(getattr(args, "data_root", REPO_ROOT / "wireless-dataset"))
    scenarios = getattr(args, "scenarios", ["sc01"])
    if isinstance(scenarios, str):
        scenarios = [scenarios]
    return data_root / scenarios[0]


def selected_subcarriers(args: Namespace) -> np.ndarray:
    return (
        getattr(args, "subcarrier_start", 0)
        + getattr(args, "subcarrier_stride", 1)
        * np.arange(getattr(args, "num_subcarriers", 64), dtype=np.int64)
    )


def load_normalization(args: Namespace) -> tuple[np.ndarray | None, np.ndarray | None]:
    if getattr(args, "no_normalize", False):
        return None, None
    stats_file = getattr(args, "stats_file", None)
    if not stats_file:
        data_dir = resolve_data_dir(args)
        stats_file = str(data_dir.parent / "channel_stats_train_only.npz")
    return load_stats(stats_file)


def make_dataset(
    train_args: Namespace,
    split: str,
    channel_min: np.ndarray | None,
    channel_max: np.ndarray | None,
) -> ChannelPredictionDataset:
    data_dir = resolve_data_dir(train_args)
    sensor_dir = None
    sensor_root = getattr(train_args, "sensor_root", None)
    mode = getattr(train_args, "mode", "channel_only")
    if mode != "channel_only" and sensor_root is not None:
        sensor_dir = str(Path(sensor_root) / data_dir.name)

    return ChannelPredictionDataset(
        data_dir=str(data_dir),
        sensor_data_dir=sensor_dir,
        sc_indices=selected_subcarriers(train_args),
        history_len=getattr(train_args, "history_len", 16),
        prediction_horizon=getattr(train_args, "prediction_horizon", 4),
        split=split,
        channel_mean=channel_min,
        channel_std=channel_max,
        delta_t=getattr(train_args, "delta_t", 0.0005),
        use_image=(mode != "channel_only"),
        num_image_frames=getattr(train_args, "num_image_frames", 8),
        image_stride=getattr(train_args, "image_stride", 1),
        image_policy="latest_past",
        pad_image_sequence=True,
        train_ratio=getattr(train_args, "train_ratio", 0.75),
        val_ratio=getattr(train_args, "val_ratio", 0.25),
    )


def ratio_to_db(ratio: float) -> float:
    return 10.0 * math.log10(max(ratio, 1e-12))


def denormalize(
    tensor: torch.Tensor,
    min_t: torch.Tensor | None,
    max_t: torch.Tensor | None,
) -> torch.Tensor:
    if min_t is None or max_t is None:
        return tensor
    return tensor * (max_t - min_t + 1e-8) + min_t


def accum_nmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    total: dict,
    prefix: str,
) -> None:
    batch = pred.shape[0]
    pred_flat = pred.reshape(batch, -1)
    target_flat = target.reshape(batch, -1)
    ratio = (
        (pred_flat - target_flat).pow(2).sum(dim=1)
        / target_flat.pow(2).sum(dim=1).clamp_min(1e-12)
    )
    total[f"{prefix}_ratio_sum"] += float(ratio.sum().item())

    per_step_ratio = (
        (pred - target).pow(2).flatten(2).sum(dim=2)
        / target.pow(2).flatten(2).sum(dim=2).clamp_min(1e-12)
    )
    total[f"{prefix}_per_step_sum"] += per_step_ratio.sum(dim=0).cpu().numpy()


@torch.no_grad()
def evaluate_checkpoint(path: Path, args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    train_args = Namespace(**ckpt.get("args", {}))
    model_name = infer_model_name(ckpt, path)
    train_args.no_pretrained_image = True

    channel_min, channel_max = load_normalization(train_args)
    dataset = make_dataset(train_args, args.split, channel_min, channel_max)
    if args.max_samples is not None and args.max_samples > 0 and len(dataset) > args.max_samples:
        dataset = Subset(dataset, range(args.max_samples))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(model_name, train_args)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    min_t = max_t = None
    if channel_min is not None:
        min_t = dataset.dataset._min_t if isinstance(dataset, Subset) else dataset._min_t
        max_t = dataset.dataset._max_t if isinstance(dataset, Subset) else dataset._max_t
        min_t = min_t.to(device)
        max_t = max_t.to(device)

    horizon = getattr(train_args, "prediction_horizon", 4)
    totals = {
        "model_ratio_sum": 0.0,
        "persistence_ratio_sum": 0.0,
        "linear_ratio_sum": 0.0,
        "model_per_step_sum": np.zeros(horizon, dtype=np.float64),
        "persistence_per_step_sum": np.zeros(horizon, dtype=np.float64),
        "linear_per_step_sum": np.zeros(horizon, dtype=np.float64),
    }
    num_samples = 0

    for batch in loader:
        channel_history = batch["channel_history"].to(device)
        target = batch["target"].to(device)
        image_seq = None
        image_valid_mask = None
        if getattr(train_args, "mode", "channel_only") != "channel_only":
            image_seq = batch["image_seq"].to(device)
            image_valid_mask = batch["image_valid_mask"].to(device)

        pred = model(
            channel_history=channel_history,
            image_seq=image_seq,
            image_valid_mask=image_valid_mask,
        )

        pred_raw = denormalize(pred, min_t, max_t)
        target_raw = denormalize(target, min_t, max_t)
        history_raw = denormalize(channel_history, min_t, max_t)

        last = history_raw[:, -1]
        prev = history_raw[:, -2]
        persistence = last.unsqueeze(1).expand_as(target_raw)
        steps = torch.arange(
            1,
            horizon + 1,
            dtype=target_raw.dtype,
            device=target_raw.device,
        ).view(1, horizon, 1, 1, 1)
        linear = last.unsqueeze(1) + steps * (last - prev).unsqueeze(1)

        accum_nmse(pred_raw, target_raw, totals, "model")
        accum_nmse(persistence, target_raw, totals, "persistence")
        accum_nmse(linear, target_raw, totals, "linear")
        num_samples += int(target.shape[0])

    def metric(prefix: str) -> tuple[float, list[float]]:
        avg_ratio = totals[f"{prefix}_ratio_sum"] / max(num_samples, 1)
        per_step = totals[f"{prefix}_per_step_sum"] / max(num_samples, 1)
        return ratio_to_db(avg_ratio), [ratio_to_db(float(value)) for value in per_step]

    model_nmse, model_per_step = metric("model")
    persistence_nmse, persistence_per_step = metric("persistence")
    linear_nmse, linear_per_step = metric("linear")

    return {
        "checkpoint": str(path),
        "model": model_name,
        "mode": getattr(train_args, "mode", None),
        "split": args.split,
        "num_samples": num_samples,
        "raw_nmse_db": model_nmse,
        "raw_nmse_per_step_db": model_per_step,
        "persistence_raw_nmse_db": persistence_nmse,
        "persistence_raw_nmse_per_step_db": persistence_per_step,
        "linear_raw_nmse_db": linear_nmse,
        "linear_raw_nmse_per_step_db": linear_per_step,
        "logged_best_val_nmse_db": ckpt.get("val_nmse_db"),
        "stats_file": getattr(train_args, "stats_file", None),
    }


def main() -> None:
    args = parse_args()
    results = []
    for path in checkpoint_paths(args):
        print(f"Evaluating {path}", flush=True)
        results.append(evaluate_checkpoint(path, args))
        row = results[-1]
        print(
            f"  {row['mode']:12s} {row['model']:12s} "
            f"raw_nmse={row['raw_nmse_db']:.3f} dB "
            f"persist={row['persistence_raw_nmse_db']:.3f} dB "
            f"linear={row['linear_raw_nmse_db']:.3f} dB "
            f"n={row['num_samples']}",
            flush=True,
        )

    payload = {"results": results}
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved: {out}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
