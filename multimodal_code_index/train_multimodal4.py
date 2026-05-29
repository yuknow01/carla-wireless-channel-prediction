#!/usr/bin/env python3
"""
Train the four multimodal channel predictors on the current repo layout.

Models:
  - lstm          -> LSTMMultiModalPredictor
  - lwm           -> LWMMultiModalPredictor
  - lwm_temporal  -> LWMTemporalMultiModalPredictor
  - chiron        -> ChironMultiModalPredictor

This runner intentionally does not depend on the deleted experiment_dual_root
tree. It reads datasets through the repo-root dataset_loader.py and models
through multimodal_code_index/models, which is also exposed as repo-root models/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
for path in (str(REPO_ROOT), str(THIS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from dataset_loader import ChannelPredictionDataset, MultiScenarioDataset  # noqa: E402
from models.chiron_multimodal import ChironMultiModalPredictor  # noqa: E402
from models.lstm_multimodal import LSTMMultiModalPredictor  # noqa: E402
from models.lwm_multimodal import LWMMultiModalPredictor  # noqa: E402
from models.lwm_temporal_multimodal import LWMTemporalMultiModalPredictor  # noqa: E402
from utils import compute_dataset_stats, cosine_similarity, load_stats, nmse, save_stats  # noqa: E402


MODEL_CHOICES = ("lstm", "lwm", "lwm_temporal", "chiron")


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            scale = self.last_epoch / max(self.warmup_steps, 1)
            return [base_lr * scale for base_lr in self.base_lrs]

        progress = (self.last_epoch - self.warmup_steps) / max(
            self.total_steps - self.warmup_steps, 1
        )
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_scale
            for base_lr in self.base_lrs
        ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train LSTM/LWM/LWM-Temporal/Chiron multimodal predictors",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model", choices=MODEL_CHOICES + ("all",), default="all")
    p.add_argument("--mode", choices=("multimodal", "channel_only"), default="multimodal")

    p.add_argument("--data-root", default=str(REPO_ROOT / "wireless-dataset"))
    p.add_argument("--scenarios", nargs="+", default=["sc01"])
    p.add_argument("--data-dir", nargs="+", default=None,
                   help="Explicit scenario dirs. Overrides --data-root/--scenarios.")
    p.add_argument("--sensor-root", default=None,
                   help="Optional separate sensor root. Defaults to each data-dir itself.")
    p.add_argument("--stats-file", default=None)
    p.add_argument("--stats-max-samples", type=int, default=None,
                   help="Maximum files for normalization stats. Use 0 or omit for all files.")
    p.add_argument("--no-normalize", action="store_true")

    p.add_argument("-K", "--history-len", type=int, default=16)
    p.add_argument("-P", "--prediction-horizon", type=int, default=4)
    p.add_argument("--delta-t", type=float, default=0.0005)
    p.add_argument("--num-bs-antennas", type=int, default=16)
    p.add_argument("--num-subcarriers", type=int, default=64)
    p.add_argument("--subcarrier-start", type=int, default=0)
    p.add_argument("--subcarrier-stride", type=int, default=1)

    p.add_argument("--num-image-frames", type=int, default=8)
    p.add_argument("--image-stride", type=int, default=1)
    p.add_argument("--no-pretrained-image", action="store_true")

    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--fusion-layers", type=int, default=3)
    p.add_argument("--fusion-heads", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--chiron-depth", type=int, default=6)
    p.add_argument("--chiron-patch-w", type=int, default=32)
    p.add_argument("--lwm-temporal-depth", type=int, default=6)
    p.add_argument("--lwm-temporal-patch-w", type=int, default=16)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--max-train-samples", type=int, default=None)
    p.add_argument("--max-val-samples", type=int, default=None)
    p.add_argument("--no-shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.75)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp-dtype", choices=("fp16", "bf16"), default="fp16")

    p.add_argument("--checkpoint-dir", default=str(REPO_ROOT / "runs" / "checkpoints_multimodal4"))
    p.add_argument("--run-name", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Build datasets/models, print sizes, then exit before training.")
    return p.parse_args()


def selected_models(model_arg: str) -> Iterable[str]:
    return MODEL_CHOICES if model_arg == "all" else (model_arg,)


def resolve_data_dirs(args: argparse.Namespace) -> list[str]:
    if args.data_dir:
        return args.data_dir
    return [str(Path(args.data_root) / scenario) for scenario in args.scenarios]


def resolve_sensor_dirs(data_dirs: list[str], args: argparse.Namespace) -> list[str | None]:
    if args.mode == "channel_only":
        return [None] * len(data_dirs)
    if args.sensor_root is None:
        return [None] * len(data_dirs)
    return [
        str(Path(args.sensor_root) / Path(data_dir).name)
        for data_dir in data_dirs
    ]


def describe_visible_cuda_devices() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [
        f"cuda:{i}={torch.cuda.get_device_name(i)}"
        for i in range(torch.cuda.device_count())
    ]


def combine_datasets(parts: list):
    if not parts:
        raise ValueError("No dataset parts were created.")
    return MultiScenarioDataset(parts) if len(parts) > 1 else parts[0]


def maybe_cap_dataset(dataset, cap: int | None):
    if cap is not None and cap > 0 and len(dataset) > cap:
        return Subset(dataset, range(cap))
    return dataset


def build_model(model_name: str, args: argparse.Namespace) -> nn.Module:
    use_image = args.mode == "multimodal"
    pretrained_image = not args.no_pretrained_image

    common = dict(
        mode=args.mode,
        num_bs_antennas=args.num_bs_antennas,
        num_subcarriers=args.num_subcarriers,
        history_len=args.history_len,
        prediction_horizon=args.prediction_horizon,
        embed_dim=args.embed_dim,
        use_image=use_image,
        use_lidar=False,
        pretrained_image=pretrained_image,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        delta_t=args.delta_t,
    )

    if model_name == "lstm":
        return LSTMMultiModalPredictor(**common)

    if model_name == "lwm":
        return LWMMultiModalPredictor(**common)

    if model_name == "lwm_temporal":
        return LWMTemporalMultiModalPredictor(
            mode=args.mode,
            num_bs_antennas=args.num_bs_antennas,
            num_subcarriers=args.num_subcarriers,
            history_len=args.history_len,
            prediction_horizon=args.prediction_horizon,
            patch_w=args.lwm_temporal_patch_w,
            embed_dim=args.embed_dim,
            depth=args.lwm_temporal_depth,
            use_image=use_image,
            use_lidar=False,
            pretrained_image=pretrained_image,
            scene_attn_heads=args.fusion_heads,
        )

    if model_name == "chiron":
        return ChironMultiModalPredictor(
            mode=args.mode,
            num_bs_antennas=args.num_bs_antennas,
            num_subcarriers=args.num_subcarriers,
            history_len=args.history_len,
            prediction_horizon=args.prediction_horizon,
            patch_w=args.chiron_patch_w,
            embed_dim=args.embed_dim,
            depth=args.chiron_depth,
            max_image_frames=args.num_image_frames,
            use_image=use_image,
            use_lidar=False,
            use_ego_state=False,
            pretrained_image=pretrained_image,
            fusion_layers=args.fusion_layers,
            fusion_heads=args.fusion_heads,
            dropout=args.dropout,
        )

    raise ValueError(f"Unknown model: {model_name}")


def make_datasets(
    args: argparse.Namespace,
    data_dirs: list[str],
    sensor_dirs: list[str | None],
    sc_indices: np.ndarray,
    channel_min: np.ndarray | None,
    channel_max: np.ndarray | None,
) -> dict:
    datasets = {"train": [], "val": []}
    use_image = args.mode != "channel_only"

    for data_dir, sensor_dir in zip(data_dirs, sensor_dirs):
        for split in ("train", "val"):
            ds = ChannelPredictionDataset(
                data_dir=data_dir,
                sensor_data_dir=sensor_dir,
                sc_indices=sc_indices,
                history_len=args.history_len,
                prediction_horizon=args.prediction_horizon,
                split=split,
                channel_mean=channel_min,
                channel_std=channel_max,
                delta_t=args.delta_t,
                use_image=use_image,
                num_image_frames=args.num_image_frames,
                image_stride=args.image_stride,
                image_policy="latest_past",
                pad_image_sequence=True,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
            datasets[split].append(ds)

    train_ds = maybe_cap_dataset(combine_datasets(datasets["train"]), args.max_train_samples)
    val_ds = maybe_cap_dataset(combine_datasets(datasets["val"]), args.max_val_samples)
    return {"train": train_ds, "val": val_ds}


def make_loader(dataset, args: argparse.Namespace, *, train: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed + (0 if train else 1))
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train and not args.no_shuffle,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=train,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def move_batch(batch: dict, device: torch.device, args: argparse.Namespace) -> tuple:
    channel_history = batch["channel_history"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)

    image_seq = None
    image_time_offsets = None
    image_valid_mask = None
    if args.mode != "channel_only":
        image_seq = batch["image_seq"].to(device, non_blocking=True)
        image_time_offsets = batch["image_time_offsets"].to(device, non_blocking=True)
        image_valid_mask = batch["image_valid_mask"].to(device, non_blocking=True)

    return channel_history, target, image_seq, image_time_offsets, image_valid_mask


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    args: argparse.Namespace,
    *,
    optimizer: optim.Optimizer | None = None,
    scheduler=None,
    model_name: str = "",
    epoch: int = 0,
    scaler=None,
) -> dict:
    train = optimizer is not None
    model.train(train)
    total_loss = total_nmse = total_cos = 0.0
    batches = 0
    t0 = time.time()
    last_log_time = t0
    last_log_step = 0

    for step, batch in enumerate(loader, start=1):
        channel_history, target, image_seq, image_time_offsets, image_valid_mask = move_batch(
            batch, device, args
        )

        with torch.set_grad_enabled(train):
            amp_enabled = args.amp and device.type == "cuda"
            amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                pred = model(
                    channel_history=channel_history,
                    image_seq=image_seq,
                    image_time_offsets=image_time_offsets,
                    image_valid_mask=image_valid_mask,
                )
                loss = criterion(pred, target)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        with torch.no_grad():
            total_loss += float(loss.item())
            total_nmse += float(nmse(pred, target).item())
            total_cos += float(cosine_similarity(pred, target).item())
            batches += 1

        if train and args.log_every and step % args.log_every == 0:
            now = time.time()
            sec_per_step = (now - last_log_time) / max(step - last_log_step, 1)
            print(
                f"{model_name:12s} ep={epoch:03d} step={step:05d}/{len(loader):05d} "
                f"loss={total_loss / max(batches, 1):.6g} "
                f"nmse={total_nmse / max(batches, 1):.2f}dB "
                f"cos={total_cos / max(batches, 1):.4f} "
                f"step_time={sec_per_step:.3f}s",
                flush=True,
            )
            last_log_time = now
            last_log_step = step

    return {
        "loss": total_loss / max(batches, 1),
        "nmse_db": total_nmse / max(batches, 1),
        "cosine_sim": total_cos / max(batches, 1),
        "num_batches": batches,
    }


def train_one_model(
    model_name: str,
    args: argparse.Namespace,
    datasets: dict,
    device: torch.device,
) -> dict:
    model = build_model(model_name, args).to(device)
    raw_model = model
    num_params = raw_model.count_parameters()

    loaders = {
        "train": make_loader(datasets["train"], args, train=True),
        "val": make_loader(datasets["val"], args, train=False),
    }

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_steps=args.warmup_epochs * len(loaders["train"]),
        total_steps=args.epochs * len(loaders["train"]),
        min_lr=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    criterion = nn.MSELoss()

    run_prefix = args.run_name or (
        f"multimodal4_{args.mode}_{'_'.join(Path(d).name for d in resolve_data_dirs(args))}"
        f"_K{args.history_len}_P{args.prediction_horizon}_img{args.num_image_frames}"
    )
    if not run_prefix.endswith(model_name):
        run_prefix = f"{run_prefix}_{model_name}"

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"{run_prefix}_best.pt"
    hist_path = checkpoint_dir / f"{run_prefix}_history.json"

    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Mode: {args.mode}  params={num_params:,}")
    print(f"Train samples={len(datasets['train']):,}  val samples={len(datasets['val']):,}")
    print(f"Batch size={args.batch_size}  workers={args.num_workers}  AMP={args.amp and device.type == 'cuda'}")
    print(f"Checkpoint: {ckpt_path}")
    print("=" * 80)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            args,
            optimizer=optimizer,
            scheduler=scheduler,
            model_name=model_name,
            epoch=epoch,
            scaler=scaler,
        )
        with torch.no_grad():
            val_metrics = run_epoch(model, loaders["val"], criterion, device, args)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "time_s": elapsed,
            "lr": lr,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        print(
            f"{model_name:12s} ep={epoch:03d} "
            f"train_loss={train_metrics['loss']:.6g} train_nmse={train_metrics['nmse_db']:.2f}dB "
            f"val_loss={val_metrics['loss']:.6g} val_nmse={val_metrics['nmse_db']:.2f}dB "
            f"val_cos={val_metrics['cosine_sim']:.4f} lr={lr:.2e} time={elapsed:.1f}s",
            flush=True,
        )

        if val_metrics["nmse_db"] < best_val:
            best_val = val_metrics["nmse_db"]
            torch.save({
                "epoch": epoch,
                "model_name": model_name,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_nmse_db": best_val,
                "val_cosine_sim": val_metrics["cosine_sim"],
                "model_config": raw_model.get_config(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  saved best: {ckpt_path}", flush=True)

    summary = {
        "model_name": model_name,
        "num_parameters": num_params,
        "best_val_nmse_db": best_val,
        "history": history,
        "checkpoint": str(ckpt_path),
        "args": vars(args),
    }
    with hist_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"History saved: {hist_path}")
    return summary


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_dirs = resolve_data_dirs(args)
    sensor_dirs = resolve_sensor_dirs(data_dirs, args)
    sc_indices = (
        args.subcarrier_start
        + args.subcarrier_stride * np.arange(args.num_subcarriers, dtype=np.int64)
    )

    print("=" * 80)
    print("Multimodal 4-model training")
    print(f"Repo root: {REPO_ROOT}")
    print(f"Device: {device}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'UNSET')}")
    visible_cuda = describe_visible_cuda_devices()
    if visible_cuda:
        print(f"Visible CUDA devices: {', '.join(visible_cuda)}")
    print(f"Models: {', '.join(selected_models(args.model))}")
    print(f"Mode: {args.mode}")
    print(f"Seed: {args.seed}")
    print(f"K={args.history_len} P={args.prediction_horizon} Nsc={args.num_subcarriers}")
    print(f"Batch size={args.batch_size} workers={args.num_workers}")
    print(f"Image frames={args.num_image_frames} stride={args.image_stride}")
    for data_dir, sensor_dir in zip(data_dirs, sensor_dirs):
        print(f"  data={data_dir}" + (f" sensor={sensor_dir}" if sensor_dir else ""))
    print("=" * 80)

    channel_min = channel_max = None
    if not args.no_normalize:
        stats_path = args.stats_file
        if stats_path is None:
            stats_path = str(Path(data_dirs[0]).parent / "channel_stats_train_only.npz")

        if Path(stats_path).exists():
            print(f"Loading stats: {stats_path}")
            channel_min, channel_max = load_stats(stats_path)
        else:
            print(f"Computing stats from {data_dirs[0]}")
            channel_min, channel_max = compute_dataset_stats(
                data_dirs[0],
                num_bs_antennas=args.num_bs_antennas,
                num_subcarriers=args.num_subcarriers,
                max_samples=args.stats_max_samples,
                sc_indices=sc_indices,
            )
            Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
            save_stats(channel_min, channel_max, stats_path)
            print(f"Saved stats: {stats_path}")

    datasets = make_datasets(args, data_dirs, sensor_dirs, sc_indices, channel_min, channel_max)

    if args.dry_run:
        print("Dry run complete.")
        print(f"Train samples={len(datasets['train']):,}")
        print(f"Val samples={len(datasets['val']):,}")
        for model_name in selected_models(args.model):
            model = build_model(model_name, args)
            print(f"{model_name:12s} params={model.count_parameters():,}")
        return

    summaries = []
    for model_name in selected_models(args.model):
        summaries.append(train_one_model(model_name, args, datasets, device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_prefix = args.run_name or (
        f"multimodal4_{args.mode}_{'_'.join(Path(d).name for d in data_dirs)}"
        f"_K{args.history_len}_P{args.prediction_horizon}_img{args.num_image_frames}"
    )
    if args.model != "all" and not summary_prefix.endswith(args.model):
        summary_prefix = f"{summary_prefix}_{args.model}"
    out_path = Path(args.checkpoint_dir) / f"{summary_prefix}_summary.json"
    with out_path.open("w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Summary saved: {out_path}")


if __name__ == "__main__":
    main()
