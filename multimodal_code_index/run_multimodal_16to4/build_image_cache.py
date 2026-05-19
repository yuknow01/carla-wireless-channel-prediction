#!/usr/bin/env python3
"""Build a uint8 memmap image cache for multimodal training."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
from time import time

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize frame_*.png images once and store them as a uint8 .npy cache.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def image_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def discover_images(image_dir: Path, limit: int | None) -> list[tuple[int, Path]]:
    images = sorted(
        ((image_index(path), path) for path in image_dir.glob("frame_*.png")),
        key=lambda item: item[0],
    )
    if limit is not None:
        images = images[:limit]
    return images


def load_resized(task: tuple[int, int, str, int]) -> tuple[int, int, np.ndarray, str]:
    row, idx, path, size = task
    try:
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(path) as img:
            img = img.convert("RGB").resize((size, size), Image.BILINEAR)
            arr = np.asarray(img, dtype=np.uint8)
        return row, idx, arr, ""
    except Exception as exc:  # noqa: BLE001
        arr = np.zeros((size, size, 3), dtype=np.uint8)
        return row, idx, arr, f"{path}: {exc}"


def remove_stale_temp(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            path.unlink()


def main() -> None:
    args = parse_args()
    scenario_dir = Path(args.scenario_dir).resolve()
    image_dir = Path(args.image_dir).resolve() if args.image_dir else scenario_dir / "images"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else scenario_dir / "image_cache"
    output_dir.mkdir(parents=True, exist_ok=True)

    images_path = output_dir / f"images_{args.size}_uint8.npy"
    indices_path = output_dir / "image_indices.npy"
    metadata_path = output_dir / "metadata.json"

    if images_path.exists() and indices_path.exists() and not args.force:
        print(f"Image cache already exists: {output_dir}", flush=True)
        return

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    images = discover_images(image_dir, args.limit)
    if not images:
        raise FileNotFoundError(f"No frame_*.png files found in {image_dir}")

    temp_images_path = output_dir / f".{images_path.name}.partial"
    temp_indices_path = output_dir / f".{indices_path.name}.partial"
    temp_metadata_path = output_dir / f".{metadata_path.name}.partial"
    remove_stale_temp([temp_images_path, temp_indices_path, temp_metadata_path])

    count = len(images)
    print(
        f"Building image cache: count={count} size={args.size} workers={args.workers}",
        flush=True,
    )
    print(f"source={image_dir}", flush=True)
    print(f"output={output_dir}", flush=True)

    indices = np.asarray([idx for idx, _ in images], dtype=np.int64)
    with temp_indices_path.open("wb") as f:
        np.save(f, indices)

    cache = np.lib.format.open_memmap(
        temp_images_path,
        mode="w+",
        dtype=np.uint8,
        shape=(count, args.size, args.size, 3),
    )

    tasks = [
        (row, idx, str(path), args.size)
        for row, (idx, path) in enumerate(images)
    ]
    failures: list[str] = []
    t0 = time()

    if args.workers <= 1:
        iterator = map(load_resized, tasks)
        for done, (row, _idx, arr, error) in enumerate(iterator, start=1):
            cache[row] = arr
            if error:
                failures.append(error)
            if done % args.log_every == 0 or done == count:
                elapsed = time() - t0
                rate = done / max(elapsed, 1e-6)
                print(f"cached {done}/{count} images ({rate:.1f} img/s)", flush=True)
                cache.flush()
    else:
        with mp.Pool(processes=args.workers) as pool:
            iterator = pool.imap_unordered(load_resized, tasks, chunksize=16)
            for done, (row, _idx, arr, error) in enumerate(iterator, start=1):
                cache[row] = arr
                if error:
                    failures.append(error)
                if done % args.log_every == 0 or done == count:
                    elapsed = time() - t0
                    rate = done / max(elapsed, 1e-6)
                    print(f"cached {done}/{count} images ({rate:.1f} img/s)", flush=True)
                    cache.flush()

    cache.flush()
    del cache

    metadata = {
        "completed": True,
        "source_image_dir": str(image_dir),
        "image_count": count,
        "size": [args.size, args.size],
        "dtype": "uint8",
        "layout": "NHWC",
        "images_file": images_path.name,
        "indices_file": indices_path.name,
        "failed_count": len(failures),
        "failed_examples": failures[:100],
    }
    with temp_metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    os.replace(temp_images_path, images_path)
    os.replace(temp_indices_path, indices_path)
    os.replace(temp_metadata_path, metadata_path)
    print(f"Image cache complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
