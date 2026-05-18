#!/usr/bin/env python3
"""
Thin experiment entrypoint for 16->4 multimodal channel prediction.

This intentionally reuses multimodal_code_index/train_multimodal4.py so the
actual model implementations remain in the canonical multimodal_code_index tree.
"""

from __future__ import annotations

import runpy
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1,2")


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CODE_INDEX = REPO_ROOT / "multimodal_code_index"
TRAIN_SCRIPT = CODE_INDEX / "train_multimodal4.py"
OUTPUT_DIR = HERE / "outputs"


def _has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def _has_any_flag(args: list[str], flags: tuple[str, ...]) -> bool:
    return any(_has_flag(args, flag) for flag in flags)


def _get_flag_value(args: list[str], flag: str, default: str) -> str:
    for i, arg in enumerate(args):
        if arg == flag and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith(f"{flag}="):
            return arg.split("=", 1)[1]
    return default


def _add_default(args: list[str], flag: str, *values: str) -> None:
    if not _has_flag(args, flag):
        args.extend([flag, *values])


def _add_default_any(args: list[str], flags: tuple[str, ...], default_flag: str, *values: str) -> None:
    if not _has_any_flag(args, flags):
        args.extend([default_flag, *values])


def _bootstrap_imports() -> None:
    for path in (str(REPO_ROOT), str(CODE_INDEX)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(CODE_INDEX))


def main() -> None:
    args = list(sys.argv[1:])

    # Wrapper-only convenience flag. Underlying train_multimodal4.py defaults
    # to ImageNet weights; smoke/quick runs are safer without network downloads.
    use_pretrained_image = "--pretrained-image" in args
    args = [arg for arg in args if arg != "--pretrained-image"]

    _add_default(args, "--mode", "multimodal")
    _add_default(args, "--data-root", str(REPO_ROOT / "wireless-dataset"))
    if not _has_flag(args, "--scenarios") and not _has_flag(args, "--data-dir"):
        args.extend(["--scenarios", "sc01"])

    _add_default_any(args, ("--history-len", "-K"), "--history-len", "16")
    _add_default_any(args, ("--prediction-horizon", "-P"), "--prediction-horizon", "4")
    _add_default(args, "--num-subcarriers", "64")
    _add_default(args, "--num-image-frames", "8")

    _add_default(args, "--checkpoint-dir", str(OUTPUT_DIR / "checkpoints"))
    if not _has_flag(args, "--stats-file") and not _has_flag(args, "--no-normalize"):
        nsc = _get_flag_value(args, "--num-subcarriers", "64")
        args.extend(["--stats-file", str(OUTPUT_DIR / "stats" / f"channel_stats_nsc{nsc}.npz")])

    if not use_pretrained_image and not _has_flag(args, "--no-pretrained-image"):
        args.append("--no-pretrained-image")

    _bootstrap_imports()
    sys.argv = [str(TRAIN_SCRIPT), *args]
    runpy.run_path(str(TRAIN_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
