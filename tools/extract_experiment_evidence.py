#!/usr/bin/env python3
"""Extract publication-page evidence from the saved CARLA experiment outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def result_row(path: Path, split: str, label: str) -> dict:
    data = load(path)
    block = data[split]
    model = block["model"]["all"]
    copy = block["copy_last"]["all"]
    return {
        "model": label,
        "nmse_db": model["nmse_db"],
        "median_db": model.get("median_db"),
        "per_step_db": model.get("per_step_db"),
        "copy_db": copy["nmse_db"],
        "copy_median_db": copy.get("median_db"),
        "gain_db": round(copy["nmse_db"] - model["nmse_db"], 2),
        "n": model.get("n"),
        "gate": data.get("gate"),
    }


def curve(path: Path, label: str, protocol: str) -> dict:
    rows = []
    for line in path.read_text().splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" in item:
            rows.append(item)
    return {
        "label": label,
        "protocol": protocol,
        "epochs": [r["epoch"] for r in rows],
        "train_loss": [r.get("loss") for r in rows],
        "val_loss": [r.get("val_loss") for r in rows],
        "val_nmse_db": [r.get("val_all_db") for r in rows],
        "val_near_db": [r.get("val_near_db") for r in rows],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment-root", type=Path,
                    default=Path("/mnt/ssd_7t_2/carla-wireless-dataset"))
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).with_name("visualization_experiment_data.json"))
    args = ap.parse_args()
    base = args.experiment_root / "scenario_pilot" / "channel_prediction"

    c1_map = {"LSTM": "lstm", "LWM": "lwm", "Mamba": "mamba", "Chiron": "chiron"}
    c2_map = {
        "LSTM": "lstm", "LWM": "lwm", "Mamba": "mamba",
        "Chiron": "chiron2", "DTCN": "dtcn_lr3e4",
    }
    c4_map = {
        "LSTM": "lstm_full_bscam", "LWM": "lwm_full_bscam",
        "DTCN": "dtcn_full_bscam", "EGRP-LWM": "egrp_full_bscam",
    }

    c1 = [result_row(base / "outputs_lit1ms" / sub / "result.json", "test", name)
          for name, sub in c1_map.items()]
    c2 = [result_row(base / "outputs_split62" / sub / "result.json", "val", name)
          for name, sub in c2_map.items()]
    c4 = [result_row(base / "outputs_mm_bscam" / sub / "result.json", "val", name)
          for name, sub in c4_map.items()]

    controls = {}
    for name, sub in c4_map.items():
        data = load(base / "outputs_mm_bscam" / sub / "result.json")
        controls[name] = {
            "original": data["val"]["model"]["all"]["nmse_db"],
            "gate": data.get("gate"),
        }
        for mod in ("radar", "camera", "lidar"):
            for ctrl in ("zero", "shuffle"):
                item = data[f"val_{mod}_{ctrl}"]
                if "model" in item:
                    item = item["model"]
                controls[name][f"{mod}_{ctrl}"] = item["all"]["nmse_db"]

    curves = {
        "c2_dtcn": curve(
            base / "outputs_split62" / "dtcn_lr3e4_curve_rerun_20260716" / "train.log",
            "DTCN · C2 curve rerun", "C2 · 1–4 ms · development validation"),
    }
    for key, sub, label in (
        ("c4_lstm", "lstm_full_bscam", "LSTM full"),
        ("c4_lwm", "lwm_full_bscam", "LWM full"),
        ("c4_dtcn", "dtcn_full_bscam", "DTCN full"),
        ("c4_egrp", "egrp_full_bscam", "EGRP-LWM full"),
    ):
        curves[key] = curve(
            base / "outputs_mm_bscam" / sub / "train.log", label,
            "C4 · BS-camera full fusion · 50–200 ms · development validation")

    out = {
        "source_root": str(args.experiment_root),
        "protocols": {
            "c1": "Train 5001–5006 · Val 5007 · Test 5008 · 20 epochs · 1–4 ms. Campaign-held-out test, but not a study-wide untouched frozen test.",
            "c2": "Train 5001,5003–5006,5008 · Val 5002,5007 · test field reuses validation · 40 epochs · 1–4 ms. Development validation only.",
            "c4": "Same 6/2 seed split as C2 · BS camera · 30 epochs · 50–200 ms. Development validation; no independent test.",
        },
        "c1": c1,
        "c2": c2,
        "c4": c4,
        "c4_controls": controls,
        "curves": curves,
    }
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
