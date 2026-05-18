# Experiments

This document records how to run and report the active 16-to-4 channel
prediction experiments.

## Active Entrypoints

| File | Purpose |
|---|---|
| `multimodal_code_index/run_multimodal_16to4/run_16to4.py` | Main wrapper for current experiments |
| `multimodal_code_index/train_multimodal4.py` | Actual training loop |
| `multimodal_code_index/run_multimodal_16to4/smoke_16to4.py` | One-batch shape check |
| `multimodal_code_index/run_multimodal_16to4/launch_2gpu_by_model.sh` | Convenience launcher for model comparisons |

## Smoke Test

Run this before a long experiment:

```bash
cd /mnt/ssd_7t_2/carla-wireless-dataset
python multimodal_code_index/run_multimodal_16to4/smoke_16to4.py \
  --model lstm \
  --device cuda
```

Expected shapes:

```text
channel_history: (B, 16, 16, Nsc, 2)
target:          (B, 4, 16, Nsc, 2)
pred:            (B, 4, 16, Nsc, 2)
```

## Quick Training

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --model chiron \
  --epochs 5 \
  --batch-size 4 \
  --max-train-samples 1024 \
  --max-val-samples 256
```

## Channel-Only Baseline

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --mode channel_only \
  --model all \
  --epochs 30 \
  --batch-size 4
```

## Multimodal Run

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --mode multimodal \
  --model all \
  --epochs 30 \
  --batch-size 4 \
  --amp
```

## Full-Band Run

The wrapper defaults to `Nsc=64`. Use all generated subcarriers with:

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --mode multimodal \
  --model chiron \
  --num-subcarriers 512 \
  --epochs 30 \
  --batch-size 4
```

## Output Files

Default output root:

```text
multimodal_code_index/run_multimodal_16to4/outputs/
```

Common files:

| Path | Meaning |
|---|---|
| `outputs/checkpoints/*_best.pt` | best checkpoint by validation NMSE |
| `outputs/checkpoints/*_history.json` | per-epoch training history |
| `outputs/checkpoints/*_summary.json` | run summary |
| `outputs/logs/*.log` | stdout/stderr logs when launched through shell scripts |
| `outputs/stats/channel_stats_nsc*.npz` | channel normalization statistics |

## Metrics

The training loop reports:

- `train_loss`: MSE on normalized or raw real/imag channel tensors
- `train_nmse`: NMSE in dB
- `val_loss`: validation MSE
- `val_nmse`: validation NMSE in dB
- `val_cos`: cosine similarity

The saved best checkpoint stores:

```text
epoch
model_name
model_state_dict
optimizer_state_dict
val_nmse_db
val_cosine_sim
model_config
args
```

## Result Table Template

Use this table when reporting a comparison:

| Dataset | Mode | Scenario | Model | K | P | Nsc | Image frames | Best val NMSE | Checkpoint |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| `wireless-dataset` | `channel_only` | `sc01` | `lwm_temporal` | 16 | 4 | 64 | 0 | TBD | `..._best.pt` |
| `wireless-dataset` | `multimodal` | `sc01` | `chiron` | 16 | 4 | 64 | 8 | TBD | `..._best.pt` |

## Reproducibility Checklist

Record these values for every experiment:

- git commit hash
- dataset root and radio profile
- scenario list
- `K`, `P`, `Nsc`
- `channel_only` or `multimodal`
- image frame count and image stride
- model name
- epochs, batch size, LR, AMP flag
- stats file path
- checkpoint path
- best validation NMSE

## Current Caveats

- The active runner is single-process/single-device. `CUDA_VISIBLE_DEVICES=1,2`
  makes physical GPU 1 visible as `cuda:0` and physical GPU 2 visible as
  `cuda:1`, but a single run does not automatically use both GPUs.
- `--pretrained-image` can trigger torchvision weight loading. The wrapper uses
  `--no-pretrained-image` by default unless `--pretrained-image` is explicitly
  passed.
- LiDAR and Radar files are present in the dataset but are not consumed by the
  active 16-to-4 runner.
