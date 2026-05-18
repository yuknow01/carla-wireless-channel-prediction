# CARLA Wireless Channel Prediction

This repository contains a CARLA-to-Sionna data generation pipeline and channel
prediction models for multimodal wireless channel forecasting.

The current active experiment predicts `P=4` future channel frames from `K=16`
past channel frames and, in multimodal mode, the latest-past RGB image sequence.
The canonical training entrypoint is:

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py
```

## Project Map

| Document | Purpose |
|---|---|
| [SIM_SETTINGS.md](SIM_SETTINGS.md) | How the CARLA, Blender, and Sionna simulation is built |
| [DATASET_SPEC.md](DATASET_SPEC.md) | Dataset directories, file formats, sample construction, and tensor shapes |
| [MULTIMODAL_ARCHITECTURE.md](MULTIMODAL_ARCHITECTURE.md) | Model architecture for LSTM, LWM, LWM-Temporal, and Chiron |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Training commands, checkpoints, logs, and result tracking |
| [multimodal_code_index/run_multimodal_16to4/README.md](multimodal_code_index/run_multimodal_16to4/README.md) | Detailed notes for the current 16-to-4 experiment runner |

## Pipeline Overview

```text
CARLA Town10HD_Opt
  -> scenario setup: BS pose, vehicles, UE trajectory, RGB/LiDAR/Radar sensors
  -> static geometry export

Blender
  -> material assignment
  -> Mitsuba/Sionna scene export

Sionna ray tracing
  -> CIR path gains and delays
  -> OFDM channel H(f)
  -> angle-delay representation

PyTorch training
  -> channel-only or channel+RGB prediction
  -> LSTM / LWM / LWM-Temporal / Chiron comparison
```

## Active Dataset Convention

The current `collect_final.py` default is the `fr1_3p5ghz` radio profile:

| Item | Current value |
|---|---|
| Dataset root | `wireless-dataset/` |
| Carrier | `3.5 GHz` |
| Bandwidth | `50 MHz` |
| Raw subcarriers | `512` |
| BS antennas | `16` |
| Channel interval | `0.5 ms` |
| Sensor interval | `1 ms` |
| Scenarios | `sc01` to `sc08` |

The 16-to-4 experiment usually selects the first `64` subcarriers from the
raw 512-subcarrier channel files for Task04-compatible comparisons. Use
`--num-subcarriers 512` when training on the full band.

Legacy note: the older `dataset_final/` convention corresponds to the
`fr2_28ghz` profile and is still useful for comparison, but it should not be
mixed with `wireless-dataset/` results without explicitly saying so.

## Current Task

Default training task:

```text
Input:
  channel_history: (B, K=16, Na=16, Nsc, 2)
  image_seq:       (B, T_img=8, 3, 224, 224) in multimodal mode

Target:
  target:          (B, P=4, Na=16, Nsc, 2)

Output:
  prediction:      (B, P=4, Na=16, Nsc, 2)
```

`2` is the real/imaginary representation of the complex channel.

## Quick Start

Smoke check one model:

```bash
cd /mnt/ssd_7t_2/carla-wireless-dataset
python multimodal_code_index/run_multimodal_16to4/smoke_16to4.py \
  --model lstm \
  --device cuda
```

Short training run:

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --model chiron \
  --epochs 5 \
  --batch-size 4 \
  --max-train-samples 1024 \
  --max-val-samples 256
```

Run channel-only LWM-Temporal:

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --mode channel_only \
  --model lwm_temporal \
  --epochs 30 \
  --batch-size 4
```

Run all four multimodal models:

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --mode multimodal \
  --model all \
  --epochs 30 \
  --batch-size 4 \
  --amp
```

## Important Source Files

Simulation and data generation:

- `collect_final.py`: current full data generation pipeline
- `collect_1ms.py`: 1 ms sensor re-render helper
- `export_carla_geometry.py`: CARLA geometry export
- `blender_to_sionna.py`: Blender material conversion and Sionna scene export
- `channel_sim.py`, `channel_sim_sionna.py`: channel generation utilities

Dataset and training:

- `dataset_loader.py`: `ChannelPredictionDataset` and multi-scenario loader
- `multimodal_code_index/train_multimodal4.py`: actual training loop
- `multimodal_code_index/run_multimodal_16to4/run_16to4.py`: current wrapper with 16-to-4 defaults
- `multimodal_code_index/run_multimodal_16to4/smoke_16to4.py`: one-batch shape check

Models:

- `multimodal_code_index/models/lstm_multimodal.py`
- `multimodal_code_index/models/lwm_multimodal.py`
- `multimodal_code_index/models/lwm_temporal.py`
- `multimodal_code_index/models/lwm_temporal_multimodal.py`
- `multimodal_code_index/models/chiron_channel.py`
- `multimodal_code_index/models/chiron_multimodal.py`
- `multimodal_code_index/models/image_encoders.py`
- `multimodal_code_index/models/fusion_blocks.py`

## Output Locations

The 16-to-4 runner writes under:

```text
multimodal_code_index/run_multimodal_16to4/outputs/
  checkpoints/
  logs/
  stats/
```

Checkpoint, history, and summary filenames include mode, scenario, `K`, `P`,
image-frame count, and model name.

## Documentation Rule

When updating this repository, keep the documents tied to source files:

- Simulation claims should trace to `collect_final.py` or related collection scripts.
- Dataset shape claims should trace to `dataset_loader.py` and the active runner defaults.
- Model architecture claims should trace to files in `multimodal_code_index/models/`.
- Reported metrics should trace to logs or JSON summaries in the experiment output folder.
