# CARLA Wireless Channel Prediction

This repository contains a CARLA-to-Sionna data generation pipeline and channel
prediction models for multimodal wireless channel forecasting.

The current active experiment predicts `P=4` future channel frames from `K=16`
past channel frames and, in multimodal mode, the latest-past RGB image sequence.
Training commands and smoke checks are documented in [EXPERIMENTS.md](EXPERIMENTS.md).

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

The current `collect_final.py` default is the `fr1_3p5ghz` radio profile,
which means a 3.5 GHz FR1/sub-6 GHz channel setting:

| Item | Current value |
|---|---|
| Dataset root | `wireless-dataset/` |
| Carrier | `3.5 GHz` |
| Bandwidth | `50 MHz` |
| Raw subcarriers | `512` |
| Selected subcarriers | `64` |
| BS antennas | `16` |
| Channel interval | `0.5 ms` |
| Sensor interval | `1 ms` |
| Scenarios | `sc01` to `sc08` |

The 16-to-4 experiment defaults to `Nsc=64` by selecting the first 64
subcarriers from the raw 512-subcarrier channel files. Use `--num-subcarriers
512` when training on the full band.

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

## Important Source Files

Simulation and data generation:

- Pipeline: CARLA scenario collection -> Blender scene/material conversion -> Sionna ray tracing -> OFDM channel export.
- Detailed simulation parameters are documented in [SIM_SETTINGS.md](SIM_SETTINGS.md).
- `collect_final.py`: current full data generation pipeline
- `collect_1ms.py`: 1 ms sensor re-render helper
- `export_carla_geometry.py`: CARLA geometry export
- `blender_to_sionna.py`: Blender material conversion and Sionna scene export
- `channel_sim.py`: standalone geometric OFDM channel simulator
- `channel_sim_sionna.py`: Sionna-backed channel simulator with a compatible `ChannelSimulator` interface

Channel simulator progression:

- `channel_sim.py` was the first lightweight simulator. It generates `H(f)` directly from an analytic geometric model by assuming LOS/NLOS paths, path gains, delays, AoD, and Doppler. This made it possible to create channel-shaped data during CARLA collection without running Blender, Sionna, or GPU-heavy ray tracing.
- `channel_sim_sionna.py` was the next step. It keeps the same `ChannelSimulator.generate_channel(...)` interface, but replaces the analytic path assumptions with Sionna ray tracing in a simple street-canyon scene. This acted as a bridge between the fast geometric prototype and the final CARLA/Blender/Sionna pipeline.
- These two files are alternative channel-generation implementations, not a sequential data flow. `channel_sim_sionna.py` does not take the output of `channel_sim.py`; both produce `H(f)` through different modeling assumptions.
- The current dataset pipeline is `collect_final.py`. It calls `collect_data.py` with `--skip-geometric-channels`, then generates the final channel files with CARLA geometry, Blender scene conversion, and Sionna ray tracing. The final outputs include CIR, OFDM channel matrices, the legacy `channels/` alias, and angle-delay matrices.

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
