# CARLA Wireless Channel Prediction

This repository contains a CARLA-to-Sionna data generation pipeline and channel
prediction models for multimodal wireless channel forecasting.

The current active experiment predicts `P=4` future channel frames from `K=16`
past channel frames and, in multimodal mode, a time-aligned latest-past RGB
image sequence.
Training commands and smoke checks are archived in [docs_archive/EXPERIMENTS.md](docs_archive/EXPERIMENTS.md).

## Project Map

| Document | Purpose |
|---|---|
| [scenario_pilot/channel_prediction/EXPERIMENT_PLAN.md](scenario_pilot/channel_prediction/EXPERIMENT_PLAN.md) | **현행 마스터 실험 명세** (launch 승인 게이트) |
| [scenario_pilot/channel_prediction/FUSION_ARCHITECTURES.md](scenario_pilot/channel_prediction/FUSION_ARCHITECTURES.md) | **현행 융합 3계열 설계 + 코드 지도** |
| [reports/experiments_full_report.md](reports/experiments_full_report.md) | **현행 결과 종합** (TL;DR + 전 수치) |
| [reports/related_work_critique_and_positioning.md](reports/related_work_critique_and_positioning.md) | 타 논문 비판 + 논문 포지셔닝 |
| [SIM_SETTINGS.md](SIM_SETTINGS.md) | How the CARLA, Blender, and Sionna simulation is built |
| [DATASET_SPEC.md](DATASET_SPEC.md) | Dataset directories, file formats, sample construction, and tensor shapes |
| [docs_archive/](docs_archive/README.md) | 통합/대체된 구 문서 보관 (구 MULTIMODAL_ARCHITECTURE, EXPERIMENTS, EXPERIMENT_RESULTS 등) |
| [multimodal_code_index/run_multimodal_16to4/README.md](multimodal_code_index/run_multimodal_16to4/README.md) | (레거시) 16-to-4 experiment runner 노트 |

## Interactive Visualizations

- [Visualization home](https://yuknow01.github.io/carla-wireless-channel-prediction/)
- [LSTM recurrent channel predictor](https://yuknow01.github.io/carla-wireless-channel-prediction/visualizations/lstm.html)
- [LSTM embedding detail: 2,048 to 256](https://yuknow01.github.io/carla-wireless-channel-prediction/visualizations/lstm-2048-to-256.html)
- [LWM wideband Transformer](https://yuknow01.github.io/carla-wireless-channel-prediction/visualizations/lwm.html)
- [Mamba selective state-space backbone](https://yuknow01.github.io/carla-wireless-channel-prediction/visualizations/mamba.html)
- [DTCN frequency/delay causal TCN](https://yuknow01.github.io/carla-wireless-channel-prediction/visualizations/dtcn.html)
- [Chiron factorized spatio-temporal model](https://yuknow01.github.io/carla-wireless-channel-prediction/visualizations/chiron.html)
- [Multimodal EGRP and MLLM-B (GatedFusion baseline included)](https://yuknow01.github.io/carla-wireless-channel-prediction/visualizations/multimodal-fusion.html)
- [Channel phase and multipath interference](https://yuknow01.github.io/carla-wireless-channel-prediction/visualizations/channel-phase-interference.html)

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

Current LSTM/LWM note:

- `lstm` and `lwm` use wideband time tokens, not per-subcarrier tokens:
  `(B, K, Na, Nsc, 2) -> (B, K, D)`.
- In multimodal mode, RGB frames are summarized per frame, aligned to the
  channel-history time grid with `image_time_offsets`, and fused per time step.
- `lwm_temporal` and `chiron` keep their existing wideband/patch-time designs.

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
