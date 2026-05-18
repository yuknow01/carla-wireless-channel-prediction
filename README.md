# CARLA-Wireless: Multimodal Wireless Channel Prediction Dataset

A complete data generation pipeline for multimodal wireless channel prediction research, combining **CARLA** simulator, **Blender** 3D modeling, and **NVIDIA Sionna** ray-tracing.

## Example Outputs

### Pipeline: CARLA Scene + Sionna Ray-Traced Channel
![Pipeline Output](assets/pipeline_output.png)
*Top-left: BS camera view from CARLA. Top-right: Ray-traced OFDM channel magnitude. Bottom-left: Beam pattern with peak AoD. Bottom-right: Angle-delay profile showing multipath structure.*

### Pipeline Steps: CARLA -> Blender -> Sionna
![Pipeline Steps](assets/pipeline_steps.png)
*Step 1: CARLA captures the urban scene. Step 2: Blender assigns ITU radio materials (concrete, brick, metal, marble). Step 3: Sionna computes ray-traced CIR with material-aware reflections.*

### 8 Scenario Camera Views
![Scenarios](assets/scenarios_overview.png)
*Diverse BS placements across CARLA Town10HD: straight roads, junctions, coastal areas, elevated rooftop.*

## Overview

This pipeline generates high-fidelity wireless channel datasets with synchronized multi-modal data:
- **Channel (CIR + OFDM)**: Sionna ray-traced channels with frequency-dependent ITU materials
- **RGB Images**: BS camera views of the road and vehicles
- **Position/Velocity**: UE trajectory data
- **Angle-Delay Matrices**: LWM-Temporal compatible format

Low-band rebuild convention:
- `collect_final.py` now defaults to `--radio-profile fr1_3p5ghz` (`3.5 GHz`, `50 MHz`, `512` subcarriers) and writes `wireless-dataset/`.
- To reproduce the old 28 GHz dataset, run with `--radio-profile fr2_28ghz`, which writes `dataset_final/`.
- A stricter cellular low-band preset is also available: `--radio-profile lowband_700mhz` (`700 MHz`, `10 MHz`).
- New low-band profiles save both `ofdm/` and the legacy-compatible `channels/` alias by default.

Legacy/current checked 28 GHz experiment convention:
- Use `dataset_final/` for the authoritative **0.5 ms / 2 kHz Sionna RT channel** (`ofdm/`, `channels/`, `cir/`, `angle_delay/`).
- Use `dataset_1ms/` for the latest **1 ms / 1 kHz CARLA sensors** (`images/`, `lidar/`, `radar`) aligned by the same sim-step index.
- Do not use `dataset_1ms/channels/` as the ray-traced channel unless Phase 3 has explicitly populated `ofdm/`, `cir/`, and `angle_delay/`; the current checked dataset contains Phase-1 geometric placeholder channels there.

### Pipeline Architecture

```
Phase 1: CARLA                       Phase 2/3: Blender + Sionna
────────────────────────────         ────────────────────────────────
Export CARLA Town10HD geometry  →    Load scene + ITU materials
Blender material assignment          Ray-trace at 2 kHz (max_depth=5)
  → Mitsuba XML export               Top-25 paths by received power
CARLA data collection:               No Doppler interpolation
  - RGB camera (1ms)                 OFDM channel synthesis (16×512)
  - LiDAR point clouds (1ms)         Angle-delay matrix (32×32)
  - RADAR point clouds (1ms)         CIR save every 0.5ms step
  - Positions/velocities (0.5ms)
  - Vehicle positions (0.5ms)

Latest sensor re-render:
  dataset_1ms/ stores RGB/LiDAR/Radar every 1ms
  and reuses dataset_final/ as the channel source for experiments.
```

### Key Specifications

| Parameter | Value |
|-----------|-------|
| Default Carrier Frequency | 3.5 GHz (`fr1_3p5ghz`) |
| Default Bandwidth | 50 MHz (512 x ~97.7 kHz) |
| Default Subcarrier Spacing | ~97.7 kHz (50 MHz / 512) |
| Legacy Profile | 28 GHz, 50 MHz via `--radio-profile fr2_28ghz` |
| TX Antenna | 1x16 ULA (BS) |
| RX Antenna | Single antenna (UE) |
| Ray-tracing | Sionna, max_depth=5 (LOS + up to 5th order reflection, deep NLOS) |
| Max Paths | top-25 by received power (aligned with DeepVerse) |
| Channel Sampling Rate | 2000 Hz (0.5ms channel interval) |
| Ray-trace Rate | 2000 Hz native; no Doppler interpolation in `dataset_final` |
| Default Sensor Rate | 1000 Hz (1ms) in `wireless-dataset` rebuild |
| Materials | ITU-R P.2040: concrete, brick, metal, marble, glass, ground |
| Vehicle Scattering | Metal reflectors updated per ray-trace frame |
| Map | CARLA Town10HD |

## Installation

### Prerequisites

- Ubuntu 20.04+
- NVIDIA GPU (RTX 3090/4090 recommended)
- Python 3.10+

### Setup

```bash
# 1. Clone this repo
git clone https://github.com/YOUR_USERNAME/carla-wireless-dataset.git
cd carla-wireless-dataset

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download and setup CARLA 0.9.15
bash setup_carla.sh

# 4. Download Blender 4.2 (headless)
bash setup_blender.sh

# 5. Install Sionna
pip install sionna
```

### Environment

```bash
# Required for Sionna on some systems
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
```

## Usage

### Quick Start

```bash
# 1. Start CARLA server (headless)
./run_carla.sh

# 2. Run full pipeline (CARLA -> Blender -> Sionna)
python collect_final.py
```

### Step-by-Step

```bash
# Phase 1a: Export CARLA geometry to OBJ
python export_carla_geometry.py --output sionna_scene_final/sc01 \
    --center-x 5 --center-y 38 --radius 80

# Phase 1b: Blender material assignment -> Mitsuba XML
./blender/blender --background --python blender_to_sionna.py -- \
    --input sionna_scene_final/sc01 --output sionna_scene_final/sc01_sionna \
    --scene-name sc01

# Phase 1c: CARLA data collection (positions, images, lidar, radar)
python collect_data.py --map Town10HD_Opt --steps 400000 --delta-t 0.0005 \
    --image-interval 20 --num-vehicles 40 \
    --bs-position 5 38 7 --bs-camera-pitch -15 --bs-camera-yaw 180 \
    --camera-fov 110 --scenario-id sc01 --seed 42 --output-dir dataset_final

# Phase 2/3: Sionna ray-tracing + OFDM synthesis (runs on GPU)
# Handled internally by collect_final.py → sionna_worker()
# To run manually for a single scenario:
CUDA_VISIBLE_DEVICES=2 python -c "
from collect_final import sionna_worker, SCENARIOS
sionna_worker([SCENARIOS[0]], 2)
"

# Latest 1ms sensor re-render (CARLA only; use dataset_final channels for training)
python collect_1ms.py --phase1-only --scenarios sc01,sc02 --output-dir dataset_1ms
```

## Dataset Format

```
dataset_final/
  sc01/
    ofdm/              # H(f): (16, 512) complex128, 0.5ms interval
    cir/               # {a: complex gains, tau: delays} per ray-trace frame (0.5ms)
    angle_delay/       # (32, 32) complex64, LWM-Temporal compatible, 0.5ms interval
    channels/          # H(f): (16, 512) complex128, same as ofdm/ (aliased)
    images/            # BS camera RGB, 1ms interval
    lidar/             # LiDAR point cloud (N, 4): [x, y, z, intensity], 1ms
    radar/             # RADAR point cloud (N, 4): [x, y, z, velocity], 1ms
    positions/         # UE [x, y, z], 0.5ms interval
    velocities/        # UE [vx, vy, vz], 0.5ms interval
    vehicles_all/      # All vehicle positions, 0.5ms interval
    metadata.json
  sc02/ ... sc08/

dataset_1ms/
  sc01/
    images/            # BS camera RGB, 1ms interval
    lidar/             # LiDAR point cloud, 1ms interval
    radar/             # RADAR point cloud, 1ms interval
    positions/         # UE [x, y, z], 0.5ms interval
    velocities/        # UE [vx, vy, vz], 0.5ms interval
    vehicles_all/      # All vehicle positions, 0.5ms interval
    channels/          # Phase-1 geometric placeholder unless Phase 3 is rerun
  sc02/ ... sc08/
```

### Channel Data Details

| Data | Shape | dtype | Rate |
|------|-------|-------|------|
| OFDM channel `dataset_final/ofdm/` | (16, 512) | complex128 | 2000 Hz (0.5ms) |
| CIR `dataset_final/cir/` | a: (16, ≤25), tau: (≤25,) | complex64 / float32 | 2000 Hz (0.5ms) |
| Angle-delay `dataset_final/angle_delay/` | (32, 32) | complex64 | 2000 Hz (0.5ms) |
| BS camera `dataset_1ms/images/` | 1280×720 RGB | uint8 | 1000 Hz (1ms) |
| LiDAR `dataset_1ms/lidar/` | (N, 4) | float32 | 1000 Hz (1ms) |
| RADAR `dataset_1ms/radar/` | (N, 4) | float32 | 1000 Hz (1ms) |
| Positions `positions/` | (3,) | float64 | 2000 Hz (0.5ms) |
| Velocities `velocities/` | (3,) | float64 | 2000 Hz (0.5ms) |

## Channel Prediction Models

```bash
# Channel-only prediction
python train.py --data-dir dataset_final/sc01 dataset_final/sc02 \
    --mode channel_only --epochs 30 -K 32 -P 1 \
    --num-bs-antennas 16 --num-subcarriers 512

# Multimodal prediction (Depth-pretrained image encoder)
python train.py --data-dir dataset_final/sc01 dataset_final/sc02 \
    --mode multimodal --image-encoder-type depth --epochs 30 \
    --num-bs-antennas 16 --num-subcarriers 512
```

## Scenarios

| # | Description | BS Position | Features |
|---|-------------|-------------|----------|
| sc01 | E-W street, building wall | (5, 38, 7) | Urban canyon |
| sc02 | E-W street, west section | (-75, 38, 7) | Different segment |
| sc03 | N-S road, building corner | (112, 35, 7) | Perpendicular road |
| sc04 | SW junction, building facade | (-50, -30, 7) | Intersection |
| sc05 | NW coastal, building wall | (-85, 35, 7) | Open environment |
| sc06 | Center junction, sidewalk | (-43, 8, 6) | Dense urban |
| sc07 | Southern boulevard | (5, -50, 7) | Wide road |
| sc08 | E-W street, rooftop BS | (5, 38, 12) | Height variation |

## Agent Execution Prompts

Copy-paste these prompts to have an AI coding agent (Claude Code, Cursor, Copilot, etc.) execute each pipeline stage.

### Prompt 1: Environment Setup

```
You are setting up the CARLA-Wireless dataset pipeline. Do the following in order:

1. Install Python dependencies: `pip install -r requirements.txt`
2. Install CARLA 0.9.15: `bash setup_carla.sh`
3. Install Blender 4.2: `bash setup_blender.sh`
4. Install Sionna: `pip install sionna`
5. Verify CARLA: launch `./run_carla.sh` in background, then run:
   `python -c "import carla; c=carla.Client('localhost',2000); c.set_timeout(30); print('OK:', c.get_server_version())"`
6. Verify Sionna: `python -c "import sionna; print('Sionna', sionna.__version__)"`
7. Verify Blender: `./blender/blender --version`

Report success/failure for each step.
```

### Prompt 2: Data Collection (CARLA + Blender + Sionna)

```
You are generating the CARLA-Wireless dataset. The environment is already set up.

1. Make sure CARLA server is running: `./run_carla.sh` (headless, background)
2. Run the full pipeline: `python collect_final.py`
   This executes:
   - Phase 1: CARLA sequential collection (8 scenarios)
     * Export CARLA Town10HD geometry as OBJ per scenario area
     * Blender assigns ITU radio materials and exports Mitsuba XML
     * CARLA collects positions, images, lidar, radar, velocities (400k steps/scenario)
   - Phase 2: Sionna ray-tracing on 3 GPUs parallel (GPU 1, 2, 3 round-robin)
3. Monitor progress by checking: `ls wireless-dataset/sc*/ofdm/ | wc -l` per scenario
4. When complete, verify each scenario has matching counts in ofdm/, images/, lidar/, positions/
5. Report total frames collected per scenario and overall dataset size.

Key parameters:
- 8 scenarios in CARLA Town10HD
- Default rebuild: 3.5 GHz, 1x16 ULA, 512 subcarriers (~97.7kHz spacing, 50MHz BW)
- Legacy 28 GHz mode: `python collect_final.py --radio-profile fr2_28ghz`
- max_depth=5 (LOS + up to 5th order reflection), top-25 paths by power
- Ray-trace at 2kHz native (0.5ms); no Doppler interpolation in the generated dataset
- Latest multimodal experiments pair this with 1ms sensors from `dataset_1ms`
- Stationary UE frames are automatically skipped
- BS is placed near building walls (6-8m height)

If Sionna crashes during Phase 2 (OOM), run scenarios sequentially:
`CUDA_VISIBLE_DEVICES=2 python -c "from collect_final import sionna_worker, SCENARIOS; sionna_worker([SCENARIOS[i]], 2)"` for each i.
```

### Prompt 3: Train Channel Prediction Model

```
You are training a channel prediction model on the CARLA-Wireless dataset.

Dataset is at: dataset_final/sc01 through dataset_final/sc08
Each scenario contains: ofdm/ (OFDM channels, shape 16x512), images/ (RGB), lidar/, radar/, positions/, velocities/

1. Train channel-only baseline (all 8 scenarios):
   python train.py \
     --data-dir dataset_final/sc01 dataset_final/sc02 dataset_final/sc03 \
                dataset_final/sc04 dataset_final/sc05 dataset_final/sc06 \
                dataset_final/sc07 dataset_final/sc08 \
     --mode channel_only --epochs 30 --batch-size 64 \
     -K 32 -P 1 --delta-t 0.002 \
     --num-bs-antennas 16 --num-subcarriers 512 \
     --device cuda:0 --run-name baseline_channel_only

2. Train multimodal model with depth-pretrained image encoder:
   python train.py \
     --data-dir dataset_final/sc01 dataset_final/sc02 dataset_final/sc03 \
                dataset_final/sc04 dataset_final/sc05 dataset_final/sc06 \
                dataset_final/sc07 dataset_final/sc08 \
     --mode multimodal --image-encoder-type depth \
     --epochs 30 --batch-size 32 \
     -K 32 -P 1 --delta-t 0.002 \
     --num-bs-antennas 16 --num-subcarriers 512 \
     --device cuda:1 --run-name multimodal_depth

3. Evaluate across prediction horizons:
   python evaluate.py --data-dir dataset_final/sc01 \
     --compare-modes --checkpoint-dir checkpoints/ \
     --horizons 1 5 10 20 50

4. Report: Test NMSE (dB), Cosine Similarity for each model.
   Compare channel_only vs multimodal_depth.

Model architecture:
- ChannelEncoder: Transformer (2 layers, 4 heads) over K=32 past channels
- ImageEncoder: MiDaS depth estimation backbone (frozen) + CNN feature extractor
- Fusion: Cross-attention (image queries, channel keys/values)
- PredictionHead: MLP -> predicted H(f) at t+P

Channels are (16, 512) complex, normalized per-element.
Stationary frames (speed < 0.5 m/s) are filtered in the dataloader.
```

### Prompt 4: Visualize Channels

```
You are creating channel visualizations for the CARLA-Wireless dataset.

1. Open and run: dataset_explorer.ipynb
   - Set SCENARIO = 'dataset_final/sc01'
   - This generates: channel snapshots, beam patterns, angle-delay profiles,
     temporal evolution plots

2. For scenario overview:
   - Open and run: visualize_carla_scenarios.ipynb
   - This shows BS camera views across all 8 scenarios in Town10HD

3. For the web visualization:
   - Preprocess data: generate web/data/channel_data.json from a scenario
   - Start server: cd web && python app.py --port 8888
   - Access at: http://localhost:8888/channel
   - Features: interactive timeline, top-down ray geometry, beam pattern, BS camera view

4. Key plots to generate:
   - Channel power vs time (shows fast fading at 0.5ms resolution)
   - Beam waterfall (AoD evolution over time) — 16-antenna BS
   - Angle-delay profile (32x32) at specific timesteps
   - Channel autocorrelation (coherence time estimation)
   - Side-by-side: BS camera image + channel heatmap + beam pattern
   - LiDAR/RADAR point clouds alongside channel response
```

### Prompt 5: Add New Scenario

```
You are adding a custom scenario to the CARLA-Wireless dataset.

1. Find a good BS location in CARLA Town10HD:
   python -c "
   import carla; c=carla.Client('localhost',2000); c.set_timeout(30)
   w=c.get_world(); spawns=w.get_map().get_spawn_points()
   import numpy as np
   for s in spawns[:20]: print(f'({s.location.x:.0f}, {s.location.y:.0f}, {s.location.z:.0f})')
   "

2. Test the camera view at your chosen BS position:
   - Place a camera, spawn vehicles, capture a test image
   - Verify: road is visible, vehicles are visible, BS height is realistic (6-8m)

3. Add to collect_final.py SCENARIOS list:
   {"name": "sc_new", "desc": "Your description",
    "carla_bs": [x, y, z], "pitch": -15, "yaw": angle, "seed": N}

4. Run geometry export + Blender for the new area:
   python export_carla_geometry.py --output sionna_scene_final/sc_new --center-x X --center-y Y --radius 80
   ./blender/blender --background --python blender_to_sionna.py -- --input sionna_scene_final/sc_new --output sionna_scene_final/sc_new_sionna --scene-name sc_new

5. Collect CARLA data:
   python collect_data.py --map Town10HD_Opt --steps 50000 --bs-position X Y Z --scenario-id sc_new --seed N --output-dir dataset_final

6. Run Sionna post-processing:
   CUDA_VISIBLE_DEVICES=0 python -c "from collect_final import sionna_worker, SCENARIOS; ..."
```

## References

- Mao et al., "Multimodal-Wireless: A Large-Scale Dataset for Sensing and Communication," arXiv:2511.03220
- Alikhani et al., "LWM-Temporal: Sparse Spatio-Temporal Attention for Wireless Channel Representation Learning," arXiv:2603.10024
- CARLA Simulator: https://carla.org
- NVIDIA Sionna: https://nvlabs.github.io/sionna/

## Dataset Access

The generated dataset (several GB) is available upon request.

## License

MIT License
