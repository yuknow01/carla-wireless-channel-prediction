# Dataset Specification

This document defines the dataset layout and the sample tensors used by the
active prediction code. The source of truth for training samples is
`dataset_loader.py`.

## Active Roots

| Root | Meaning |
|---|---|
| `wireless-dataset/` | Current default output of `collect_final.py` with `fr1_3p5ghz` |
| `dataset_final/` | Legacy `fr2_28ghz` dataset root |
| `dataset_1ms/` | 1 ms CARLA sensor re-render helper output |

Use one root consistently for a reported experiment. Do not mix profiles unless
the experiment explicitly says so.

## Scenario Layout

Each scenario directory follows this layout:

```text
wireless-dataset/
  sc01/
    channels/
      channel_000000.npy
      ...
    ofdm/
      ofdm_000000.npy
      ...
    cir/
      cir_000000.npz
      ...
    angle_delay/
      ad_000000.npy
      ...
    images/
      frame_000000.png
      ...
    lidar/
      lidar_000000.npy
      ...
    radar/
      radar_000000.npy
      ...
    positions/
      positions_000000.npy
      ...
    velocities/
      velocities_000000.npy
      ...
    vehicles_all/
      vehicles_000000.npy
      ...
    metadata.json
  sc02/
  ...
  sc08/
```

## Stored Modalities

| Modality | Directory | File type | Shape / content |
|---|---|---|---|
| Channel alias | `channels/` | `.npy` complex | `(16, 512)` in current generated data |
| OFDM channel | `ofdm/` | `.npy` complex | `(16, 512)` |
| CIR | `cir/` | `.npz` | `a`, `tau` path gains and delays |
| Angle-delay | `angle_delay/` | `.npy` complex | `(32, 32)` |
| RGB image | `images/` | `.png` | `1280 x 720` before loader resize |
| LiDAR | `lidar/` | `.npy` | point cloud values |
| Radar | `radar/` | `.npy` | radar detections |
| UE position | `positions/` | `.npy` | `(3,)` |
| UE velocity | `velocities/` | `.npy` | `(3,)` |
| Vehicle states | `vehicles_all/` | `.npy` | surrounding vehicle states |

The active runner currently loads only `channels/` and `images/`.

## Training Sample

`ChannelPredictionDataset` builds temporal windows from indexed channel files.
For a reference time `t`, a valid sample requires:

```text
history: [t-K+1, ..., t]
target:  [t+1, ..., t+P]
```

With the active 16-to-4 defaults:

| Field | Shape |
|---|---|
| `channel_history` | `(K=16, Na=16, Nsc, 2)` |
| `target` | `(P=4, Na=16, Nsc, 2)` |
| `image_seq` | `(T_img=8, 3, 224, 224)` |
| `image_valid_mask` | `(T_img,)` |
| `image_dt` | `(T_img,)` |
| `sample_index` | scalar |

The batch shapes are:

```text
channel_history: (B, 16, 16, Nsc, 2)
target:          (B, 4, 16, Nsc, 2)
image_seq:       (B, 8, 3, 224, 224)
prediction:      (B, 4, 16, Nsc, 2)
```

`2` stores real and imaginary parts after converting from the complex channel
array.

## Subcarrier Selection

Raw generated channels contain 512 subcarriers. The active experiment wrapper
defaults to:

```text
--num-subcarriers 64
--subcarrier-start 0
--subcarrier-stride 1
```

This means the dataset loader selects subcarrier indices:

```text
0, 1, 2, ..., 63
```

Use full-band training with:

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --num-subcarriers 512
```

## Image Policy

The active dataset loader uses the `latest_past` image policy:

- For each target reference time `t`, find the latest image index less than or
  equal to `t`.
- Load up to `num_image_frames` images using `image_stride`.
- Pad missing frames with zero images when `pad_image_sequence=True`.
- Return `image_valid_mask` so models can ignore padded image frames.
- Resize/normalize images for the ResNet18 image encoder.

## Splits

The active 16-to-4 runner uses chronological splits, not random splits:

| Split | Default ratio |
|---|---:|
| Train | `0.75` |
| Validation | `0.25` |
| Test | `0.0` in the current runner default |

`MultiScenarioDataset` concatenates scenario datasets when multiple scenarios
are passed.

## Normalization

`train_multimodal4.py` computes or loads channel normalization statistics:

- Default stats path for the wrapper:
  `multimodal_code_index/run_multimodal_16to4/outputs/stats/channel_stats_nsc{Nsc}.npz`
- `--no-normalize` disables this.
- `--stats-max-samples` can cap the number of files used for computing stats.

The same subcarrier selection is applied to statistics and samples.

## Current Loader Scope

Implemented and active:

- channel history
- future channel target
- latest-past RGB image sequence

Stored but not active in the current runner:

- LiDAR
- Radar
- explicit UE position/velocity
- `vehicles_all`

Those fields can be used by future models, but the active runner would need
batch loading and model-call changes before they affect training.
