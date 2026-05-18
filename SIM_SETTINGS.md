# Simulation Settings

This document describes how the synthetic wireless dataset is generated. The
current source of truth is `collect_final.py`; older dataset roots remain in the
repository for comparison, but the default runtime profile is now
`fr1_3p5ghz`.

## Source Files

| File | Role |
|---|---|
| `collect_final.py` | Full CARLA -> Blender -> Sionna pipeline |
| `run_collect_final.sh` | Launch helper for full collection |
| `collect_1ms.py` | 1 ms sensor re-render helper |
| `export_carla_geometry.py` | Static CARLA geometry export |
| `blender_to_sionna.py` | Blender material conversion and Sionna scene export |
| `channel_sim.py`, `channel_sim_sionna.py` | Channel generation utilities |

## Pipeline

```text
Phase 1: CARLA
  - Start one CARLA worker per configured GPU/port.
  - Load Town10HD_Opt.
  - Spawn traffic and select a UE vehicle.
  - Save UE state, all-vehicle state, RGB, LiDAR, and Radar.
  - Export static scene geometry.

Phase 2: Blender
  - Convert exported geometry.
  - Assign radio-relevant materials.
  - Export a Sionna-compatible scene.

Phase 3: Sionna
  - Load the scene.
  - Add/update dynamic vehicle scatterers.
  - Ray-trace each simulation step.
  - Save CIR, OFDM channel, and angle-delay matrices.
```

## Runtime Defaults

The defaults below are taken from `collect_final.py`.

| Setting | Value |
|---|---|
| Map | `Town10HD_Opt` |
| Scenarios | `sc01` to `sc08` |
| Simulation steps | `400000` per scenario |
| Simulation interval | `0.0005 s` |
| Scenario duration | `200 s` |
| Sensor save interval | every `2` simulation steps = `1 ms` |
| Ray-trace interval | every `1` simulation step = `0.5 ms` |
| Vehicles spawned by current final pipeline | `20` |
| BS antennas | `16` |
| UE antennas | `1` |
| Max ray depth | `5` |
| Max saved paths | top `25` by received power |
| Ray samples per source | `1e6` |
| Scattering coefficient | `0.3` |

`collect_1ms.py` has its own default vehicle count of `40`; do not assume that
helper and `collect_final.py` use the same traffic count.

## Radio Profiles

`collect_final.py` selects the radio profile from `CW_RADIO_PROFILE`. If the
environment variable is not set, it uses `fr1_3p5ghz`.

| Profile | Dataset root | Carrier | Bandwidth | Subcarriers | Notes |
|---|---:|---:|---:|---:|---|
| `fr1_3p5ghz` | `wireless-dataset/` | `3.5 GHz` | `50 MHz` | `512` | Current default |
| `fr2_28ghz` | `dataset_final/` | `28 GHz` | `50 MHz` | `512` | Legacy mmWave reference |
| `lowband_700mhz` | `dataset_lowband_700mhz/` | `700 MHz` | `10 MHz` | `512` | Low-band profile |

The code also writes the legacy-compatible `channels/` alias when
`save_channel_alias` is enabled for the profile.

Example:

```bash
CW_RADIO_PROFILE=fr1_3p5ghz python collect_final.py
CW_RADIO_PROFILE=fr2_28ghz python collect_final.py
```

## Scenario Table

| ID | Description | BS position `(x, y, z)` | Camera pitch/yaw | Seed |
|---|---|---:|---:|---:|
| `sc01` | E-W street, BS on building wall east | `(5, 38, 7)` | `-15 / 180` | `42` |
| `sc02` | E-W street, BS on building wall west | `(-75, 38, 7)` | `-15 / 180` | `101` |
| `sc03` | N-S road, BS on building corner east | `(112, 35, 7)` | `-15 / -90` | `202` |
| `sc04` | SW junction, BS on building facade | `(-50, -30, 7)` | `-18 / 45` | `303` |
| `sc05` | NW coastal, BS on building wall | `(-85, 35, 7)` | `-15 / 135` | `404` |
| `sc06` | Center junction, BS on sidewalk pole | `(-43, 8, 6)` | `-15 / 90` | `505` |
| `sc07` | Southern boulevard, BS on building wall | `(5, -50, 7)` | `-15 / 0` | `606` |
| `sc08` | E-W street, elevated rooftop BS | `(5, 38, 12)` | `-22 / 180` | `707` |

## Sensors

All active sensors are mounted from the BS viewpoint.

| Sensor | Main settings | Save interval |
|---|---|---|
| RGB camera | `1280 x 720`, FOV `110 deg` | `1 ms` in current final pipeline |
| LiDAR | 32 channels, 100 m range, 100k points/s, 10 Hz rotation | `1 ms` |
| Radar | 60 deg horizontal FOV, 10 deg vertical FOV, 100 m range, 1500 points/s | `1 ms` |

The current 16-to-4 training runner uses channel history and RGB image frames.
LiDAR and Radar are collected but are not passed by the active runner.

## Channel Generation

For each ray-trace frame, Sionna saves:

| Output | Directory | Shape |
|---|---|---|
| CIR path gains/delays | `cir/` | `a: (16, P_path)`, `tau: (P_path,)` |
| OFDM channel | `ofdm/` | `(16, 512)` complex |
| Legacy channel alias | `channels/` | `(16, 512)` complex |
| Angle-delay matrix | `angle_delay/` | `(32, 32)` complex |

`P_path` is capped at `25` after sorting paths by received power.

The OFDM frequency grid is centered around the carrier:

```text
freq[k] = (k - NUM_SC / 2) * (BANDWIDTH / NUM_SC)
```

## Coordinate Convention

CARLA and Sionna use different coordinate conventions. The active conversion in
`collect_final.py` maps CARLA `(x, y, z)` to Sionna `(x, -y, z)`. The UE height
is fixed to `1.5 m` for Sionna receiver placement.

## Known Limitations

- Vehicle scatterers are simplified mesh boxes, not detailed car geometry.
- The active 16-to-4 runner does not consume LiDAR or Radar even though the
  dataset includes them.
- Results from `wireless-dataset/` and `dataset_final/` should be reported as
  separate dataset/profile settings.
