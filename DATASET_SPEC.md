# CARLA-Wireless Dataset — 명세서

> **버전**: v1.0 (Δt = 0.5 ms, Nyquist-compliant urban)
> **생성일**: 2026-04-09
> **파이프라인**: CARLA → Blender → Sionna
> **저장 경로**: `/mnt/ssd_7t_2/carla-wireless-dataset/dataset_final/`, `/mnt/ssd_7t_2/carla-wireless-dataset/dataset_1ms/`

---

## 1. 개요

도시 교차로 시나리오에서 **28 GHz 밀리미터파 채널**과 **멀티모달 센서 데이터**(카메라/LiDAR/Radar/차량 위치)를 동기화하여 수집한 합성 데이터셋. Ray-tracing 기반 채널 생성과 CARLA 자율주행 시뮬레이션을 결합했다. 시간축 Nyquist 조건을 28 GHz · 38 km/h 기준으로 만족한다 (`Δt = 0.5 ms`, `fs = 2 kHz`).

- **시나리오**: 8 개 (동일한 `Town10HD_Opt` 맵 위의 서로 다른 BS 배치 · 관찰 지점)
- **시나리오당 시뮬레이션 길이**: 200 초 (400,000 sim step × 0.5 ms)
- **시나리오당 채널 샘플**: 400,000 (`dataset_final`, 0.5 ms / 2 kHz Sionna OFDM H matrix)
- **시나리오당 센서 프레임**: 20,000 (`dataset_final`, 10 ms) 또는 200,000 (`dataset_1ms`, 1 ms)
- **최신 실험 기준**: `dataset_final`의 0.5 ms Sionna 채널 + `dataset_1ms`의 1 ms 센서 프레임을 sim-step index로 결합
- **총 raw 데이터 크기**: `dataset_final` 약 **1.1 TB**, `dataset_1ms` 약 **2.4 TB**

---

## 2. 파이프라인 구조

```
┌──────────────┐    OBJ     ┌──────────────┐   PLY+XML    ┌──────────────┐
│  ① CARLA     │ ────────▶ │  ② Blender   │ ──────────▶ │  ③ Sionna    │
│ 정적 geometry│            │  포맷 변환   │              │ Ray-tracing  │
│ 추출         │            │  ITU 재질    │              │ (GPU, 3 병렬)│
└──────────────┘            └──────────────┘              └──────────────┘
       │                                                           ▲
       │                                                           │
       │  ④ CARLA 동적 시뮬레이션 (2 kHz, 200 s)                   │
       │  40 대 차량 자율주행 / UE 이동 / BS 센서 수집             │
       │  positions, velocities, images, lidar, radar, vehicles_all │
       └────────────────────────────────────────────────────────────┘
```

- **Phase 1** = ① + ② + ④ — CARLA 쪽에서 실행
- **Phase 2** = ③ — Sionna ray-tracing, host Python 에서 실행 (3 GPU 병렬)

---

## 3. 환경

| 항목 | 값 |
|---|---|
| GPU | 3 × NVIDIA RTX 6000 Ada Generation (각 49 GiB VRAM) |
| NVIDIA Driver | 570.195.03 |
| CUDA API 지원 | 12.8 (runtime 라이브러리는 pip `nvidia-cudnn-cu12 9.20`, `nvidia-cublas-cu12 12.9` 로 제공) |
| CARLA | 0.9.15 (Docker: `carlasim/carla:0.9.15`) |
| UE4 | 4.26.2 |
| Quality level | **Epic** |
| Map | `Town10HD_Opt` |
| Python | 3.10.20 (conda env `carla-wireless`) |
| TensorFlow | 2.21.0 (GPU 빌드) |
| Sionna | 1.2.2 (`sionna.rt`) |
| NumPy | 2.2.6 |
| 저장 파티션 | `/data` (md0 RAID5, 60 TB) |

### GPU 할당 (Phase 1 · Phase 2 공통)

| GPU | Phase 1 역할 | Phase 2 역할 |
|---|---|---|
| 1 | CARLA 컨테이너 `carla-gpu1` (port 2010, TM 8110) · 시나리오 sc01/sc04/sc07 | Sionna worker GPU 1 · sc01/sc04/sc07 |
| 2 | CARLA 컨테이너 `carla-gpu2` (port 2020, TM 8120) · 시나리오 sc02/sc05/sc08 | Sionna worker GPU 2 · sc02/sc05/sc08 |
| 3 | CARLA 컨테이너 `carla-gpu3` (port 2030, TM 8130) · 시나리오 sc03/sc06    | Sionna worker GPU 3 · sc03/sc06 |

---

## 4. 시나리오 구성 (8개)

모두 `Town10HD_Opt` 맵, UE 는 CARLA traffic manager 의 자율주행 첫 번째 차량 (`vehicle.chevrolet.impala`), BS 는 고정 설치.

| ID | 설명 | BS 위치 (x, y, z) [m] | 카메라 pitch · yaw [°] | Seed |
|---|---|---|---|---|
| sc01 | E-W 도로, 건물 벽면 (동측) | (5, 38, 7)     | −15, 180 | 42  |
| sc02 | E-W 도로, 건물 벽면 (서측) | (−75, 38, 7)   | −15, 180 | 101 |
| sc03 | N-S 도로, 건물 모서리 (동)  | (112, 35, 7)   | −15, −90 | 202 |
| sc04 | SW 교차로, 건물 facade     | (−50, −30, 7)  | −18, 45  | 303 |
| sc05 | NW 해안가 건물 벽면        | (−85, 35, 7)   | −15, 135 | 404 |
| sc06 | 중앙 교차로, 인도 pole     | (−43, 8, 6)    | −15, 90  | 505 |
| sc07 | 남부 대로, 건물 벽면       | (5, −50, 7)    | −15, 0   | 606 |
| sc08 | E-W 도로, 옥상 (고도)      | (5, 38, 12)    | −22, 180 | 707 |

- BS 높이 6–7 m (벽/인도) 및 12 m (옥상) — 실전 스몰셀 배치 근사
- 각 시나리오는 별도의 CARLA random seed 를 사용 (교통 흐름 다양화)

---

## 5. 시뮬레이션 · 채널 파라미터

### 5.1 시간축

| 파라미터 | 값 | 비고 |
|---|---|---|
| `DELTA_T` | **0.0005 s (0.5 ms)** | CARLA sync 모드 fixed_delta_seconds |
| Sim 레이트 | **2,000 Hz** | |
| `SIM_STEPS` | **400,000** / 시나리오 | 실시간 200 s 시뮬레이션 |
| `RT_INTERVAL` | **1** (매 sim step) | Sionna ray-trace 주기 = 0.5 ms → Nyquist f_d = 1 kHz → v_max ≈ 38 km/h |
| `IMAGE_INTERVAL` | **20** (= 10 ms) | `dataset_final` 센서 저장 주기 (100 Hz) |
| `IMAGE_INTERVAL_1MS` | **2** (= 1 ms) | `dataset_1ms` 센서 재렌더 저장 주기 (1 kHz) |
| Doppler 보간 | **없음** (native per-step RT) | |

### 5.2 물리 채널

| 파라미터 | 값 |
|---|---|
| Carrier frequency | **28 GHz** (5G NR FR2) |
| Bandwidth | **50 MHz** |
| 파장 λ | **10.71 mm** |
| Subcarriers (FFT) | **512** |
| Subcarrier spacing | ≈ **97.66 kHz** (= BW/NSC) |
| BS 안테나 | **1 × 16 ULA** (수평, 0.5 λ, dipole, V-pol) |
| UE 안테나 | **1** (단일 dipole, V-pol) |
| Max path depth | **5** (LOS + 최대 5차 반사, 깊은 NLOS) |
| Max paths kept | **25** (수신 전력 top-25, DeepVerse 정렬) |
| Ray samples per source | **1 × 10⁶** |
| Scattering coef. | 0.3 |
| Solver options | LOS + specular + diffuse + refraction + diffraction + edge_diffraction 모두 enabled |

### 5.3 ITU 재질 속성 (`sionna.rt`)

| 재질 | ε_r | σ [S/m] | 대응 mesh |
|---|---|---|---|
| concrete | 5.24 | 0.58  | buildings / roads / sidewalks / ground |
| brick    | 3.75 | 0.12  | walls |
| metal    | — (σ=1e7) | 1e7 | fences, 차량 scatterer (box mesh) |
| marble   | 6.00 | 0.25  | (필요 시) |

### 5.4 차량 산란체

- 차량 수: **40 대** (CARLA TM 자율주행)
- Sionna scene 에 차량을 **box mesh (2.25 × 0.9 × 0.75 m, metal)** 로 추가
- 매 sim step 마다 `vehicles_all/vehicles_{step:06d}.npy` 에 40 대 전체 위치/yaw 기록 → Sionna worker 가 **매 RT 프레임 최신 위치로 scatterer 업데이트**
- 최대 20 대까지 scene 에 추가 (Sionna 계산량 제한)

---

## 6. 센서 구성 (BS에 부착)

### 6.1 RGB 카메라

| 파라미터 | 값 |
|---|---|
| Blueprint | `sensor.camera.rgb` |
| 해상도 | **1280 × 720** |
| FOV | 110° |
| 저장 주기 | `dataset_final`: 10 ms (매 20 sim step), `dataset_1ms`: 1 ms (매 2 sim step) |
| 파일 수 / 시나리오 | `dataset_final`: **20,000**, `dataset_1ms`: **200,000** |
| Pitch / Yaw | 시나리오별 (섹션 4) |
| 저장 포맷 | PNG (RGB) |

### 6.2 LiDAR

| 파라미터 | 값 |
|---|---|
| Blueprint | `sensor.lidar.ray_cast` |
| Channels | 32 |
| Range | 100 m |
| Points per second | 100,000 |
| Rotation frequency | 10 Hz |
| Upper / lower FOV | +5° / −25° |
| 저장 주기 | `dataset_final`: 10 ms, `dataset_1ms`: 1 ms |
| 저장 포맷 | `.npy`, shape `(N, 4)` = `(x, y, z, intensity)` |

⚠ LiDAR 는 회전주파수 10 Hz 이므로 저장 파일은 **부분 스윕(sector)** 이다. `dataset_final`의 10 ms 파일은 약 36° sector, `dataset_1ms`의 1 ms 파일은 약 3.6° sector에 해당한다. 전체 360° 클라우드가 필요하면 약 100 ms 구간을 누적해야 한다.

### 6.3 Radar

| 파라미터 | 값 |
|---|---|
| Blueprint | `sensor.other.radar` |
| Horizontal FOV | 60° |
| Vertical FOV | 10° |
| Range | 100 m |
| Points per second | 1,500 |
| 저장 주기 | `dataset_final`: 10 ms, `dataset_1ms`: 1 ms |
| 저장 포맷 | `.npy`, shape `(N, 4)` = `(velocity, azimuth, altitude, depth)` |

각 파일 당 평균 **≈15 detections** (희박).

---

## 7. 디렉터리 구조

```
dataset_final/
├── sc01/
│   ├── positions/            positions_NNNNNN.npy   (400,000)  UE 위치 (x, y, z) m
│   ├── velocities/           velocities_NNNNNN.npy  (400,000)  UE 속도 (vx, vy, vz) m/s
│   ├── channels/             channel_NNNNNN.npy     (400,000)  H matrix (덮어씀, 아래 ofdm과 동일)
│   ├── ofdm/                 ofdm_NNNNNN.npy        (400,000)  OFDM channel H (16, 512) complex128
│   ├── angle_delay/          ad_NNNNNN.npy          (400,000)  angle-delay (32, 32) complex64
│   ├── cir/                  cir_NNNNNN.npz         (400,000)  CIR keyframe: a, tau
│   ├── vehicles_all/         vehicles_NNNNNN.npy    (400,000)  주변 40대 (x, y, z, yaw)
│   ├── images/               frame_NNNNNN.png       (20,000)   1280×720 RGB
│   ├── lidar/                lidar_NNNNNN.npy       (20,000)   32-ch, partial sweep
│   ├── radar/                radar_NNNNNN.npy       (20,000)   horizontal 60° / vertical 10°
│   └── metadata.json                                              수집 파라미터 + 시간
├── sc02/  (동일 구조)
├── sc03/
├── sc04/
├── sc05/
├── sc06/
├── sc07/
└── sc08/

dataset_1ms/
├── sc01/
│   ├── positions/            positions_NNNNNN.npy   (400,000)  UE 위치 (0.5 ms)
│   ├── velocities/           velocities_NNNNNN.npy  (400,000)  UE 속도 (0.5 ms)
│   ├── vehicles_all/         vehicles_NNNNNN.npy    (400,000)  주변 40대 (0.5 ms)
│   ├── channels/             channel_NNNNNN.npy     (400,000)  Phase-1 geometric placeholder (16, 64)
│   ├── images/               frame_NNNNNN.png       (200,000)  1 ms RGB
│   ├── lidar/                lidar_NNNNNN.npy       (200,000)  1 ms LiDAR
│   └── radar/                radar_NNNNNN.npy       (200,000)  1 ms Radar
├── sc02/  (동일 구조)
...
└── sc08/

sionna_scene_final/
├── sc01/                 ← CARLA 에서 export 한 원본 OBJ (buildings, roads, ...)
├── sc01_sionna/          ← Blender 가 변환한 PLY mesh + Sionna XML
│   ├── meshes/
│   ├── sc01.xml          ← Sionna `load_scene(...)` 에 넣는 파일
│   └── scene_info.json
...
```

---

## 8. 파일 포맷 상세

### 8.1 핵심 numpy 배열

| 경로 패턴 | shape | dtype | 의미 |
|---|---|---|---|
| `sc{k}/positions/positions_{i:06d}.npy` | `(3,)` | `float64` | UE 위치 (x, y, z), CARLA 월드 프레임 (m) |
| `sc{k}/velocities/velocities_{i:06d}.npy` | `(3,)` | `float64` | UE 속도 (vx, vy, vz) (m/s) |
| `sc{k}/vehicles_all/vehicles_{i:06d}.npy` | `(n_veh, 4)` | `float64` | (x, y, z, yaw[°]) · CARLA 월드 프레임 |
| `sc{k}/ofdm/ofdm_{i:06d}.npy` | `(16, 512)` | `complex128` | OFDM 채널 H = Σ_m a_m exp(−j2π f_k τ_m), antenna × subcarrier |
| `sc{k}/channels/channel_{i:06d}.npy` | `(16, 512)` | `complex128` | Phase 2 가 OFDM 으로 덮어씀, 같은 내용 |
| `sc{k}/angle_delay/ad_{i:06d}.npy` | `(32, 32)` | `complex64` | Beamspace × delay 태핑 (LWM-Temporal 호환, 16→32 zero-pad, 첫 32 tap) |
| `sc{k}/cir/cir_{i:06d}.npz` | — | — | 원본 path: `a` (16, ≤25) complex64, `tau` (≤25,) float32 |

### 8.2 이미지 / LiDAR / Radar

| 경로 | 포맷 | 주기 |
|---|---|---|
| `dataset_final/sc{k}/images/frame_{i:06d}.png` | PNG (RGB, 1280×720) | 10 ms |
| `dataset_final/sc{k}/lidar/lidar_{i:06d}.npy` | `(N, 4)` float32 | 10 ms |
| `dataset_final/sc{k}/radar/radar_{i:06d}.npy` | `(N, 4)` float32 | 10 ms |
| `dataset_1ms/sc{k}/images/frame_{i:06d}.png` | PNG (RGB, 1280×720) | 1 ms |
| `dataset_1ms/sc{k}/lidar/lidar_{i:06d}.npy` | `(N, 4)` float32 | 1 ms |
| `dataset_1ms/sc{k}/radar/radar_{i:06d}.npy` | `(N, 4)` float32 | 1 ms |

**인덱스 i 는 sim step 번호** (0 ~ 399,999). 이미지/LiDAR/Radar 는 `i % 20 == 0` 인 step 에만 존재.
`dataset_1ms`의 이미지/LiDAR/Radar 는 `i % 2 == 0` 인 step 에 존재한다. 최신 멀티모달 실험에서는 channel target/history는 `dataset_final/sc{k}/ofdm` 또는 `channels`에서 읽고, 센서는 같은 step index의 `dataset_1ms/sc{k}/images|lidar|radar`에서 읽는다.

### 8.3 좌표계

- **CARLA world frame** (left-handed, +X east, +Y south, +Z up)
- Sionna 로 변환할 때는 **y 부호 뒤집음** (`y_sionna = −y_carla`) — CARLA left-handed → Sionna right-handed

### 8.4 `metadata.json` (시나리오당 1개)

각 시나리오의 `dataset_final/sc{k}/metadata.json` 에 기록된 주요 필드:

```json
{
  "scenario_id": "sc01",
  "map_name": "Town10HD_Opt",
  "delta_t": 0.0005,
  "num_steps": 400000,
  "seed": 42,
  "num_vehicles": 40,
  "image_interval": 20,
  "bs_position": [5.0, 38.0, 7.0],
  "bs_camera_pitch": -15.0,
  "bs_camera_yaw": 180.0,
  "camera_width": 1280, "camera_height": 720, "camera_fov": 110.0,
  "carrier_freq": 2.8e10, "bandwidth": 5.0e7,
  "num_subcarriers": 512, "num_bs_antennas": 16, "num_ue_antennas": 1,
  "steps_collected": 400000, "images_collected": 20000,
  "collection_time_s": 25254.72,
  "timestamp": "2026-04-09T09:48:17",
  "ue_vehicle_type": "vehicle.chevrolet.impala",
  "carla_version": "0.9.15", "weather": "ClearNoon",
  "channel_backend": "sionna_carla_blender_raytracing",
  "pipeline": "CARLA→Blender→Sionna",
  "sim_delta_t_s": 0.0005, "sim_rate_hz": 2000.0,
  "rt_interval_frames": 1, "rt_interval_s": 0.0005, "rt_rate_hz": 2000.0,
  "doppler_interpolated": false,
  "nyquist_fd_hz": 1000.0,
  "max_depth": 5, "ray_samples": 1000000,
  "materials": "ITU concrete/brick/metal/marble at 28GHz"
}
```

---

## 9. 데이터 포인트 개수 (시나리오당)

| 모달리티 | 개수 | dtype | 단일 파일 크기 |
|---|---|---|---|
| `dataset_final/positions` | 400,000 | float64 (3,) | 152 B |
| `dataset_final/velocities` | 400,000 | float64 (3,) | 152 B |
| `dataset_final/ofdm` | 400,000 | complex128 (16, 512) | 128 KB |
| `dataset_final/channels` | 400,000 | complex128 (16, 512) | 128 KB (ofdm 중복) |
| `dataset_final/angle_delay` | 400,000 | complex64 (32, 32) | 8 KB |
| `dataset_final/cir` | 400,000 | compressed npz | ~3 KB |
| `dataset_final/vehicles_all` | 400,000 | float64 (40, 4) | ~1.4 KB |
| `dataset_final/images` | 20,000 | PNG 1280×720 | 파일별 가변 |
| `dataset_final/lidar` | 20,000 | float32 (N, 4) | 파일별 가변 |
| `dataset_final/radar` | 20,000 | float32 (N, 4) | 파일별 가변 |
| `dataset_1ms/images` | 200,000 | PNG 1280×720 | 파일별 가변 |
| `dataset_1ms/lidar` | 200,000 | float32 (N, 4) | 파일별 가변 |
| `dataset_1ms/radar` | 200,000 | float32 (N, 4) | 파일별 가변 |

실측 크기: `dataset_final` 약 **1.1 TB**, `dataset_1ms` 약 **2.4 TB**.

---

## 10. 로딩 예시 (Python)

### 10.1 기본 로딩

```python
import numpy as np
from pathlib import Path
from PIL import Image
import json

base = Path("/mnt/ssd_7t_2/carla-wireless-dataset")
root = base / "dataset_final/sc01"
meta = json.load(open(root / "metadata.json"))

# 단일 step 에서 채널 + 위치
i = 12345
H       = np.load(root / f"ofdm/ofdm_{i:06d}.npy")          # (16, 512) c128
ad      = np.load(root / f"angle_delay/ad_{i:06d}.npy")     # (32, 32) c64
ue_pos  = np.load(root / f"positions/positions_{i:06d}.npy")  # (3,)
ue_vel  = np.load(root / f"velocities/velocities_{i:06d}.npy")
veh_all = np.load(root / f"vehicles_all/vehicles_{i:06d}.npy")  # (≤40, 4)

# 원본 multipath 파라미터 (필요 시)
with np.load(root / f"cir/cir_{i:06d}.npz") as cir:
    a, tau = cir["a"], cir["tau"]      # (16, ≤25), (≤25,)

# 동일 시각의 legacy 센서(dataset_final, 10 ms 단위)
if i % meta["image_interval"] == 0:
    img = Image.open(root / f"images/frame_{i:06d}.png")
    lidar = np.load(root / f"lidar/lidar_{i:06d}.npy")      # (N, 4)
    radar = np.load(root / f"radar/radar_{i:06d}.npy")

# 최신 1 ms 센서(dataset_1ms)와 channel 결합
sensor_root = base / "dataset_1ms/sc01"
if i % 2 == 0:
    img_1ms = Image.open(sensor_root / f"images/frame_{i:06d}.png")
    lidar_1ms = np.load(sensor_root / f"lidar/lidar_{i:06d}.npy")
    radar_1ms = np.load(sensor_root / f"radar/radar_{i:06d}.npy")
```

### 10.2 시간 시퀀스 로딩 (LWM-Temporal 호환)

```python
# 1초 구간 (2000 frame, 2 kHz)
T = 2000
start = 0
seq_H  = np.stack([np.load(root / f"ofdm/ofdm_{i:06d}.npy") for i in range(start, start+T)])
seq_ad = np.stack([np.load(root / f"angle_delay/ad_{i:06d}.npy") for i in range(start, start+T)])
# shape: seq_H (2000, 16, 512) c128,  seq_ad (2000, 32, 32) c64
```

### 10.3 멀티모달 정렬

센서 `i_sensor` 번째 저장 파일의 step 번호:
```python
step_final = i_sensor * 20   # dataset_final sensors: 0, 20, 40, ..., 399980
step_1ms   = i_sensor * 2    # dataset_1ms sensors: 0, 2, 4, ..., 399998
```

최신 실험에서는 `step_1ms`의 센서 파일을 읽고, 같은 step 번호의 `dataset_final` 채널/포지션 파일을 같이 읽으면 **1 ms 해상도 멀티모달 pair** 를 얻는다 (총 200,000 쌍/시나리오).

---

## 11. Nyquist 분석

`λ = c / f_c = 10.71 mm`. 2 kHz 샘플링에서 Nyquist Doppler = **1,000 Hz** → **v_max (safe) ≈ 38.6 km/h**.

| 차량 속도 | f_d | 상태 |
|---|---|---|
| 30 km/h | 778 Hz | ✅ Nyquist 만족 |
| 38 km/h | 988 Hz | ✅ 한계 |
| 50 km/h | 1,298 Hz | ⚠ 오버샷 (아날로그 aliasing 시작) |
| 100 km/h | 2,596 Hz | ❌ 완전 aliased |

**도심 (Town10HD_Opt) 기본 CARLA TM 속도 분포는 30–50 km/h** 이므로, 일부 고속 순간에서는 soft aliasing 이 발생할 수 있음. 50 km/h 엄격 Nyquist 가 필요하면 `DELTA_T = 0.00025` (4 kHz) 로 재수집 권장.

---

## 12. 재현성

- **랜덤 seed**: 시나리오별 고정 (섹션 4 표), CARLA TM seed 동일
- **CARLA 버전**: Docker pin (`carlasim/carla:0.9.15`)
- **Map**: `Town10HD_Opt` 기본 (수정 없음)
- **스크립트**:
  - `collect_final.py` — 전체 파이프라인 오케스트레이션
  - `collect_data.py` — Phase 1 CARLA 수집 (per scenario)
  - `export_carla_geometry.py` — 정적 geometry 추출
  - `blender_to_sionna.py` — PLY/XML 변환
  - `run_collect_final.sh` — LD_LIBRARY_PATH / env 설정 런처
- **실행 명령** (재현):
  ```bash
  # 1. CUDA runtime 설치 (최초 1회)
  pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 \
              nvidia-cuda-nvrtc-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 \
              nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12

  # 2. CARLA 컨테이너 3개 기동 (GPU 1/2/3)
  for i in 1 2 3; do
    port=$((2000 + i*10))
    docker run -d --rm --name carla-gpu$i --gpus "\"device=$i\"" \
      -p $port:$port -p $((port+1)):$((port+1)) -p $((port+2)):$((port+2)) \
      carlasim/carla:0.9.15 \
      ./CarlaUE4.sh -RenderOffScreen -carla-rpc-port=$port -quality-level=Epic -nosound
  done

  # 3. 수집 시작
  cd /mnt/ssd_7t_2/carla-wireless-dataset
  nohup ./run_collect_final.sh > /tmp/collect_main.log 2>&1 &
  ```

---

## 13. 알려진 이슈 / 제약사항

1. **고속 구간 soft aliasing**: v > 38 km/h 에서 Doppler aliasing 발생 가능.
2. **LiDAR partial sweep**: 한 파일 = 7.2° sector, 전체 회전은 10 파일 = 100 ms 단위로만 재구성 가능.
3. **Radar sparsity**: `dataset_final`은 파일당 평균 약 15 points (1,500 pts/s × 10 ms), `dataset_1ms`는 더 희박할 수 있음.
4. **Cleanup C++ race**: 시나리오 종료 시 `terminate called without an active exception` 경고가 로그에 찍히나, **데이터 저장 후에 발생**하므로 파일 내용에 영향 없음.
5. **Channel file 중복**: `channels/` 와 `ofdm/` 은 동일 데이터 (Phase 2 가 `channels/` 를 OFDM 으로 덮어씀). 저장 공간 절약이 중요하면 `channels/` 삭제 가능.
6. **Angle-delay truncation**: 32×32 angle-delay 는 앞 32 delay tap (≈ 640 ns = 192 m 경로) 만 보존. 그 이상 delay 는 잘림.
7. **Ray-trace BS 위치 offset**: BS 카메라 좌표는 CARLA world 기준이지만, Sionna 에 전달할 때 `(x, −y, z)` 로 y 부호만 뒤집음 (좌우-loidal 좌표계 차이).
8. **`dataset_1ms/channels` 주의**: 현재 `dataset_1ms`의 `channels/`는 `(16, 64)` geometric placeholder다. 최신 멀티모달 실험의 채널 입력/타깃은 `dataset_final`의 `(16, 512)` Sionna RT channel을 사용해야 한다.

---

## 14. TODO (수집 완료 후 채워 넣기)

수집이 끝난 뒤 다음 값들을 이 섹션에 업데이트:

- [ ] 실측 Phase 1 wall-clock (시나리오별)
- [ ] 실측 Phase 2 wall-clock (GPU별)
- [ ] 실측 각 시나리오 최종 파일 개수 (ofdm/ad/positions/images/lidar/radar/vehicles_all/cir)
- [ ] 실측 시나리오당 / 전체 저장 용량
- [ ] Sionna RT 실측 frame rate (fr/s per GPU)
- [ ] UE 평균/최대 속도 (시나리오별) — Nyquist 실제 커버리지 검증용
- [ ] 채널 통계 (PDP, angular spread) 샘플 플롯
- [ ] `md5sum` 체크섬 또는 샘플 해시

---

## 15. 인용 / 라이선스

- **CARLA**: MIT License, cite Dosovitskiy et al. *"CARLA: An Open Urban Driving Simulator,"* CoRL 2017.
- **Sionna**: Apache 2.0, cite Hoydis et al. *"Sionna: An Open-Source Library for Next-Generation Physical Layer Research,"* 2022.
- **본 데이터셋**: 내부 연구용 · 추후 공개 시 라이선스 별도 결정.

---

## 16. 연락 / 문의

| 항목 | 정보 |
|---|---|
| 수집 장비 | `/mnt/ssd_7t_2/carla-wireless-dataset/` |
| 소유자 | local workspace (`/mnt/ssd_7t_2`) |
| 버전 관리 | (해당 없음, 로컬 디렉터리) |
