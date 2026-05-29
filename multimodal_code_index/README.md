# Multimodal Code Index

이 폴더는 멀티모달 채널 예측 관련 코드의 "정리용 인덱스"입니다.
원본 코드는 옮기지 않았고, 필요한 파일과 폴더를 symlink로 묶었습니다.

모델 구현의 canonical 위치는 `multimodal_code_index/models/`입니다.
루트 `models/`는 기존 `from models...` import 호환을 위한 symlink이며,
별도 모델 사본을 두지 않습니다.

## 목적

할 일은 멀티모달 입력을 이용한 wireless channel prediction입니다.
현재 레포에서는 두 실험 라인이 중요합니다.

1. `01_dual_root_4models`
   - LSTM, LWM, LWM Temporal, Chiron 멀티모달 4모델 비교 실험
   - 재측정과 모델 비교를 할 때 우선 볼 경로

2. `02_task04_active`
   - 현재 문서상 active channel + image 실험
   - slot-index positional encoding, image time offset, FoV mask 처리가 들어간 라인

## 폴더 구조

| 폴더 | 내용 | 원본 위치 |
|---|---|---|
| `01_dual_root_4models/` | 4개 멀티모달 모델 실험 runner/config/scripts 인덱스. 모델은 `multimodal_code_index/models/` 사용 | `experiment_dual_root/`, `models/` |
| `02_task04_active/` | Task04 active 모델/데이터 래퍼/실험 runner | `scripts/`, `feedback_solve/tasks/04_image_16x4_strategy/` |
| `03_root_model_zoo/` | model zoo 바로가기 | `models/`, `train.py`, `evaluate.py` |
| `04_docs/` | 멀티모달 구조, 설계, prior-work 관련 문서 | `reports/`, `data_verification/reports/` |
| `05_current_data/` | 현재 확인된 데이터셋 바로가기 | `wireless-dataset/` |

## 먼저 볼 파일

### 4모델 재측정 라인

- `01_dual_root_4models/train_multimodal4.py`
  - `lstm`, `lwm`, `lwm_temporal`, `chiron`을 선택해 학습하는 main runner
- `01_dual_root_4models/models/`
  - `multimodal_code_index/models/`로 가는 symlink
  - 실제 4모델 구현은 `multimodal_code_index/models/lstm_multimodal.py`,
    `multimodal_code_index/models/lwm_multimodal.py`,
    `multimodal_code_index/models/lwm_temporal_multimodal.py`,
    `multimodal_code_index/models/chiron_multimodal.py`
- `01_dual_root_4models/dataset_loader.py`
  - `channels/`와 `images/`를 읽는 데이터 로더
- `01_dual_root_4models/configs/`
  - 기존 실험 config. 현재는 옛 경로인 `dataset_final` / `dataset_1ms`를 가리킴
- `01_dual_root_4models/old_results/`
  - 이전 데이터셋 기준 결과. 새 데이터 결과와 섞으면 안 됨

### Task04 active 라인

- `02_task04_active/sc01_multimodal_experiment.py`
  - `channel_only`, `multimodal`, `image_only`, last-channel baseline 실험 runner
- `02_task04_active/task04_code/multi_modal_predictator_task04.py`
  - `MultiModalPredictator`에 slot-index PE를 추가한 모델
- `02_task04_active/task04_code/task04_dataset_wrapper.py`
  - `k_img`, FoV flag, out-of-FoV image masking에 필요한 필드 추가
- `02_task04_active/current_architecture.md`
  - 현재 active 구조 설명 문서

### Model zoo / MSCP-inspired 모델

- `03_root_model_zoo/models/`
  - `multimodal_code_index/models/`로 가는 symlink
- `03_root_model_zoo/models/mscp_multimodal.py`
  - 논문 MSCP 방향에 맞춰 `channel_history + image + scene/geometry feature`를 함께 받는 실험용 모델
  - `ue_position`, `ue_velocity`, `vehicles_all`, `scene_features` 같은 명시적 환경 feature를 scene token으로 인코딩
  - 현재 runner에는 아직 연결하지 않았으므로 학습하려면 dataset/runner에서 scene field를 batch로 넘겨야 함

## 현재 데이터셋 기준 주의점

현재 실제 데이터는 다음 경로에 있습니다.

```text
/mnt/ssd_7t_2/carla-wireless-dataset/wireless-dataset/sc01 ... sc08
```

`scXX` 안에는 다음 모달리티가 있습니다.

```text
channels/
ofdm/
images/
lidar/
radar/
positions/
velocities/
metadata.json
```

반면 `experiment_dual_root/configs/*.env`는 아직 아래 옛 경로를 기본값으로 사용합니다.

```text
CHANNEL_ROOT=/mnt/ssd_7t_2/carla-wireless-dataset/dataset_final
SENSOR_ROOT=/mnt/ssd_7t_2/carla-wireless-dataset/dataset_1ms
```

재측정할 때는 현재 데이터 루트를 쓰는 override config를 지정해야 합니다.

```bash
CONFIG_FILE=/mnt/ssd_7t_2/carla-wireless-dataset/multimodal_code_index/01_dual_root_4models/current_wireless_dataset_sc01_sc02.env \
bash /mnt/ssd_7t_2/carla-wireless-dataset/experiment_dual_root/scripts/train_all_4_multimodal_sc01_sc02.sh
```

단일 모델만 돌릴 때는 다음 형태를 사용합니다.

```bash
CONFIG_FILE=/mnt/ssd_7t_2/carla-wireless-dataset/multimodal_code_index/01_dual_root_4models/current_wireless_dataset_sc01_sc02.env \
MODEL=chiron \
bash /mnt/ssd_7t_2/carla-wireless-dataset/experiment_dual_root/scripts/train_one_multimodal4_sc01_sc02.sh
```

## 현재 active 입력 모달리티

현 상태에서 실제 runner 기준으로 보면:

- `experiment_dual_root` 4모델 runner는 기본적으로 `channel + image`만 사용
- 모델 클래스에는 `PointNetEncoder` 기반 LiDAR 경로가 있으나 runner에서 `use_lidar=False`
- Radar는 데이터에는 있으나 현재 active model/runner 입력 경로에는 없음
- Task04도 `channel + image` 중심

즉, 지금 바로 재측정 가능한 멀티모달은 `channel + RGB image`입니다.
LiDAR/Radar까지 포함하려면 데이터 로더, batch 이동, model build 인자를 별도로 확장해야 합니다.

## 현재 LSTM/LWM 구조

`lstm`, `lwm`은 기존 per-subcarrier token 구조가 아니라 wideband time
token 구조입니다.

```text
channel_history: (B, K, Na, Nsc, 2)
  -> (B, K, Na*Nsc*2)
  -> channel encoder
  -> (B, K, D)
```

멀티모달 모드에서는 RGB frame을 frame token으로 요약한 뒤
`image_time_offsets`와 `delta_t`로 channel history의 K개 time step에 맞춰
정렬하고, 각 시간마다 `[channel_t, rgb_t]`를 attention으로 fusion합니다.

이 변경 때문에 `lstm`, `lwm`의 기존 checkpoint와 결과는 새 구조와 직접
비교하지 않는 것이 안전합니다. 최소 재실험 대상은 `lstm`, `lwm`의
`channel_only`와 `multimodal`입니다.

## 모델 명칭 정리

- 보고서/발표에서는 `lwm`보다 `Transformer encoder`로 표기하는 것이 좋음
- `lwm_temporal`은 `Spatio-temporal Transformer encoder`로 표기하는 것이 좋음
- `mscp_multimodal.py`는 paper-faithful 복제가 아니라 현재 데이터셋에서 바로 확장 가능한 MSCP-inspired 구현임
- `NOVA`는 현재 4-model multimodal runner에 포함되지 않고 channel-only 성격이 강해서 `03_root_model_zoo/models/` 선별 목록에서 제외함

## 권장 작업 순서

1. 현재 데이터셋 sanity check
   - `wireless-dataset/scXX/channels`, `images`, `lidar`, `radar` count와 index alignment 확인

2. baseline 재측정
   - last-channel copy
   - `channel_only`
   - `channel + image`

3. 4모델 재측정
   - `lstm`
   - `lwm`
   - `lwm_temporal`
   - `chiron`

4. 결과 분리
   - `old_results/`는 이전 데이터셋 결과
   - 새 결과는 `current_wireless_dataset` 같은 run name으로 분리

## 편집 원칙

이 폴더 안의 대부분은 symlink입니다.
코드를 수정해야 하면 symlink 파일을 따라가도 되지만, 실제 변경 대상은 원본 파일입니다.

모델 코드를 수정할 때는 항상 `multimodal_code_index/models/`를 수정합니다.
`multimodal_code_index/01_dual_root_4models/models`와
`multimodal_code_index/03_root_model_zoo/models`, 루트 `models/`는 모두
`multimodal_code_index/models/`의 별칭입니다.

4모델 실험 수정 우선순위:

1. `experiment_dual_root/code/train_multimodal4.py`
2. `experiment_dual_root/code/dataset_loader.py`
3. `multimodal_code_index/models/*_multimodal.py`
4. `experiment_dual_root/configs/*.env`
5. `experiment_dual_root/scripts/*.sh`
