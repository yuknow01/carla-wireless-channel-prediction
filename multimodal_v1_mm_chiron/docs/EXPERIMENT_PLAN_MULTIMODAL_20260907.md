# EXPERIMENT_PLAN_MULTIMODAL_20260907.md — 채널 예측(K=16→H=4, 10 ms)에 센서를 붙이는 실험 계획 (안, **미실행**)

- 작성 2026-09-07, **v2 갱신 2026-09-07 17:30**: 실험용 모델 `cp/cp_multimodal.py`(`mm:chiron`)·센서 로더·캐시 구현 완료 → 실측 비용 반영, M2 에 channel+camera+LiDAR 추가. 구조 문서 = `MULTIMODAL_CP_ARCHITECTURE.md`. 사용자 요청 "멀티모달 실험을 하려고 한다면 계획". 실행은 이 문서를 사용자가 검토·승인한 뒤에만 한다(launch 게이트).
- 표기: **[근거]** = 파일·실행 결과로 확인한 사실, **[추측]** = 추정·가정(실행으로 확인 필요). 문장마다 붙인다.
- 기반 문서: `MULTIMODAL_ARCHITECTURES.md`(§4 선행 센서 기여, §5 설계 2안·절제 프로토콜), `REPORT_STEP3_MULTIMODAL.md`(센서 조건표), `EXPERIMENT_LOG_CHANNEL_PRED.md`(S1 결과), `EXPERIMENT_PLAN_FINAL_20260904.md`(공통 프로토콜).

## 0. 한 줄 요약

**질문**: 이 데이터셋에서 RSU(BS-측) 센서가 채널 예측 NMSE를 개선하는가. **기본 가설 = 증분 0** — 같은 데이터셋의 빔 예측([B] 재현)과 다른 데이터셋의 채널 값 예측에서 카메라·LiDAR 기여가 zero/shuffle/foreign 대조로 반복해서 0으로 나왔다 [근거: `MULTIMODAL_ARCHITECTURES.md` §4.1~4.3]. 예외 후보는 (i) 위치·기하(경로가 결정론적이라 미래 기하가 위상 궤적을 제약할 수 있음 [추측]), (ii) 레이더(다른 데이터셋에서 차폐 예고 AUC 0.78 [근거: `scenario_pilot/REPORT_BSCAM.md` §1]). 따라서 **싼 안 (b) 위치 경유 → 비싼 안 (a) 센서 직접 입력** 순으로, 각 단계에 절제(zero/shuffle/foreign)를 붙여 "증분이 있는지"를 시드 편차 대비로 판정한다.

## 1. 전제(확정된 사실)

| 항목 | 내용 | 근거 |
|---|---|---|
| 태스크·프로토콜 | K=16→H=4, Δ=10 ms, 64×64 채널, RX 행 표본, 창별 RMS 정규화, 분할 B1(val = ringroad·gastation·Town10_crossroad), 4지평 NMSE 손실, patience 5, best-val ckpt, AMP 없음, seed 42 | [근거] `EXPERIMENT_PLAN_FINAL_20260904.md`, `cp/train_cp.py` |
| 채널 전용 성능(비교 대상) | chiron 19.2 M median −16.16/−14.96/−13.79/−12.96 dB(pooled 평균 −10.20), NOVA 15.7 M −15.90/−14.97/−13.92/−13.11(−10.19), LWM v1.1 구조 2.9 M −17.06/−15.28/−13.87/−12.89(−9.99) | [근거] `EXPERIMENT_LOG_CHANNEL_PRED.md`, `outputs_cp/*/result.json` |
| 시드 편차 | **미측정**(S2 시드 반복이 사용자 지시로 취소됨) — 센서 증분의 유의성 판정에 필수 | [근거] `REPRO_V2_EXPERIMENTS_SINCE_20260829.md` 9/7 09:40·11:35 항목 |
| 센서·채널 정렬 | RSU·CAV 센서가 채널과 같은 6자리 프레임 번호로 10 ms 1:1 정렬, 매 프레임 실제 캡처 | [근거] `REPORT_STEP3_MULTIMODAL.md` §3-0·3-1 |
| 센서 종류 | RSU: RGB 640×480, 깊이 uint8, LiDAR pcd 약 28k점, 레이더 json 1.4~1.7k점; CAV: RGB×4, LiDAR, yaml(true/predicted ego pos, GPS, IMU, 주변 차량) | [근거] STEP3 §3-0 |
| BS-측 센서 원칙 | 센서 입력은 RSU(BS 탑재)만. CAV 카메라·LiDAR는 쓰지 않음 | [근거] 메모리 `feedback_bs_side_sensors`; [B] IV-C1 |
| **실험용 모델·캐시(구현 완료, 9/7)** | `cp/cp_multimodal.py::MMChiron`(`mm:chiron`, chiron 백본 + 게이트 cross-attention 3층, 게이트 0 초기화 → 채널 전용 chiron 과 출력 max\|diff\| 0.0 실측), 로더 `cp/cp_sensor_data.py`(`MMWindowSet`, 창 단위 LRU), 사전 캐시 `derived_cp/sensor_cache/`(LiDAR 8×8 풀링 807 MB, 레이더 래스터 64×64 270 MB, 위치 특징 9.3 MB; **53,800 프레임 전부 커버, 누락 0**), 스모크 학습 1 epoch 통과, 기존 모델 무회귀 확인 | [근거] `MULTIMODAL_CP_ARCHITECTURE.md` §0.2~0.4, `scripts/13_probe_multimodal_cp.json`, `outputs_cp_smoke/` |
| 실측 비용(B=32, GPU0, 로더 포함 epoch) | chiron 채널 전용 0.40 h · cam 0.86 · lidar 0.87 · radar 0.81 · cam+lidar 0.91 · cam+lidar+radar 0.94 · pos(token) 0.79 · pos(broadcast) 0.69 h/epoch; 학습 피크 메모리 4.7~7.4 GB | [근거] `MULTIMODAL_CP_ARCHITECTURE.md` §2 |
| 위치 변환 점검 | true 위치의 방위각 φ vs `derived/aod` AoD: 53,800 프레임 median \|Δ\| 0.003°, 2° 이내 99.76 %(최대 39° 구간은 NLOS 추정) → xy·방위각 규약은 맞음, z 원점·부호는 미검증 | [근거] 같은 문서 §0.3·§4 |
| 기존 캐시 | RSU 카메라0 → ViT 패치 캐시 `derived/feat_patch_rsu224/`(4.9 GB, fp16 [1,196,768]/프레임), RSU LiDAR → PointPillars 백본 캐시 `derived/feat_pp_rsu/`(217 GB, [384,100,176]/프레임) — [B] 재현에서 만든 것, 멀티모달 프로브에서 실제 로드 성공 | [근거] `ls derived/`, `MULTIMODAL_ARCHITECTURES.md` §0.2·§2.1 |
| RSU 센서의 한계 | 시나리오당 RSU 센서 1벌을 CAV 3~4대가 공유 → 센서만으로는 어느 차가 UE인지 식별 불가 | [근거] `MULTIMODAL_ARCHITECTURES.md` §5.0-2 |
| 위치 정밀도 | predicted_ego_pos xy 오차 중앙값 0.39~0.41 m(≈37 λ @28 GHz), GPS 1.27 m | [근거] STEP3 §3-2 |
| 자원 규칙 | GPU 0·1만, 동시 run ≤ 2, launch는 계획 승인 후 | [근거] 메모리 `feedback_gpu_0_1_only`, `feedback_experiment_plan_gate` |

## 2. 설계 결정(제안)

1. **기반 모델 = chiron**(`models/chiron_channel.py`, 19.2 M). 이유: S1 상위권이면서 run당 5.6 h로 반복 실험이 가능하고, 공용 `ChannelPredictionHead`·저장소 융합 블록(`fusion_blocks.py`)과 토큰 차원(256)이 맞는다 [근거: `MODEL_ARCHITECTURES.md` §2.7, `MULTIMODAL_ARCHITECTURES.md` §5.1]. LWM v1.1 구조(2.9 M)는 성능은 최고급이나 run당 30 h라 절제 반복에 부적합하고, NOVA는 토큰 2,048개라 융합 비용이 4배다 [추측: 비용 계산은 §5.1 추정].
2. **융합 = `GatedCrossModalFusion`**(게이트 g→0이면 채널 토큰이 그대로 통과 → channel_only 함수를 포함하므로 센서가 무용해도 성능이 깎일 구조적 이유가 없고, g 자체가 진단값) [근거: `MULTIMODAL_ARCHITECTURES.md` §1.2·§5.1]. 프레임별 인과 마스크가 필요하면 새 파일에 서브클래스(기존 코드 무수정).
3. **센서 = RSU만**, 시간 정렬은 프레임 인덱스 그대로(delta_t 불필요) [근거: §5.0-1].
4. **판정은 시드 편차 대비**: 먼저 chiron seed 0/1을 돌려 σ_seed를 얻는다(§3 M0). 단일 시드 차이 0.2~0.5 dB는 지금 상위 3개 모델 간 격차와 같은 크기라 시드 없이는 판정 불가 [근거: S1 표; 추측: σ 크기].

## 3. 단계별 계획

### M0. 사전 검증 (GPU 11 h + CPU)
| # | 작업 | 산출 | 비용 |
|---|---|---|---|
| M0-1 | chiron seed 0, seed 1 학습(레시피 동일) → σ_seed(지평별 median·pooled) | `outputs_cp/s2_chiron_lr3e-4_s{0,1}` | 5.6 h × 2 [근거: S1 chiron 소요] |
| M0-2 | 위치 좌표 변환 재검증: CAV predicted/true 위치를 RSU 로컬로 바꿔 atan2 각과 빔 라벨(`derived/beam_labels_paper`)의 상관을 채널 예측 index(`derived_cp/*_index.json`의 pos)로 재확인 | 스크립트+로그(`scripts/11_position_check.py`) | CPU 수 분 [추측] |
| M0-3 | 캐시 커버리지: 53,800 프레임 전부에 `feat_patch_rsu224`·`feat_pp_rsu` 항목이 있는지, split B1 val 장면 포함 여부 | 로그 | CPU 수 분 |
| M0-4 | 센서 로더 `cp/cp_sensor_data.py` 구현: 창(16프레임)마다 (a) RSU ViT 패치 [16,196,768] fp16, (b) PointPillars [16,384,100,176] → 8×8 평균 풀링 캐시 [16,64,384](사전 생성, 약 49 KB/프레임 → 전체 약 2.6 GB [추측]), (c) 레이더 json → (az, depth) 64×64 래스터 2채널(점유·속도) 캐시, (d) CAV yaml → 위치·속도·주변 차량 벡터. B=2 프로브로 shape 확인 | 코드+프로브 로그 | 구현 1일, 캐시 생성 CPU 수 시간 [추측] |

### M1. 안 (b) 위치 경유 — 값싼 기하 정보부터 (GPU 약 28 h)
구조: chiron 패치 토큰 [B,16,32,256]에 `EgoStateEncoder(10→256)` 출력 [B,16,1,256]을 브로드캐스트 가산((i), 추가 파라미터 ≈ 0.14 M) [근거: §5.2]. 미래 기하 p_t + v·h를 헤드 질의 4개에 가산(선택).

| run | 위치 입력 | 목적 |
|---|---|---|
| M1-a | `true_ego_pos`(오라클) | 상한. 여기서도 증분 0이면 기하 정보 자체가 무용 → M2로 갈 근거 약화 |
| M1-b | `predicted_ego_pos` | 현실 입력(1순위) |
| M1-c | `GPS` | 정밀도 대조 |
| M1-d | true + 가우시안 σ=2 m | 정밀도 의존성(선택) |
| M1-e | M1-b + 주변 차량 토큰(`vehicles/<id>` → object_encoder, 융합 (iii)) | 차폐 기하(선택, +3.7 M) |

절제(학습 없음, ckpt 재평가): real / zero / shuffle(배치 순열) / foreign(다른 시나리오 창의 위치) + `eval_sanity.py`(hist_shuffle 등). run 당 = epoch 0.69 h(broadcast)·0.79 h(token) × 약 13 epoch(chiron 의 best 8 + patience 5 가정) ≈ 9~10 h [근거: 실측 epoch, 추측: epoch 수] → 5 run 약 46 h, 절제 1 h.

**게이트 G1**: M1-a·b의 증분 Δ(median dB, 지평별)가 |Δ| > 2σ_seed 이고 zero/shuffle/foreign에서 real 대비 ≥ 0.5 dB 나빠지면 "기하 기여 있음" → M2·M3 진행. 아니면 M2는 레이더 1 run만 하고 종료(권고) [추측: 임계값은 사용자 확정 필요].

### M2. 안 (a) 센서 직접 입력 — 캐시 재사용 (GPU 약 30 h)
구조: chiron 백본 출력 [B,512,256]을 질의, 센서 토큰을 키/값으로 `GatedCrossModalFusion` 3층 [근거: §5.1]. 센서 토큰 = 카메라(ViT 패치 196 → `cam_attn` 요약 1개/프레임, [B] 재현 `rgb_proj`·`cam_attn` 재사용) + LiDAR(풀링 64 → 학습 질의 16개) + 레이더(래스터 → 소형 CNN 1개/프레임) → 프레임당 최대 18토큰 × 16 = 288 kv 토큰. 추가 파라미터 ≈ 2.1 M(카메라 0.41 + LiDAR 0.37 + 레이더 0.65 + 융합 3층 ≈ 0.6) [추측: §5.1 계산 기준].

| run | 센서 | 비고 |
|---|---|---|
| M2-cam | channel + camera(ViT 패치 캐시) | [B] 재현에서 기여 0이었던 입력 [근거: §4.1] |
| M2-lidar | channel + LiDAR(PointPillars 8×8 풀링 캐시) | 위와 같음 |
| M2-radar | channel + radar(래스터 64×64, FOV 110°) | 유일하게 다른 데이터셋에서 신호가 있던 센서 [근거: §4.3] |
| M2-cam+lidar | channel + camera + LiDAR | **[B] 논문·재현의 full 과 같은 구성** → 빔 예측 결과와 직접 대조 (v2 추가) |
| M2-all | channel + camera + LiDAR + radar | 전체 + 게이트·어텐션(토큰당 정규화)으로 모달별 기여 진단 |

절제: real / zero / shuffle / foreign / sensor-only(채널 0) + 게이트·어텐션 진단 + 장면·속도별 분해 [근거: §5.3 표 1~8]. run 당 = epoch 0.81~0.94 h × 약 13 epoch ≈ 10.5~12 h [근거: 실측 epoch] → 5 run 약 57 h, 절제 2 h. 게이트가 0 초기화라 학습 시작점은 채널 전용 chiron 과 동일 함수(실측 max|diff| 0.0).

### M3. 확장(조건부, G1·G2 통과 시에만)
증분이 확인된 구성만 LWM v1.1 구조(30 h/run)·NOVA(20 h/run)에 이식해 "센서 증분이 백본과 무관한지" 확인. 2 run 50 h [추측].

## 4. 판정 규칙(제안, 사용자 확정 필요)

> **수식** (2026-09-08 추가, 코드 정의 그대로; 전체 모음 `FORMULAS.md` §10·11)

**§10 시드 편차·판정 규칙 — `EXPERIMENT_PLAN_MULTIMODAL_20260907.md` §4, `REPRO_V2` 9/8 04:57**

$$\sigma_{seed}(h)=\operatorname{std}_{s\in\{42,0,1\}}\big[\text{median\_dB}_s(h)\big]\ (\text{ddof}=1)=0.25/0.24/0.20/0.22\ \text{dB},\qquad \Delta_h=\text{median\_dB}_{MM}(h)-\overline{\text{median\_dB}}_{\text{chiron}}(h)$$
"센서 기여 있음" $\iff |\Delta_h|>2\sigma_{seed}(h)$ **이고** zero·shuffle·foreign 세 절제에서 real 대비 $\ge0.5$ dB 악화 **이고** 정직성 검사 통과. 앞 조건만 만족하면 잡음·재분포, 뒤 조건만 만족하면 공동적응(의존은 있으나 기여 없음)으로 분류.

**§11 정직성·절제 조건 — `cp/eval_sanity.py`, `cp/eval_sensor_ablation_cp.py`**

hist_shuffle: $\mathbf X_b\leftarrow\mathbf X_{\pi(b)}$ (타깃 유지) → 0 dB 보다 나빠야 정상; last_zero: $X_{K_h}\leftarrow0$; hist_only_last: $X_{1..K_h-1}\leftarrow0$. 센서 절제: zero $\mathbf s\leftarrow0$, shuffle $\mathbf s_b\leftarrow\mathbf s_{\pi(b)}$(창 단위 derangement), foreign $\mathbf s_b\leftarrow\mathbf s_{b'}$($b'$ 는 다른 시나리오 창), sensor_only $\mathbf X\leftarrow0$, none = 융합 블록 미실행.

- 증분 Δ_h = median NMSE dB(멀티모달) − median NMSE dB(채널 전용 chiron, 같은 seed), h = 10/20/30/40 ms. pooled 평균도 병기(지표 간 순위 불일치가 이미 관측됨 [근거: 9/7 02:25 항목]).
- "센서 기여 있음" = (i) |Δ_h| > 2σ_seed(M0-1) **이고** (ii) zero·shuffle·foreign 3조건 모두에서 real 대비 ≥ 0.5 dB 악화(센서 **내용**에 의존) **이고** (iii) 정직성 검사 통과. (i)만 만족하고 (ii) 불만족이면 "학습 잡음·정규화 효과"로 분류 [추측: 0.5 dB 임계값].
- 발표·게이트에는 내부 기준선(copy-last 등)을 쓰지 않는다 [근거: 메모리 `feedback_baselines_paper_only`]. 채널 예측에는 논문 기준값이 없으므로 "채널 전용 동일 모델 대비 증분"만 보고한다.

## 5. 규모 표

| 단계 | run 수 | GPU 시간(실측 epoch × 13 ep) | 벽시계(2 GPU) | 선행 조건 |
|---|---|---|---|---|
| M0 | 2 + CPU | 10 h(chiron seed 0/1 각 5.2 h) | 0.3일 | M0-2·3·4 는 **완료**(9/7: 좌표 xy 점검·커버리지·로더/캐시) |
| M1 | 5 (+절제) | 47 h | 1.0일 | M0 |
| M2 | 5 (+절제) | 59 h | 1.3일 | M0(캐시 완료) |
| M3(선택) | 2 | 50 h | 1일 | G1·G2 통과 |
| 합계(M0~M2) | 12 | **116 h** | **약 2.5일** | 구현은 완료, 남은 코드 = 절제 평가 스크립트(`cp/eval_sensor_ablation_cp.py`, 반나절) |

## 6. 산출물·기록
- 코드: **완료** `cp/cp_multimodal.py`(`mm:chiron`, `--sensors cam,lidar,radar,pos --pos_source predicted|true|gps --pos_mode token|broadcast --fuse_layers 3 --causal_mask --backbone_init <ckpt>`), `cp/cp_sensor_data.py`(로더·캐시 생성), `scripts/13_probe_multimodal_cp.py`, `train_cp.py` 인자 7개(기본값에서 기존 동작 동일, 무회귀 확인). **미구현**: `cp/eval_sensor_ablation_cp.py`(real/zero/shuffle/foreign/sensor-only + 게이트·어텐션 진단) — M0 와 병행 구현.
- 기록: run마다 `REPRO_V2_EXPERIMENTS_SINCE_20260829.md`에 즉시 기재, 원장 `EXPERIMENT_LOG_CHANNEL_PRED.md` 자동 갱신, 종료 시 `REPORT_MULTIMODAL_RESULT.md`.

## 7. 리스크·미결
1. RSU 센서가 UE를 식별하지 못하므로 안 (a)의 상한이 구조적으로 낮을 수 있다 [근거: §5.0-2]. 이 경우 M2 결과 0은 "센서 정보 부재"가 아니라 "연관(association) 부재"일 수 있어 해석에 주의.
2. 좌표 변환(CARLA→RSU 로컬→Sionna 배열 좌표)은 재검증되지 않았다 → M0-2 필수 [근거: §5.2].
3. (해소) 레이더 래스터·LiDAR 풀링·위치 캐시는 9/7 생성 완료(1.09 GB, 커버리지 100 %).
4. 로더 I/O: 창 단위 LRU 로 정상상태 52 ms/배치(B=32) 실측 → epoch 당 +0.31 h. 전량 RAM 적재(5.9 GiB)나 프리페치 워커로 더 줄일 수 있음 [추측]. RAM 압박 조건은 미측정.
5. 시드 편차가 크면(σ ≥ 0.5 dB) 모든 판정이 무력화됨 → M0-1 결과에 따라 seed 3개(42/0/1) 평균으로 전환 [추측].
6. 승인 필요 항목: 기반 모델(chiron 유지 여부), 임계값(2σ·0.5 dB), 레이더 포함 여부, M1→M2 게이트 적용 여부, GPU 범위(0·1).
