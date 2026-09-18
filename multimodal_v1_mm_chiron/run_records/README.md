# run_records — 최근 학습 run 33개의 기록 (가중치 제외)

원본: `mmw_reproduction/outputs_cp/<run>/`. 각 폴더에 `config.json`(인자), `result.json`(best epoch val 지표), `metrics.csv`(epoch별 train/val 손실), `sensor_ablation.json`(mm run 만: real/zero/shuffle/foreign/sensor_only/hist_shuffle), `train.log`, `DONE`.
`best.pt`(run 당 94.5 MB, 합계 2.3 GB)는 git 에 넣지 않았다 — 필요하면 GitHub Release 자산으로 별도 배포.

지표: val NMSE dB. pooled = Σerr/Σpower (논문 표준 dB(mean) 대응), median = 표본 median. h1 = 10 ms 지평, h4 = 40 ms. 절제 Δ = (조건 − real) pooled h1, 양수 = 나빠짐, 0 근처 = 센서 내용 무관.
분할: B1 = 시나리오 홀드아웃(val 3 장면), T1 = 시간 기준 80/20 (G=20). `_s0`/`_s1` = seed 0/1, 접미 없음 = seed 42. `zerocache` = 레이더 캐시를 0 으로 바꾼 대조군. `repo:chiron` = 채널 전용 기준선(같은 백본, 융합 없음).

| run | model | sensors | pos_mode | split | seed | params | best ep | pooled h1 | pooled h4 | median h1 | Δzero | Δforeign |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mm_cam | mm:chiron | cam | - | B1 | 42 | 23,176,960 | 15 | -11.87 | -9.31 | -16.86 | +0.17 | +0.04 |
| mm_cam_lidar | mm:chiron | cam,lidar | - | B1 | 42 | 23,543,296 | 5 | -11.67 | -8.99 | -15.69 | +0.31 | +0.01 |
| mm_cam_lidar_radar | mm:chiron | cam,lidar,radar | - | B1 | 42 | 23,615,968 | 16 | -12.11 | -9.45 | -17.75 | -0.07 | -0.04 |
| mm_cam_lidar_radar_s0 | mm:chiron | cam,lidar,radar | - | B1 | 0 | 23,615,968 | 20 | -12.33 | -9.58 | -17.81 | -0.05 | -0.01 |
| mm_cam_lidar_s0 | mm:chiron | cam,lidar | - | B1 | 0 | 23,543,296 | 10 | -11.76 | -8.98 | -16.20 | -0.04 | +0.03 |
| mm_cam_s0 | mm:chiron | cam | - | B1 | 0 | 23,176,960 | 14 | -11.98 | -9.20 | -16.66 | +0.35 | +0.09 |
| mm_lidar | mm:chiron | lidar | - | B1 | 42 | 23,082,496 | 9 | -11.73 | -9.06 | -16.34 | -0.02 | -0.01 |
| mm_lidar_s0 | mm:chiron | lidar | - | B1 | 0 | 23,082,496 | 12 | -11.65 | -8.98 | -16.45 | +0.00 | -0.00 |
| mm_pos_gps_bc | mm:chiron | pos | broadcast | B1 | 42 | 19,230,464 | 8 | -11.14 | -8.46 | -15.23 | +2.62 | +4.05 |
| mm_pos_gps_bc_s0 | mm:chiron | pos | broadcast | B1 | 0 | 19,230,464 | 20 | -10.95 | -8.42 | -16.03 | +3.33 | +4.76 |
| mm_pos_pred_bc | mm:chiron | pos | broadcast | B1 | 42 | 19,230,464 | 15 | -11.42 | -8.57 | -16.16 | +2.15 | +4.20 |
| mm_pos_pred_bc_s0 | mm:chiron | pos | broadcast | B1 | 0 | 19,230,464 | 14 | -11.49 | -8.65 | -16.31 | +2.68 | +4.28 |
| mm_pos_pred_token | mm:chiron | pos | token | B1 | 42 | 22,784,768 | 13 | -11.59 | -9.18 | -16.70 | +5.28 | +6.47 |
| mm_pos_pred_token_s0 | mm:chiron | pos | token | B1 | 0 | 22,784,768 | 10 | -11.91 | -9.31 | -16.68 | +8.03 | +7.65 |
| mm_pos_true_bc | mm:chiron | pos | broadcast | B1 | 42 | 19,230,464 | 8 | -11.68 | -8.76 | -15.92 | +2.00 | +3.64 |
| mm_pos_true_bc_s0 | mm:chiron | pos | broadcast | B1 | 0 | 19,230,464 | 13 | -11.37 | -8.44 | -15.96 | +3.04 | +4.99 |
| mm_radar | mm:chiron | radar | - | B1 | 42 | 22,788,832 | 23 | -12.25 | -9.70 | -18.31 | +1.29 | +0.04 |
| mm_radar_s0 | mm:chiron | radar | - | B1 | 0 | 22,788,832 | 20 | -12.26 | -9.72 | -18.23 | +2.58 | +0.00 |
| mm_radar_zerocache | mm:chiron | radar | - | B1 | 42 | 22,788,832 | 7 | -11.77 | -8.95 | -16.18 | +0.00 | +0.00 |
| mm_radar_zerocache_s0 | mm:chiron | radar | - | B1 | 0 | 22,788,832 | 8 | -11.94 | -9.16 | -16.41 | +0.00 | +0.00 |
| t1_mm_cam | mm:chiron | cam | - | T1 | 42 | 23,176,960 | 14 | -10.50 | -7.67 | -13.35 | +0.03 | +0.01 |
| t1_mm_cam_lidar | mm:chiron | cam,lidar | - | T1 | 42 | 23,543,296 | 18 | -10.65 | -7.76 | -13.54 | +0.04 | +0.04 |
| t1_mm_cam_lidar_radar | mm:chiron | cam,lidar,radar | - | T1 | 42 | 23,615,968 | 16 | -10.81 | -8.08 | -14.03 | +0.01 | +0.01 |
| t1_mm_lidar | mm:chiron | lidar | - | T1 | 42 | 23,082,496 | 12 | -10.57 | -7.71 | -13.51 | -0.00 | -0.00 |
| t1_mm_pos_pred_token | mm:chiron | pos | token | T1 | 42 | 22,784,768 | 11 | -10.43 | -8.10 | -13.54 | +3.95 | +5.07 |
| t1_mm_radar | mm:chiron | radar | - | T1 | 42 | 22,788,832 | 16 | -10.79 | -7.88 | -13.98 | +0.11 | +0.03 |
| t1_mm_radar_zerocache | mm:chiron | radar | - | T1 | 42 | 22,788,832 | 13 | -10.42 | -7.59 | -13.30 | +0.00 | +0.00 |
| s1_chiron_lr3e-4 | repo:chiron | - | - | B1 | 42 | 19,156,736 | 8 | -11.59 | -9.00 | -16.16 | - | - |
| s2_chiron_lr3e-4_s0 | repo:chiron | - | - | B1 | 0 | 19,156,736 | 9 | -11.51 | -8.66 | -15.90 | - | - |
| s2_chiron_lr3e-4_s1 | repo:chiron | - | - | B1 | 1 | 19,156,736 | 13 | -11.58 | -8.80 | -16.39 | - | - |
| t1_chiron_s0 | repo:chiron | - | - | T1 | 0 | 19,156,736 | 17 | -10.48 | -7.53 | -13.43 | - | - |
| t1_chiron_s1 | repo:chiron | - | - | T1 | 1 | 19,156,736 | 17 | -10.15 | -7.29 | -12.82 | - | - |
| t1_chiron_s42 | repo:chiron | - | - | T1 | 42 | 19,156,736 | 18 | -10.76 | -7.75 | -13.71 | - | - |
