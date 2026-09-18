# 수식 정리 — 채널 합성·정규화·손실·지표·기준선·나이퀴스트·센서 특징·융합·판정 (한눈에)

- 작성 2026-09-08. 모든 식은 **실제 코드의 정의를 그대로 옮긴 것**이며 각 식 뒤에 코드 위치를 적었다. 표기: LaTeX(`$$`)와 그 아래 같은 식의 텍스트판을 병기(뷰어가 수식을 렌더링하지 못해도 읽을 수 있게).
- 기호: 프레임 $t$(10 ms 격자), RX 행 $r=1..16$, TX 안테나 $n=1..N_t$ ($N_t=64$), 부반송파 $k=0..K-1$ ($K=64$), 경로 $l=1..P$, 이력 길이 $K_h=16$, 예측 지평 $H=4$ ($h=1..4$ ↔ 10/20/30/40 ms).

## 1. 채널 합성 (데이터셋 경로 파라미터 → 주파수 응답) — `cp/prep_h_memmap.py`

$$H_t[r,n,k]=\sum_{l=1}^{P} a_{t,l}[r,n]\,e^{-j2\pi f_k \tau_{t,l}},\qquad f_k=\left(k-\frac{K}{2}\right)\Delta f,\quad K=64,\ \Delta f=120\ \text{kHz}$$

텍스트: `H_t[r,n,k] = Σ_l a_l[r,n] · exp(−j2π f_k τ_l)`, `f_k = (k − K/2)·Δf` → 대역 $K\Delta f = 7.68$ MHz. $a$ 는 Sionna 경로 이득(복소, `a[1,1,Nr,1,Nt,P,1]`), $\tau$ 는 LOS 기준 정규화 지연(데이터셋 규약: LOS $\tau=0$, 절대 위상 제거 — `REPORT_STEP1` §1.3, `REPORT_STEP2` §2.4(c)). $\tau<0$ 더미 경로($a=0$)는 무시. 저장은 fp16 `[frame, rx, tx, subcarrier, re/im]`.

## 2. 창·표본·정규화 — `cp/cp_data.py::make_windows`, `WindowSet.batch`

입력·타깃(RX 행 $r$ 하나가 표본 1개):
$$\mathbf X_b=\{H_{t-K_h+1},\dots,H_t\}[r]\in\mathbb C^{K_h\times N_t\times K},\qquad \mathbf Y_b=\{H_{t+1},\dots,H_{t+H}\}[r]\in\mathbb C^{H\times N_t\times K}$$

창별 RMS 정규화(**이력 $\mathbf X$ 만으로** 계산, 타깃에는 같은 값을 나눔 → 미래 정보 누수 없음):
$$s_b=\sqrt{\frac{1}{K_h N_t K}\sum_{\kappa,n,k}\big|X_b[\kappa,n,k]\big|^2},\qquad \tilde{\mathbf X}_b=\mathbf X_b/s_b,\quad \tilde{\mathbf Y}_b=\mathbf Y_b/s_b$$

텍스트: `scale = sqrt(mean over (K_h, Nt, K) of |X|²)`; `X = X/scale`, `Y = Y/scale` (코드 line 55–56). 분할 B1: val = {Town05_ringroad, Town03_gastation, Town10_crossroad} 장면 전체(궤적 9), train = 나머지 13 장면(궤적 41); 창은 궤적 안에서만 만들고(경계 불초과) train stride 1, val stride $H=4$.

## 3. 학습 손실 — `cp/train_cp.py::loss_fn`

$$\mathcal L=\frac{1}{B}\sum_{b=1}^{B}\frac{\sum_{h=1}^{H}\big\|\hat{\mathbf Y}_{b,h}-\tilde{\mathbf Y}_{b,h}\big\|_2^2}{\sum_{h=1}^{H}\big\|\tilde{\mathbf Y}_{b,h}\big\|_2^2}$$

텍스트: `loss = mean_b [ Σ_h ||pred_h − Y_h||² / Σ_h ||Y_h||² ]` (4지평 합산 NMSE, 실수·허수 합산 norm). 최적화 AdamW($\lambda_{wd}=10^{-4}$), warmup 1 epoch 뒤 cosine, gradient clip 1.0, AMP 없음. 조기종료·best 선정 기준: $v_{sel}=\frac14\sum_{h}\text{pooled\_dB}(h)$, patience 5 (`train_cp.py` line 99–109).

## 4. 평가 지표 — `cp/cp_data.py::nmse_per_sample`, `summarize`

표본 $b$·지평 $h$ 별 NMSE(복소 채널 $\hat Y,Y\in\mathbb C^{N_t\times K}$):
$$\text{NMSE}_{b,h}=\frac{\|\hat Y_{b,h}-Y_{b,h}\|_F^2}{\|Y_{b,h}\|_F^2},\qquad \text{dB}=10\log_{10}(\cdot)$$

- **median dB** (리포트의 주 지표): $\operatorname{median}_b\,10\log_{10}\text{NMSE}_{b,h}$ — 표본별 dB 의 중앙값.
- **pooled dB** (에너지 가중, 조기종료·선정 기준): $10\log_{10}\dfrac{\sum_b\|\hat Y_{b,h}-Y_{b,h}\|^2}{\sum_b\|Y_{b,h}\|^2}$.
- **정렬 상관·정렬 NMSE**: $\rho_{b,h}=\dfrac{\left|\langle \hat Y_{b,h},Y_{b,h}\rangle\right|}{\|\hat Y_{b,h}\|\,\|Y_{b,h}\|}$ (복소 내적의 크기), $\text{NMSE}^{align}_{b,h}=1-\rho_{b,h}^2$ — 복소 스칼라(크기·위상) 정합 후 잔여 오차.
- 장면별·속도별: 같은 median 을 val 장면 3개, 속도 구간 {0–1, 1–4, 4–6, 6–9} m/s(이력 마지막 프레임의 속도)로 나눠 계산.

텍스트: `raw = ||pred−Y||²/||Y||²`, `median_db = median(10log10 raw)`, `pooled_db = 10log10(Σerr/Σpw)`, `rho = |<pred,Y>|/(||pred||·||Y||)`, `align = 1 − rho²`.

## 5. 내부 진단 기준선 — `cp/baselines_g0.py` (발표·판정에는 미사용)

- copy-last: $\hat Y_h=X_{K_h}$ (마지막 이력 프레임 복사).
- 선형 $K_h'$차 ($K_h'\in\{1,2,4\}$, 실계수, 전 안테나·부반송파 공유): $\hat Y_h=\sum_{m=1}^{K_h'} c_{h,m}\,X_{K_h-m+1}$, 계수는 train 창의 최소제곱 $(\mathbf G+10^{-6}\mathbf I)\,\mathbf c_h=\mathbf r_h$ (Gram 행렬 $\mathbf G$, 상관 벡터 $\mathbf r_h$).
- 0 예측: $\hat Y_h=0$ → NMSE = 0 dB (정직성 검사의 기준점).

## 6. 나이퀴스트·위상 회전 — `scripts/08_nyquist_check.py`, `REPORT_STEP2` §2.3(a)

$$\lambda=\frac{c}{f_c}=\frac{3\times10^8}{28\times10^9}\approx1.07\ \text{cm},\qquad f_{D,\max}\Delta t=\frac{v\,\Delta t}{\lambda},\qquad v_{\text{Nyq}}=\frac{0.5\,\lambda}{\Delta t}\approx0.54\ \text{m/s}\ (\Delta t=10\ \text{ms})$$

절대 위상 기준으로는 프레임 간 회전이 0.5 사이클을 넘는 프레임이 91.5~95.3 %(표본 추출 불충분)이나, 데이터셋이 LOS 기준으로 위상을 정규화해 **저장된 채널에 남는 것은 경로별 상대 지연 변화** $f_c\,\Delta\tau_l$ 뿐이며 그 중앙값은 0.108 사이클/프레임(0.5 초과 경로 33.5 %, 전력 가중 3.8 %) — 예측 가능성은 이 상대 회전이 결정.

## 7. 토큰화 — `MODEL_ARCHITECTURES.md` §3, `cp/cp_lwm11.py::patchify`

패치 $p_h\times p_w$ (안테나 × 부반송파)일 때 프레임당 토큰 수 $n_{tok}=\dfrac{N_t}{p_h}\cdot\dfrac{K}{p_w}$, 표본당 $K_h\,n_{tok}$:
chiron $4\times32\to32$ (512/표본), NOVA $4\times8\to128$ (2,048), LWM v1.1 $4\times4\to256(+\text{CLS})$ (4,112), 프레임 통째(Transformer·LSTM·Mamba·LWM 구조) $\to1$ (16). LWM v1.1 패치 벡터(32값) = 행(안테나 4)마다 열(부반송파 4)의 $(\Re,\Im)$ 인터리브, [CLS] $=0.2\cdot\mathbf 1_{32}$.

## 8. 센서 특징 — `cp/cp_sensor_data.py`

위치(CAV yaml → RSU 로컬, `REPRO_V2_PREREQUISITES` §5.3 규약, z 축 미검증):
$$\mathbf p'=R_z(-\psi_{rsu})\,(\mathbf p-\mathbf p_{rsu}),\qquad \mathbf f=\big[\,dx,\,dy,\,dz,\;r=\|\mathbf p'\|,\;\sin\varphi,\;\cos\varphi,\;v_x,\,v_y,\,v_z,\;\|\mathbf v\|\,\big],\quad \varphi=\operatorname{atan2}(y',x')$$
모델 입구에서 고정 상수로 나눔 `POS_SCALE=[100,100,100,100,1,1,10,10,10,10]` (m→/100, m/s→/10). $\varphi$ 는 빔각과 같은 기준(AoD 대비 median 0.003°).

레이더 래스터(프레임당 검출 $\{v_i,\text{az}_i,\text{depth}_i\}$):
$$\text{bin}_r=\left\lfloor \frac{\text{depth}_i}{120\ \text{m}}\cdot64\right\rfloor,\quad \text{bin}_{az}=\left\lfloor\frac{\text{az}_i+\text{FOV}/2}{\text{FOV}}\cdot64\right\rfloor\ (\text{FOV}=110^\circ),\qquad C_0[\text{bin}]=\#\{i\},\ C_1[\text{bin}]=\operatorname{mean}_i v_i$$
모델 입구에서 $C_0\leftarrow\log(1+C_0)$. 카메라 = ViT 패치 캐시 $[196,768]$, LiDAR = PointPillars BEV $[384,100,176]$ → adaptive avg-pool $8\times8$ → $[64,384]$.

## 9. 게이트 cross-attention 융합 — `cp/cp_multimodal.py::GatedXAttnBlock`

채널 토큰 $\mathbf x\in\mathbb R^{512\times256}$, 센서 토큰 $\mathbf s\in\mathbb R^{n_s\times256}$:
$$\mathbf q=\text{LN}(\mathbf x),\ \mathbf k=\text{LN}(\mathbf s),\qquad \mathbf a=\text{MHA}(\mathbf q,\mathbf k,\mathbf k),\qquad \mathbf g=\sigma\!\big(W_g[\mathbf x;\mathbf a]+\mathbf b_g\big),\qquad \mathbf y=\text{FFN}_{\text{SwiGLU}}(\mathbf x+\mathbf g\odot\mathbf a)$$
초기화: $W_g,\mathbf b_g$, MHA out-proj, FFN $w_3$ 를 0 → $\mathbf a=0,\ \mathbf g=\sigma(0)=0.5,\ \mathbf y=\mathbf x$ (학습 시작점 = 채널 전용 chiron, 실측 max|diff| 0). 3층 반복 후 기존 `ChannelPredictionHead`(학습 질의 4개 ↔ 512 토큰 cross-attention → MLP). pos broadcast 모드는 대신 $\text{tokens}[b,\kappa,:,:]\mathrel{+}=\text{MLP}(\mathbf f_\kappa)$ 를 백본 입구에서 가산.

## 10. 시드 편차·판정 규칙 — `EXPERIMENT_PLAN_MULTIMODAL_20260907.md` §4, `REPRO_V2` 9/8 04:57

$$\sigma_{seed}(h)=\operatorname{std}_{s\in\{42,0,1\}}\big[\text{median\_dB}_s(h)\big]\ (\text{ddof}=1)=0.25/0.24/0.20/0.22\ \text{dB},\qquad \Delta_h=\text{median\_dB}_{MM}(h)-\overline{\text{median\_dB}}_{\text{chiron}}(h)$$
"센서 기여 있음" $\iff |\Delta_h|>2\sigma_{seed}(h)$ **이고** zero·shuffle·foreign 세 절제에서 real 대비 $\ge0.5$ dB 악화 **이고** 정직성 검사 통과. 앞 조건만 만족하면 잡음·재분포, 뒤 조건만 만족하면 공동적응(의존은 있으나 기여 없음)으로 분류.

## 11. 정직성·절제 조건 — `cp/eval_sanity.py`, `cp/eval_sensor_ablation_cp.py`

hist_shuffle: $\mathbf X_b\leftarrow\mathbf X_{\pi(b)}$ (타깃 유지) → 0 dB 보다 나빠야 정상; last_zero: $X_{K_h}\leftarrow0$; hist_only_last: $X_{1..K_h-1}\leftarrow0$. 센서 절제: zero $\mathbf s\leftarrow0$, shuffle $\mathbf s_b\leftarrow\mathbf s_{\pi(b)}$(창 단위 derangement), foreign $\mathbf s_b\leftarrow\mathbf s_{b'}$($b'$ 는 다른 시나리오 창), sensor_only $\mathbf X\leftarrow0$, none = 융합 블록 미실행.
