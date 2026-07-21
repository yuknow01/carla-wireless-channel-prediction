#!/usr/bin/env python3
"""Build self-contained interactive HTML explanations for the active models."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "visualizations"
EVIDENCE = json.loads((Path(__file__).with_name("visualization_experiment_data.json")).read_text())


CSS = r"""
:root{--bg:#07111e;--panel:#102038;--panel2:#0b192a;--line:#29445f;--text:#eff7ff;--muted:#9db2c8;--accent:#3dd9eb;--accent2:#5f8dff;--green:#58df9b;--orange:#ffad5c;--red:#ff7080}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(circle at 48% -12%,#183b62 0,#07111e 43%);color:var(--text);font-family:Inter,Pretendard,"Noto Sans KR",system-ui,sans-serif}main{max-width:1220px;margin:auto;padding:34px 25px 70px}.topnav{display:flex;justify-content:space-between;align-items:center;margin-bottom:28px}.home{color:var(--muted);text-decoration:none;border:1px solid var(--line);border-radius:10px;padding:8px 12px}.home:hover{color:var(--text);border-color:var(--accent)}.badge{font-size:12px;font-weight:850;color:var(--accent);background:color-mix(in srgb,var(--accent) 14%,transparent);border-radius:99px;padding:6px 11px}.hero{display:grid;grid-template-columns:1.35fr .65fr;gap:18px;align-items:stretch}.hero h1{font-size:38px;line-height:1.15;letter-spacing:-1px;margin:10px 0}.hero p{color:var(--muted);line-height:1.7;max-width:780px}.shape-card{background:linear-gradient(145deg,#132943,#0b192a);border:1px solid var(--line);border-radius:18px;padding:20px;display:flex;flex-direction:column;justify-content:center}.shape-card .shape{font:850 18px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--accent);line-height:1.65}.shape-card small{color:var(--muted);line-height:1.5}.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:18px}.fact{background:#0b192a;border:1px solid var(--line);border-radius:12px;padding:13px}.fact b{display:block;color:var(--accent);font-size:15px}.fact span{font-size:12px;color:var(--muted)}.card{margin-top:20px;background:linear-gradient(145deg,rgba(18,38,63,.97),rgba(9,22,38,.97));border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 14px 44px rgba(0,0,0,.2)}h2{font-size:20px;margin:0 0 16px}h3{margin:0 0 8px;font-size:17px}.pipeline{display:grid;grid-template-columns:repeat(var(--cols),1fr);gap:8px}.stage{position:relative;min-height:105px;padding:14px 10px;border:1px solid var(--line);border-radius:12px;background:#0a1828;color:var(--text);cursor:pointer;text-align:left;transition:.2s}.stage:hover,.stage.active{transform:translateY(-3px);border-color:var(--accent);box-shadow:0 0 0 1px color-mix(in srgb,var(--accent) 25%,transparent),0 10px 26px rgba(0,0,0,.24)}.stage .n{font:800 11px ui-monospace,monospace;color:var(--accent)}.stage strong{display:block;margin:8px 0 5px;font-size:14px}.stage code{font-size:11px;color:var(--muted);white-space:normal}.detail{display:grid;grid-template-columns:.75fr 1.25fr;gap:13px;margin-top:13px}.detail>div{background:#081522;border:1px solid var(--line);border-radius:12px;padding:16px}.detail .bigshape{font:850 17px ui-monospace,monospace;color:var(--accent);margin-top:9px}.detail p{color:var(--muted);line-height:1.65;margin:0}.interactive{min-height:285px}.control{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:4px 0 17px}.control input[type=range]{width:min(480px,65vw);accent-color:var(--accent)}button{border:1px solid var(--line);background:#12304a;color:var(--text);border-radius:9px;padding:8px 12px;font-weight:750;cursor:pointer}button:hover,button.active{border-color:var(--accent);color:var(--accent)}.readout{font:800 14px ui-monospace,monospace;color:var(--accent)}.tokens{display:grid;grid-template-columns:repeat(16,1fr);gap:5px}.token{height:46px;border:1px solid var(--line);background:#091725;border-radius:8px;display:grid;place-items:center;color:#637b94;font:11px ui-monospace,monospace;transition:.2s}.token.on{background:color-mix(in srgb,var(--accent) 20%,#0b192a);border-color:var(--accent);color:#dffcff}.token.now{background:var(--accent);color:#06121c;font-weight:900;transform:translateY(-3px);box-shadow:0 0 20px color-mix(in srgb,var(--accent) 45%,transparent)}.meter{height:13px;border:1px solid var(--line);border-radius:99px;background:#071421;overflow:hidden}.meter>div{height:100%;width:0;background:linear-gradient(90deg,var(--accent2),var(--accent));transition:.25s}.callout{margin-top:15px;padding:13px 15px;border-left:4px solid var(--accent);background:#091725;border-radius:9px;color:#c9dbea;line-height:1.6}.equation{font:750 15px ui-monospace,monospace;color:#d8ebff;background:#071421;border:1px solid var(--line);padding:14px;border-radius:11px;line-height:1.75}.grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.notes{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.note{padding:15px;border-radius:11px;background:#0a1828;border-top:3px solid var(--accent);color:#cbdced;line-height:1.6}.note b{display:block;color:var(--text);margin-bottom:5px}.table{width:100%;border-collapse:collapse;font-size:13px}.table td{border-bottom:1px solid var(--line);padding:10px}.table td:first-child{color:var(--muted);width:28%}.patch-grid{display:grid;grid-template-columns:repeat(2,1fr);grid-template-rows:repeat(4,58px);gap:7px;max-width:520px}.patch{border:1px solid var(--line);border-radius:9px;background:#0a1828;color:#9cb1c7;cursor:pointer}.patch:hover,.patch.active{border-color:var(--accent);background:color-mix(in srgb,var(--accent) 22%,#0a1828);color:#fff}.fusion-flow{display:grid;grid-template-columns:1fr 50px 1fr 50px 1fr;gap:8px;align-items:center}.flowbox{border:1px solid var(--line);border-radius:12px;background:#0a1828;padding:16px;text-align:center;min-height:118px;display:flex;flex-direction:column;justify-content:center}.flowbox strong{margin-bottom:8px}.flowbox code{color:var(--accent)}.arrow{text-align:center;color:var(--accent);font-size:26px}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:13px}.tabbody{padding:16px;border:1px solid var(--line);border-radius:12px;background:#081522;color:#cbdced;line-height:1.65}.warn{border-color:#733b4a;background:#2b1720;color:#ffd9df}.paper{background:#f7f9fc;color:#162235;border-radius:15px;padding:22px;border:1px solid #d7deea}.paper-flow{display:grid;grid-template-columns:repeat(9,auto);align-items:center;gap:7px}.pbox{min-height:102px;min-width:132px;border:1.7px solid #4b6f9a;border-radius:8px;background:#fff;padding:12px;text-align:center;display:flex;flex-direction:column;justify-content:center;box-shadow:0 3px 10px #18314f18}.pbox b{font-size:14px}.pbox code{font-size:11px;color:#365d8a;margin-top:7px;white-space:normal}.parrow{font-size:24px;color:#547aa5;text-align:center}.paper-supervision{display:grid;grid-template-columns:1fr 44px 1.3fr;align-items:center;gap:8px;margin-top:18px;padding-top:18px;border-top:1px dashed #aebbc9}.truth{border-color:#d48646}.compare{border:2px solid #cb5363;background:#fff6f7}.paper-caption{font-size:12px;color:#617086;line-height:1.55;margin-top:14px}.metric-flow{display:grid;grid-template-columns:1fr 42px 1fr 42px 1fr;gap:8px;align-items:center}.metric-flow .flowbox{background:#081522}.chart-tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}.chart-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.chartbox{background:#071421;border:1px solid var(--line);border-radius:12px;padding:12px}.chartbox h3{font-size:13px;color:#cbdced}.chartbox canvas{display:block;width:100%;height:210px}.protocol{font-size:12px;color:var(--muted);line-height:1.55;margin:10px 0}.results{margin-top:18px}.results h3{margin-top:18px}.table-scroll{max-width:100%;overflow-x:auto}.result-table{width:100%;min-width:820px;border-collapse:collapse;font-size:12.5px}.result-table th,.result-table td{padding:9px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}.result-table th:first-child,.result-table td:first-child{text-align:left}.result-table tr.focus{background:color-mix(in srgb,var(--accent) 12%,transparent)}.result-table tr.focus td:first-child{color:var(--accent);font-weight:850}.missing{padding:20px;border:1px dashed #6d8095;border-radius:12px;color:var(--muted);line-height:1.65}.footer{margin-top:24px;color:#6f869f;font-size:12px}.footer code{color:#9bb1c8}@media(max-width:900px){.hero,.detail,.grid2{grid-template-columns:1fr}.facts{grid-template-columns:repeat(2,1fr)}.pipeline{grid-template-columns:repeat(2,1fr)}.notes{grid-template-columns:1fr}.fusion-flow,.metric-flow{grid-template-columns:1fr}.arrow,.parrow{transform:rotate(90deg)}.tokens{grid-template-columns:repeat(8,1fr)}.paper-flow{grid-template-columns:1fr}.paper-supervision{grid-template-columns:1fr}.chart-grid{grid-template-columns:1fr}}
"""


MODELS = {
    "lstm": {
        "title": "LSTM · Recurrent Channel Predictor",
        "subtitle": "각 시점의 2,048차원 wideband CSI와 이전 hidden/cell state를 이용해 256차원 시간 표현을 순차적으로 생성합니다.",
        "accent": "#ffad5c",
        "input": "(B,16,16,64,2)", "output": "(B,4,16,64,2)",
        "facts": [("5.13 M", "parameters"), ("h=256", "hidden width"), ("3 layers", "uni-LSTM"), ("K=16", "history frames")],
        "stages": [
            ("CSI history", "(B,16,16,64,2)", "과거 16개 복소수 CSI 프레임입니다."),
            ("Flatten", "(B,16,2048)", "각 시점의 16×64×2 값을 하나의 2,048차원 벡터로 펼칩니다."),
            ("LSTM layer 1", "(B,16,256)", "현재 2,048차원 입력과 이전 hidden/cell state로 첫 번째 256차원 시간 표현을 만듭니다."),
            ("LSTM layers 2–3", "(B,16,256)", "상위 두 recurrent layer가 시간 표현을 다시 처리합니다. 시간 길이 16은 유지됩니다."),
            ("P=4 queries", "(B,4,256)", "미래 네 시점의 learnable query가 16개 LSTM 출력 전체를 cross-attention으로 참조합니다."),
            ("Residual head", "(B,4,16,64,2)", "MLP가 미래별 ΔH를 만들고 마지막 관측 CSI에 더해 최종 예측을 생성합니다."),
        ],
        "kind": "scan", "scan_dim": "256D",
        "scan_equation": "(hₜ,cₜ) = LSTM(xₜ,hₜ₋₁,cₜ₋₁)<br>hₜ ∈ ℝ²⁵⁶",
        "scan_message": "현재 2,048D CSI와 이전 hidden/cell state를 이용해 256D hₜ를 생성합니다.",
        "interaction_text": "시간 슬라이더를 움직이면 16개 CSI 프레임이 LSTM에 순서대로 입력되고 h₁…h₁₆이 쌓이는 과정을 확인할 수 있습니다.",
        "notes": [
            ("2,048→256의 의미", "reshape가 아니라 학습된 LSTM gate가 현재 CSI와 이전 상태를 결합해 256차원 특징을 생성합니다."),
            ("왜 전체 hidden sequence인가?", "마지막 h₁₆만 쓰지 않고 P=4 query가 h₁…h₁₆ 전체를 참조해 미래 시점별 문맥을 만듭니다."),
            ("Residual prediction", "전체 채널을 새로 생성하지 않고 H_last에 더할 변화량 ΔH를 학습합니다."),
        ],
        "rows": [("Input", "2,048 values/frame"), ("LSTM", "hidden 256 · 3 layers · dropout 0.1"), ("Output tokens", "16×256"), ("Prediction head", "P=4 queries · MLP hidden 512"), ("Initialization", "zero-init ΔH · copy-last start")],
    },
    "lwm": {
        "title": "LWM · Wideband Transformer",
        "subtitle": "한 시점의 전체 CSI를 하나의 시간 토큰으로 만들고 12-layer self-attention으로 과거 16개 토큰의 관계를 학습합니다.",
        "accent": "#b778ff",
        "input": "(B,16,16,64,2)", "output": "(B,4,16,64,2)",
        "facts": [("2.40 M", "parameters"), ("d=64", "backbone width"), ("12 layers", "self-attention"), ("8 heads", "attention heads")],
        "stages": [
            ("CSI history", "(B,16,16,64,2)", "과거 16개 복소수 CSI 프레임입니다."),
            ("Flatten", "(B,16,2048)", "각 시점의 16×64×2 값을 하나의 2,048차원 벡터로 펼칩니다."),
            ("Embedding", "(B,16,64)", "학습 Linear와 시간 위치 임베딩으로 각 프레임을 64차원 토큰으로 바꿉니다."),
            ("LWM ×12", "(B,16,64)", "8-head self-attention과 FFN(256)을 12회 적용합니다. 모든 토큰은 관측된 과거이므로 full attention이 미래 leakage를 만들지 않습니다."),
            ("Adapter", "(B,16,256)", "공통 예측 head와 멀티모달 fusion을 위해 64→256 projection과 LayerNorm을 적용합니다."),
            ("P-query head", "(B,4,16,64,2)", "미래 query 4개가 16개 시간 토큰을 참조해 ΔH를 만들고 마지막 CSI에 더합니다."),
        ],
        "kind": "depth", "depth": 12, "unit": "Transformer layer",
        "interaction_text": "슬라이더로 12개 attention layer가 순차적으로 적용되는 과정을 확인합니다. sequence length 16은 유지되고 표현만 반복해서 갱신됩니다.",
        "notes": [
            ("왜 wideband token인가?", "한 시점의 모든 안테나와 서브캐리어를 함께 보아 시간 변화 모델링에 집중합니다."),
            ("Adapter의 역할", "LWM 내부 폭은 64이지만 공통 head와 센서 branch는 256차원이므로 차원을 맞춥니다."),
            ("제한점", "안테나–서브캐리어의 2D 위치는 flatten 과정에서 명시적으로 보존되지 않습니다."),
        ],
        "rows": [("Input projection", "2,048 → 64"), ("FFN", "64 → 256 → 64"), ("Dropout", "0.1"), ("Prediction head", "P=4 queries · MLP hidden 512"), ("Initialization", "zero-init ΔH · copy-last start")],
    },
    "mamba": {
        "title": "Mamba · Selective State-Space Backbone",
        "subtitle": "2,048차원 wideband CSI를 128차원 시간 토큰으로 바꾸고, 선택적 state-space block이 과거를 순차적으로 압축합니다.",
        "accent": "#58df9b",
        "input": "(B,16,16,64,2)", "output": "(B,4,16,64,2)",
        "facts": [("2.18 M", "parameters"), ("d=128", "token width"), ("4 blocks", "Mamba depth"), ("state=16", "SSM state")],
        "stages": [
            ("CSI history", "(B,16,16,64,2)", "과거 16개 복소수 CSI 프레임입니다."),
            ("Flatten", "(B,16,2048)", "각 프레임을 2,048차원 wideband 벡터로 펼칩니다."),
            ("Embedding", "(B,16,128)", "Linear 2,048→128, LayerNorm, 학습 시간 위치 임베딩을 적용합니다."),
            ("Mamba ×4", "(B,16,128)", "각 block이 selective SSM과 local convolution을 사용해 입력에 따라 과거 정보의 유지·갱신을 조절합니다."),
            ("Final norm", "(B,16,128)", "4개 residual block 이후 LayerNorm을 적용합니다."),
            ("P-query head", "(B,4,16,64,2)", "4개 미래 query와 MLP hidden 512로 ΔH를 생성하고 copy-last에 더합니다."),
        ],
        "kind": "scan", "depth": 16, "unit": "time step", "scan_dim": "128D",
        "scan_equation": "stateₜ = SelectiveSSM(xₜ,stateₜ₋₁)<br>yₜ = projection(stateₜ,xₜ)",
        "scan_message": "현재 128D token과 이전 selective state를 이용해 정보를 유지·갱신합니다.",
        "interaction_text": "시간 슬라이더를 움직이면 Mamba가 과거 토큰을 왼쪽에서 오른쪽으로 읽으며 state를 갱신하는 개념을 볼 수 있습니다.",
        "notes": [
            ("Selective state", "고정된 평균이 아니라 현재 입력에 따라 어떤 과거 정보를 유지하거나 잊을지 조절합니다."),
            ("시간 복잡도", "attention의 K² 관계 대신 sequence scan을 사용해 긴 시퀀스에서 선형적인 시간 처리를 지향합니다."),
            ("중요한 범위", "이 모델은 CARLA용 소형 Mamba backbone이며 MambaCSP 논문의 전체 모델 구조와 동일하지 않습니다."),
        ],
        "rows": [("Embedding", "2,048 → 128"), ("Mamba", "depth 4 · state 16 · conv 4 · expand 2"), ("Dropout", "0.1"), ("Prediction head", "P=4 queries · MLP hidden 512"), ("Initialization", "zero-init ΔH · copy-last start")],
    },
    "dtcn": {
        "title": "DTCN · Delay-aware Temporal Convolution",
        "subtitle": "동일한 CSI를 주파수와 지연 두 좌표계로 임베딩하고, dilated causal convolution으로 시간 변화를 학습합니다.",
        "accent": "#3dd9eb",
        "input": "(B,16,16,64,2)", "output": "(B,4,16,64,2)",
        "facts": [("7.63 M", "parameters"), ("d=256", "token width"), ("6 blocks", "causal GLU TCN"), ("RF=37", "theoretical frames")],
        "stages": [
            ("CSI history", "(B,16,16,64,2)", "복소수 CSI 과거 16개 프레임입니다."),
            ("Raw branch", "(B,16,256)", "주파수 영역 Re/Im을 2,048차원으로 펼친 뒤 Linear로 256차원 임베딩합니다."),
            ("Delay branch", "(B,16,256)", "64개 서브캐리어에 complex IFFT를 적용해 delay tap Re/Im을 만든 뒤 256차원으로 임베딩합니다."),
            ("Dual fusion", "(B,16,512)→256", "두 branch를 concatenate하고 Linear+LayerNorm으로 256차원 시간 토큰을 만듭니다."),
            ("Causal GLU ×6", "(B,16,256)", "kernel 3, dilation 1·2·4·8·1·2의 left-padded Conv1d로 현재 토큰이 과거만 보도록 합니다."),
            ("P-query head", "(B,4,16,64,2)", "4개 query와 MLP hidden 1024가 ΔH를 만들고 마지막 CSI에 더합니다."),
        ],
        "kind": "dilation", "dilations": [1, 2, 4, 8, 1, 2],
        "interaction_text": "레이어를 바꾸면 dilation이 넓어지면서 마지막 시점이 참조할 수 있는 과거 범위가 누적되는 모습을 확인할 수 있습니다.",
        "notes": [
            ("왜 delay branch인가?", "주파수 응답에 섞여 있는 다중경로 지연 구조를 IFFT 좌표계에서 명시적으로 제공합니다."),
            ("왜 causal conv인가?", "각 시간 토큰이 자신의 미래 history token을 보지 않고 과거 방향으로만 정보를 결합합니다."),
            ("해석 주의", "delay tap이 곧 특정 LOS/NLOS 경로를 의미하지는 않으며 실제 경로 판별에는 RT metadata가 필요합니다."),
        ],
        "rows": [("Dual embedding", "raw 256 + delay 256 → 256"), ("TCN", "kernel 3 · dilations 1,2,4,8,1,2"), ("Gate", "GLU inside each residual block"), ("Prediction head", "P=4 queries · MLP hidden 1024"), ("Initialization", "zero-init ΔH · copy-last start")],
    },
    "chiron": {
        "title": "Chiron · Factorized Spatio-Temporal Model",
        "subtitle": "안테나–서브캐리어 2D 격자를 patch로 유지하고 시간 attention과 공간 attention을 분리해 처리합니다.",
        "accent": "#ffad5c",
        "input": "(B,16,16,64,2)", "output": "(B,4,16,64,2)",
        "facts": [("13.90 M", "parameters"), ("8", "patches/frame"), ("128", "total tokens"), ("6 blocks", "factorized depth")],
        "stages": [
            ("CSI grid", "(B,16,16,64,2)", "16개 시간 각각에 16 antenna × 64 subcarrier 복소수 격자가 있습니다."),
            ("2D patches", "4×32×2=256 values", "한 프레임을 antenna 4개 × subcarrier 32개의 patch 8개로 나눕니다."),
            ("Patch embed", "(B,16,8,256)", "각 patch의 256개 값을 Linear·LayerNorm·GELU로 256차원 토큰으로 바꿉니다."),
            ("Temporal view", "(B×8,16,256)", "동일한 spatial patch 위치를 16개 시간에 걸쳐 symmetric convolution과 bidirectional attention으로 처리합니다."),
            ("Spatial view", "(B×16,8,256)", "각 시간 안에서 8개 antenna–subcarrier patch 사이를 self-attention으로 처리합니다."),
            ("P-query head", "(B,4,16,64,2)", "총 128개 토큰을 4개 미래 query가 참조해 미래 CSI를 생성합니다."),
        ],
        "kind": "patch",
        "interaction_text": "8개 patch를 눌러 각 token이 어느 안테나와 서브캐리어 범위를 포함하는지 확인합니다.",
        "notes": [
            ("2D의 의미", "시각화가 2D라는 뜻이 아니라 안테나–서브캐리어 위치를 patch token 구조로 보존한다는 뜻입니다."),
            ("Factorized 처리", "시간축 K=16과 공간축 S=8을 따로 attention하여 하나의 128×128 attention보다 구조를 명시합니다."),
            ("Bidirectional의 범위", "입력 16개는 모두 관측된 과거이므로 그 내부의 양방향 attention은 미래 target leakage가 아닙니다."),
        ],
        "rows": [("Patch", "4 antennas × 32 subcarriers × Re/Im"), ("Embedding", "256 values → d256"), ("Core", "temporal + spatial + gated FFN ×6"), ("Heads / conv", "4 heads · temporal kernel 7"), ("Prediction head", "P=4 queries · MLP hidden 1024")],
    },
    "multimodal-fusion": {
        "title": "Multimodal · GatedFusion / EGRP / MLLM",
        "subtitle": "채널 backbone의 시간 토큰을 query로, BS camera·Radar·LiDAR 토큰을 key/value로 사용해 미래 복소수 CSI를 예측합니다.",
        "accent": "#5f8dff",
        "input": "CSI + W=5 sensors", "output": "(B,4,16,64,2)",
        "facts": [("15", "full sensor tokens"), ("128", "sensor width"), ("g=0", "gate initialization"), ("Target A", "final CSI output")],
        "stages": [
            ("Channel backbone", "(B,K,D)", "LSTM/LWM/DTCN은 D=256, Mamba는 128, Chiron은 K×S=128 tokens와 D=256을 사용합니다."),
            ("Sensor inputs", "W=5 per modality", "Radar (B,5,2,16,16), BS camera (B,5,3,64,64), LiDAR (B,5,1,16,16)입니다."),
            ("FrameCNN", "각 modality (B,5,128)", "모달리티별 CNN이 해상도가 다른 센서 프레임을 동일한 128차원 토큰으로 바꿉니다."),
            ("Cross-attention", "Q=channel · K/V=sensors", "Full 조건에서는 15개 센서 토큰을 concatenate하고 channel 차원 D로 projection합니다."),
            ("Gate", "z′=z+g·Norm(att)", "GatedFusion은 학습 scalar g, EGRP는 샘플별 onset 확률을 사용합니다."),
            ("CSI head", "(B,4,16,64,2)", "융합된 channel token으로 ΔH를 만들고 마지막 관측 CSI에 더합니다."),
        ],
        "kind": "fusion",
        "interaction_text": "gate를 움직여 sensor residual이 channel token에 얼마나 주입되는지 확인하고, 모델 탭에서 GatedFusion·EGRP·MLLM 차이를 비교합니다.",
        "notes": [
            ("최종 타깃", "멀티모달 모델의 최종 출력은 Target A인 미래 복소수 CSI입니다."),
            ("EGRP의 보조 타깃", "Frozen radar onset head의 Target B 확률은 센서 주입 gate를 만들기 위한 보조 신호입니다."),
            ("인과 검증", "Original 성능만으로 센서 이득을 주장하지 않고 Zero·Shuffle에서 개선이 사라지는지 확인해야 합니다."),
        ],
        "rows": [("Sensor window", "W=5 at 10 ms spacing"), ("Sensor encoders", "FrameCNN · output 128"), ("Fusion", "4-head cross-attention · dropout 0.1"), ("Regularization", "modality dropout 0.15"), ("GatedFusion", "scalar gate initialized to 0"), ("EGRP", "α·sigmoid(frozen radar onset logit)")],
    },
}


PAPER = {
    "lstm": [("CSI history", "16×16×64×2"), ("Flatten", "16×2,048"), ("3-layer LSTM", "16×256"), ("P=4 query head", "4×2,048"), ("Prediction Ŷ", "4×16×64×2")],
    "lwm": [("CSI history", "16×16×64×2"), ("Wideband embed", "16×64"), ("Transformer ×12", "8-head attention"), ("Adapter + P head", "64→256→2,048"), ("Prediction Ŷ", "4×16×64×2")],
    "mamba": [("CSI history", "16×16×64×2"), ("Wideband embed", "16×128"), ("Mamba ×4", "state 16 · conv 4"), ("P=4 query head", "128→2,048"), ("Prediction Ŷ", "4×16×64×2")],
    "dtcn": [("CSI history", "16×16×64×2"), ("Raw + IFFT delay", "256 + 256"), ("Causal GLU TCN ×6", "dilation 1·2·4·8·1·2"), ("P=4 query head", "256→2,048"), ("Prediction Ŷ", "4×16×64×2")],
    "chiron": [("CSI grid", "16×16×64×2"), ("2D patch embed", "16×8×256"), ("Factorized blocks ×6", "temporal + spatial"), ("P=4 query head", "128 tokens→2,048"), ("Prediction Ŷ", "4×16×64×2")],
    "multimodal-fusion": [("CSI + sensors", "K=16 · W=5"), ("Backbone + FrameCNN", "channel D · sensor 128"), ("Cross-attention", "Q=channel · K/V=sensor"), ("Gate + P head", "z+g·att → ΔH"), ("Prediction Ŷ", "4×16×64×2")],
}

FOCUS = {"lstm": "LSTM", "lwm": "LWM", "mamba": "Mamba", "dtcn": "DTCN", "chiron": "Chiron"}
CAMPAIGNS = {
    "lstm": ("c1", "c2", "c4"), "lwm": ("c1", "c2", "c4"),
    "mamba": ("c1", "c2"), "dtcn": ("c2", "c4"),
    "chiron": ("c1", "c2"), "multimodal-fusion": ("c4",),
}
CURVES = {
    "lstm": ("c4_lstm",), "lwm": ("c4_lwm",), "mamba": (),
    "dtcn": ("c2_dtcn", "c4_dtcn"), "chiron": (),
    "multimodal-fusion": ("c4_lstm", "c4_lwm", "c4_dtcn", "c4_egrp"),
}
CAMPAIGN_TITLES = {
    "c1": "C1 · 1–4 ms campaign-held-out test",
    "c2": "C2 · 1–4 ms split62 development validation",
    "c4": "C4 · BS-camera full-fusion development validation",
}


def evidence_for(slug: str) -> dict:
    return {
        "focus": FOCUS.get(slug),
        "campaigns": [
            {"id": c, "title": CAMPAIGN_TITLES[c],
             "protocol": EVIDENCE["protocols"][c], "rows": EVIDENCE[c]}
            for c in CAMPAIGNS[slug]
        ],
        "curves": [EVIDENCE["curves"][key] for key in CURVES[slug]],
        "controls": EVIDENCE["c4_controls"] if slug == "multimodal-fusion" else None,
    }


def paper_html(slug: str) -> str:
    blocks = []
    for i, (title, shape) in enumerate(PAPER[slug]):
        if i:
            blocks.append('<div class="parrow">→</div>')
        blocks.append(f'<div class="pbox"><b>{title}</b><code>{shape}</code></div>')
    return (
        '<div class="paper"><div class="paper-flow">' + "".join(blocks) + '</div>'
        '<div class="paper-supervision">'
        '<div class="pbox truth"><b>Ground truth Y</b><code>(B,4,16,64,2)</code></div>'
        '<div class="parrow">→</div>'
        '<div class="pbox compare"><b>Prediction Ŷ vs Ground truth Y</b>'
        '<code>Train: normalized weighted MSE<br>Evaluation: raw-space NMSE (dB)</code></div>'
        '</div><div class="paper-caption">위쪽은 inference 경로입니다. 학습할 때만 저장된 미래 CSI 정답 Y가 comparator로 들어가며, 추론 시에는 정답을 입력으로 사용하지 않습니다.</div></div>'
    )


def result_tables_html(data: dict) -> str:
    ev = data["evidence"]
    chunks = []
    for campaign in ev["campaigns"]:
        rows = []
        for item in campaign["rows"]:
            focus = " focus" if item["model"] == ev.get("focus") else ""
            steps = " / ".join(f"{x:.2f}" for x in item.get("per_step_db") or [])
            rows.append(
                f'<tr class="{focus.strip()}"><td>{item["model"]}</td>'
                f'<td>{item["nmse_db"]:.2f}</td><td>{item["median_db"]:.2f}</td>'
                f'<td>{item["copy_db"]:.2f}</td><td>{item["gain_db"]:+.2f}</td><td>{steps}</td></tr>'
            )
        chunks.append(
            f'<h3>{campaign["title"]}</h3><div class="protocol">{campaign["protocol"]}</div>'
            '<div class="table-scroll"><table class="result-table"><thead><tr><th>Model</th><th>Mean NMSE dB ↓</th>'
            '<th>Median dB ↓</th><th>Copy-last dB</th><th>Gain dB ↑</th><th>Horizon별 dB</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
        )
    if ev.get("controls"):
        rows = []
        for name, item in ev["controls"].items():
            rows.append(
                f'<tr><td>{name}</td><td>{item["original"]:.2f}</td><td>{item["gate"]:.4f}</td>'
                f'<td>{item["radar_zero"]:.2f}/{item["radar_shuffle"]:.2f}</td>'
                f'<td>{item["camera_zero"]:.2f}/{item["camera_shuffle"]:.2f}</td>'
                f'<td>{item["lidar_zero"]:.2f}/{item["lidar_shuffle"]:.2f}</td></tr>'
            )
        chunks.append(
            '<h3>C4 sensor-content audit</h3><div class="protocol">각 셀은 Zero/Shuffle NMSE dB입니다. Original과 동일하면 해당 센서 내용에 의존한 성능 이득이 관측되지 않았다는 뜻입니다.</div>'
            '<div class="table-scroll"><table class="result-table"><thead><tr><th>Model</th><th>Original</th><th>Gate</th><th>Radar 0/S</th><th>Camera 0/S</th><th>LiDAR 0/S</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>'
        )
    return "".join(chunks)


def curve_html(data: dict) -> str:
    if not data["evidence"]["curves"]:
        return '<div class="missing"><b>저장된 epoch 단위 학습 로그가 없습니다.</b><br>Mamba·Chiron의 C2 결과 JSON에는 최종/best NMSE는 있지만 train loss·validation loss의 epoch history가 남아 있지 않습니다. 따라서 곡선을 임의 복원하지 않고 아래 결과 표만 제공합니다.</div>'
    return (
        '<div class="chart-tabs" id="curveTabs"></div><div class="protocol" id="curveProtocol"></div>'
        '<div class="chart-grid">'
        '<div class="chartbox"><h3>Train loss · normalized weighted MSE</h3><canvas id="trainChart"></canvas></div>'
        '<div class="chartbox"><h3>Validation loss · same weighted MSE</h3><canvas id="valChart"></canvas></div>'
        '<div class="chartbox"><h3>Validation NMSE · dB</h3><canvas id="nmseChart"></canvas></div>'
        '</div><div class="callout">Loss와 NMSE는 서로 다른 값입니다. loss는 RMS-normalized 공간의 onset-weighted 평균 제곱오차이고, NMSE는 sample별 raw-space error ratio를 평균한 뒤 dB로 변환합니다.</div>'
    )


def page(slug: str, data: dict) -> str:
    data = dict(data)
    data["evidence"] = evidence_for(slug)
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    fact_html = "".join(f'<div class="fact"><b>{v}</b><span>{k}</span></div>' for v, k in data["facts"])
    note_html = "".join(f'<div class="note"><b>{t}</b>{d}</div>' for t, d in data["notes"])
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in data["rows"])
    paper = paper_html(slug)
    curves = curve_html(data)
    results = result_tables_html(data)
    return f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{data['title']}</title><style>{CSS}</style></head>
<body style="--accent:{data['accent']}"><main>
  <nav class="topnav"><a class="home" href="../">← 전체 모델</a><span class="badge">ACTUAL CARLA EXPERIMENT ARCHITECTURE</span></nav>
  <section class="hero"><div><h1>{data['title']}</h1><p>{data['subtitle']}</p><div class="facts">{fact_html}</div></div><div class="shape-card"><small>INPUT</small><div class="shape">{data['input']}</div><small style="margin-top:12px">OUTPUT</small><div class="shape">{data['output']}</div></div></section>
  <section class="card"><h2>논문형 End-to-End 모델 구조</h2>{paper}</section>
  <section class="card"><h2>Tensor flow · 단계를 눌러 설명 보기</h2><div class="pipeline" id="pipeline"></div><div class="detail"><div><small>SELECTED STAGE</small><h3 id="stageTitle"></h3><div class="bigshape" id="stageShape"></div></div><div><p id="stageDesc"></p></div></div></section>
  <section class="card"><h2>Interactive structure</h2><p style="color:var(--muted)">{data['interaction_text']}</p><div class="interactive" id="interactive"></div></section>
  <section class="card"><h2>예측값 Ŷ와 실제값 Y를 어떻게 비교하는가?</h2>
    <div class="metric-flow"><div class="flowbox"><strong>Per-sample scale</strong><code>sᵢ = RMS(H_last)</code><span>history와 target을 sᵢ로 나눔</span></div><div class="arrow">→</div><div class="flowbox"><strong>Model prediction</strong><code>Ŷ_norm = H_last,norm + ΔĤ</code><span>shape (B,4,16,64,2)</span></div><div class="arrow">→</div><div class="flowbox"><strong>Ground truth comparison</strong><code>Ŷ_norm ↔ Y_norm</code><span>평가는 sᵢ를 곱한 raw-space 기준</span></div></div>
    <div class="grid2" style="margin-top:14px"><div class="equation">Training loss<br>L = (1/B) Σᵢ wᵢ · mean[(Ŷᵢ,norm−Yᵢ,norm)²]<br>wᵢ = 3 (near-onset), otherwise 1</div><div class="equation">Evaluation NMSE<br>rᵢ = ‖Ŷᵢ−Yᵢ‖² / (‖Yᵢ‖²+ε)<br>NMSE[dB] = 10log₁₀[(1/N)Σᵢrᵢ]</div></div>
    <div class="callout">미래 target의 통계로 scale을 계산하지 않고 마지막 history CSI의 RMS만 사용하므로 정규화 과정에 미래 정보가 들어가지 않습니다. 낮은 NMSE가 더 좋은 결과입니다.</div>
  </section>
  <section class="card"><h2>실제 학습곡선 · Train loss / Validation loss / NMSE</h2>{curves}</section>
  <section class="card results"><h2>저장된 실험 결과</h2>{results}<div class="callout warn">C1 test, C2 validation, C4 validation은 프로토콜이 서로 다르므로 하나의 통합 순위로 직접 비교하면 안 됩니다. C2·C4의 test field는 validation을 재사용하므로 논문에서는 development validation으로 표기해야 합니다.</div></section>
  <section class="card"><h2>설계 의도와 해석</h2><div class="notes">{note_html}</div></section>
  <section class="card"><h2>실제 구현 설정</h2><table class="table">{rows}</table><div class="callout">모든 channel-only 모델은 동일한 P=4 query head와 residual prediction을 사용합니다. 마지막 output projection을 0으로 초기화해 학습 시작점이 copy-last가 되도록 맞췄습니다.</div></section>
  <div class="footer">Source of record: CARLA experiment implementation and 2026-07-19 integrated report · Generated by <code>tools/build_model_visualizations.py</code></div>
</main><script>const DATA={payload};
const pipeline=document.getElementById('pipeline');pipeline.style.setProperty('--cols',DATA.stages.length);
DATA.stages.forEach((s,i)=>{{const b=document.createElement('button');b.className='stage';b.innerHTML=`<span class="n">0${{i+1}}</span><strong>${{s[0]}}</strong><code>${{s[1]}}</code>`;b.onclick=()=>selectStage(i);pipeline.appendChild(b)}});
function selectStage(i){{[...pipeline.children].forEach((e,j)=>e.classList.toggle('active',i===j));document.getElementById('stageTitle').textContent=DATA.stages[i][0];document.getElementById('stageShape').textContent=DATA.stages[i][1];document.getElementById('stageDesc').textContent=DATA.stages[i][2]}}selectStage(0);
const root=document.getElementById('interactive');
function tokenRow(n,active=-1,upto=-1){{return `<div class="tokens">${{Array.from({{length:n}},(_,i)=>`<div class="token ${{i===active?'now':i<=upto?'on':''}}">t${{i+1}}</div>`).join('')}}</div>`}}
if(DATA.kind==='depth'){{root.innerHTML=`<div class="control"><span>적용 깊이</span><input id="ctl" type="range" min="1" max="${{DATA.depth}}" value="1"><span class="readout" id="rd"></span></div>${{tokenRow(16,-1,15)}}<div style="margin-top:18px" class="meter"><div id="fill"></div></div><div class="callout" id="msg"></div>`;const c=document.getElementById('ctl');function draw(){{const n=+c.value;document.getElementById('rd').textContent=`${{n}} / ${{DATA.depth}} ${{DATA.unit}}`;document.getElementById('fill').style.width=`${{100*n/DATA.depth}}%`;document.getElementById('msg').innerHTML=`sequence length <b>16</b>은 그대로이며, ${{n}}번째 layer까지 각 시간 토큰의 표현이 반복 갱신되었습니다.`}}c.oninput=draw;draw()}}
if(DATA.kind==='scan'){{root.innerHTML=`<div class="control"><button id="play">▶ scan</button><input id="ctl" type="range" min="1" max="16" value="1"><span class="readout" id="rd"></span></div><div id="tok"></div><div class="grid2" style="margin-top:18px"><div class="equation">${{DATA.scan_equation}}</div><div class="callout" id="msg"></div></div>`;const c=document.getElementById('ctl');let timer;function draw(){{const n=+c.value;document.getElementById('rd').textContent=`t = ${{n}} / 16`;document.getElementById('tok').innerHTML=tokenRow(16,n-1,n-2);document.getElementById('msg').innerHTML=`t${{n}}: ${{DATA.scan_message}} <b>미래 target은 scan 입력에 포함되지 않습니다.</b>`}}c.oninput=draw;document.getElementById('play').onclick=()=>{{if(timer){{clearInterval(timer);timer=null;return}}if(+c.value===16)c.value=1;draw();timer=setInterval(()=>{{if(+c.value===16){{clearInterval(timer);timer=null;return}}c.value=+c.value+1;draw()}},500)}};draw()}}
if(DATA.kind==='dilation'){{root.innerHTML=`<div class="control"><span>TCN layer</span><input id="ctl" type="range" min="1" max="${{DATA.dilations.length}}" value="1"><span class="readout" id="rd"></span></div><div id="tok"></div><div class="grid2" style="margin-top:18px"><div class="equation" id="eq"></div><div class="callout" id="msg"></div></div>`;const c=document.getElementById('ctl');function draw(){{const n=+c.value,ds=DATA.dilations.slice(0,n),rf=1+2*ds.reduce((a,b)=>a+b,0),start=Math.max(0,16-rf);document.getElementById('rd').textContent=`layer ${{n}} · dilation ${{DATA.dilations[n-1]}}`;document.getElementById('tok').innerHTML=tokenRow(16,15,14).replaceAll('class="token on"','class="token on"');[...document.querySelectorAll('#tok .token')].forEach((e,i)=>{{e.classList.toggle('on',i>=start&&i<15)}});document.getElementById('eq').innerHTML=`RF = 1 + 2·Σd<br>d = [${{ds.join(', ')}}]<br>theoretical RF = <b>${{rf}} frames</b>`;document.getElementById('msg').innerHTML=rf>=16?'누적 수용영역이 16-frame history 전체를 덮습니다. 왼쪽 padding만 사용하므로 미래 방향은 참조하지 않습니다.':`마지막 토큰이 이 단계에서 이론적으로 최근 ${{rf}}개 프레임까지 참조할 수 있습니다.`}}c.oninput=draw;draw()}}
if(DATA.kind==='patch'){{root.innerHTML=`<div class="grid2"><div><div class="patch-grid" id="patches"></div></div><div><div class="equation" id="patchInfo"></div><div class="callout">프레임당 8 tokens × 16 frames = <b>128 spatio-temporal tokens</b></div></div></div>`;const p=document.getElementById('patches');for(let i=0;i<8;i++){{const b=document.createElement('button');b.className='patch';b.textContent=`Patch ${{i+1}}`;b.onclick=()=>sel(i);p.appendChild(b)}}function sel(i){{[...p.children].forEach((e,j)=>e.classList.toggle('active',i===j));const ar=Math.floor(i/2)*4,sc=(i%2)*32;document.getElementById('patchInfo').innerHTML=`Patch ${{i+1}}<br>Antenna ${{ar}}–${{ar+3}}<br>Subcarrier ${{sc}}–${{sc+31}}<br><b>4×32×2 = 256 values → d256</b>`}}sel(0)}}
if(DATA.kind==='fusion'){{root.innerHTML=`<div class="fusion-flow"><div class="flowbox"><strong>Channel token z</strong><code>(B,K,D)</code></div><div class="arrow">+</div><div class="flowbox" id="sens"><strong>Sensor residual</strong><code>Norm(CrossAttn)</code></div><div class="arrow">→</div><div class="flowbox"><strong>Fused z′</strong><code>z + g·sensor</code></div></div><div class="control" style="margin-top:18px"><span>gate g</span><input id="gate" type="range" min="0" max="100" value="0"><span class="readout" id="rd"></span></div><div class="tabs" id="tabs"></div><div class="tabbody" id="tabbody"></div>`;const g=document.getElementById('gate');function drawGate(){{const v=+g.value/100;document.getElementById('rd').textContent=v.toFixed(2);document.getElementById('sens').style.opacity=.22+.78*v}}g.oninput=drawGate;drawGate();const tabs=[['GatedFusion','학습 가능한 하나의 scalar gate g를 사용합니다. g=0으로 시작하므로 초기 출력은 channel-only와 같고, 학습이 센서 residual의 필요성을 발견할 때만 gate가 열립니다.'],['EGRP','Frozen radar onset head가 샘플별 확률 e=σ(logit)을 만들고 gᵢ=α·eᵢ로 센서 주입량을 조절합니다. 최종 출력은 Target A CSI이며 onset은 Target B 보조 신호입니다.'],['MLLM','센서 토큰을 channel 토큰보다 앞에 배치하고 frozen GPT-2가 causal ordering으로 fusion합니다. Variant A는 GPT-2가 시간 모델링과 fusion을 함께 하고, Variant B는 LWM encoder 뒤에서 fusion만 수행합니다.']];const tb=document.getElementById('tabs'),body=document.getElementById('tabbody');tabs.forEach((x,i)=>{{const b=document.createElement('button');b.textContent=x[0];b.onclick=()=>sel(i);tb.appendChild(b)}});function sel(i){{[...tb.children].forEach((e,j)=>e.classList.toggle('active',i===j));body.textContent=tabs[i][1]}}sel(0)}}
function drawChart(id,xs,ys,color){{const canvas=document.getElementById(id);if(!canvas)return;const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(280,rect.width),h=210;canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);c.fillStyle='#071421';c.fillRect(0,0,w,h);const pairs=xs.map((x,i)=>[x,ys[i]]).filter(p=>Number.isFinite(p[1]));if(!pairs.length){{c.fillStyle='#8296aa';c.font='13px sans-serif';c.textAlign='center';c.fillText('해당 validation loss는 로그에 저장되지 않음',w/2,h/2);return}}const vals=pairs.map(p=>p[1]);let lo=Math.min(...vals),hi=Math.max(...vals);if(Math.abs(hi-lo)<1e-9){{lo-=.5;hi+=.5}}const pad=(hi-lo)*.10;lo-=pad;hi+=pad;const L=45,R=12,T=14,B=30;c.strokeStyle='#29445f';c.lineWidth=1;c.fillStyle='#8296aa';c.font='10px sans-serif';for(let i=0;i<5;i++){{const y=T+(h-T-B)*i/4,v=hi-(hi-lo)*i/4;c.beginPath();c.moveTo(L,y);c.lineTo(w-R,y);c.stroke();c.textAlign='right';c.fillText(v.toFixed(2),L-6,y+3)}}c.textAlign='center';c.fillText(String(pairs[0][0]),L,h-9);c.fillText(String(pairs[pairs.length-1][0]),w-R,h-9);c.strokeStyle=color;c.lineWidth=2.4;c.beginPath();pairs.forEach((p,i)=>{{const x=L+(w-L-R)*(p[0]-pairs[0][0])/Math.max(1,pairs[pairs.length-1][0]-pairs[0][0]),y=T+(h-T-B)*(hi-p[1])/(hi-lo);if(i===0)c.moveTo(x,y);else c.lineTo(x,y)}});c.stroke();const best=vals.indexOf(Math.min(...vals)),p=pairs[best],bx=L+(w-L-R)*(p[0]-pairs[0][0])/Math.max(1,pairs[pairs.length-1][0]-pairs[0][0]),by=T+(h-T-B)*(hi-p[1])/(hi-lo);c.fillStyle=color;c.beginPath();c.arc(bx,by,4,0,Math.PI*2);c.fill()}}
let selectedCurve=0;function selectCurve(i){{selectedCurve=i;const curves=DATA.evidence.curves,cv=curves[i];[...document.getElementById('curveTabs').children].forEach((e,j)=>e.classList.toggle('active',i===j));document.getElementById('curveProtocol').textContent=cv.protocol;drawChart('trainChart',cv.epochs,cv.train_loss,'#58df9b');drawChart('valChart',cv.epochs,cv.val_loss,'#ffad5c');drawChart('nmseChart',cv.epochs,cv.val_nmse_db,getComputedStyle(document.body).getPropertyValue('--accent').trim())}}
if(DATA.evidence.curves.length){{const ct=document.getElementById('curveTabs');DATA.evidence.curves.forEach((cv,i)=>{{const b=document.createElement('button');b.textContent=cv.label;b.onclick=()=>selectCurve(i);ct.appendChild(b)}});selectCurve(0);window.addEventListener('resize',()=>selectCurve(selectedCurve))}}
</script></body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for slug, data in MODELS.items():
        (OUT / f"{slug}.html").write_text(page(slug, data), encoding="utf-8")
        print(OUT / f"{slug}.html")


if __name__ == "__main__":
    main()
