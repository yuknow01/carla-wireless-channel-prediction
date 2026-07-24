#!/usr/bin/env python3
"""Render the code-faithful multimodal fusion explainer page."""

from __future__ import annotations

import json


def _result_rows(data: dict) -> str:
    rows = []
    for item in data["evidence"]["campaigns"][0]["rows"]:
        steps = " / ".join(f"{value:.2f}" for value in item["per_step_db"])
        rows.append(
            "<tr>"
            f"<td><strong>{item['model']}</strong></td>"
            f"<td>{item['nmse_db']:.2f}</td>"
            f"<td>{item['median_db']:.2f}</td>"
            f"<td>{item['copy_db']:.2f}</td>"
            f"<td>{item['gain_db']:+.2f}</td>"
            f"<td>{steps}</td>"
            "</tr>"
        )
    return "".join(rows)


def _audit_rows(data: dict) -> str:
    rows = []
    for name, item in data["evidence"]["controls"].items():
        rows.append(
            "<tr>"
            f"<td><strong>{name}</strong></td>"
            f"<td>{item['original']:.2f}</td>"
            f"<td>{item['gate']:.4f}</td>"
            f"<td>{item['radar_zero']:.2f} / {item['radar_shuffle']:.2f}</td>"
            f"<td>{item['camera_zero']:.2f} / {item['camera_shuffle']:.2f}</td>"
            f"<td>{item['lidar_zero']:.2f} / {item['lidar_shuffle']:.2f}</td>"
            "</tr>"
        )
    return "".join(rows)


def render_multimodal_page(data: dict) -> str:
    """Return a standalone HTML page using the experiment evidence in *data*."""

    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (
        _TEMPLATE.replace("__DATA__", payload)
        .replace("__RESULT_ROWS__", _result_rows(data))
        .replace("__AUDIT_ROWS__", _audit_rows(data))
    )


_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <meta name="description" content="CARLA wireless channel prediction의 EGRP, GatedFusion, MLLM-B 멀티모달 구조를 실제 코드 shape와 함께 설명합니다.">
  <title>Multimodal Fusion · EGRP vs MLLM-B</title>
  <style>
    :root{
      --bg:#06101c;--panel:#0d1d30;--panel2:#091725;--line:#28445f;
      --text:#eef7ff;--muted:#9db2c8;--blue:#6595ff;--cyan:#3dd9eb;
      --green:#59dfa0;--orange:#ffad5c;--red:#ff7183;--purple:#b981ff;
    }
    *{box-sizing:border-box}html{scroll-behavior:smooth}
    body{margin:0;background:radial-gradient(circle at 52% -8%,#193f69 0,#07121f 39%,#06101c 75%);color:var(--text);font-family:Inter,Pretendard,"Noto Sans KR",system-ui,sans-serif}
    body:before{content:"";position:fixed;inset:0;pointer-events:none;background-image:linear-gradient(#fff0 95%,#fff025 96%),linear-gradient(90deg,#fff0 95%,#fff020 96%);background-size:38px 38px;mask-image:linear-gradient(#0005,transparent 65%)}
    main{max-width:1280px;margin:auto;padding:28px 26px 80px;position:relative}
    a{color:inherit}.top{display:flex;align-items:center;justify-content:space-between;gap:16px;margin-bottom:30px}
    .home,.toplink{color:var(--muted);text-decoration:none;border:1px solid var(--line);border-radius:10px;padding:9px 13px;background:#081725b8}
    .home:hover,.toplink:hover{color:var(--text);border-color:var(--cyan)}
    .eyebrow{font:850 12px ui-monospace,SFMono-Regular,monospace;letter-spacing:.13em;color:var(--cyan)}
    .hero{display:grid;grid-template-columns:1.06fr .94fr;gap:22px;align-items:center;padding:24px 0 18px}
    h1{font-size:clamp(38px,5.2vw,68px);line-height:1.03;letter-spacing:-2.7px;margin:13px 0 20px}
    h1 span{background:linear-gradient(100deg,var(--cyan),var(--blue),var(--purple));background-clip:text;color:transparent}
    .lead{color:#bfd0e1;font-size:18px;line-height:1.75;max-width:760px;margin:0}
    .hero-note{margin-top:20px;border-left:4px solid var(--orange);padding:13px 15px;background:#171a23;border-radius:8px;color:#dce9f5;line-height:1.65}
    .hero-visual{display:grid;gap:12px}
    .hero-path{border:1px solid var(--line);background:linear-gradient(140deg,#102842,#0a192a);border-radius:18px;padding:20px;position:relative;overflow:hidden}
    .hero-path:after{content:"";position:absolute;width:170px;height:170px;border-radius:50%;filter:blur(55px);opacity:.18;right:-35px;bottom:-70px;background:var(--path)}
    .hero-path .label{font-size:12px;font-weight:900;color:var(--path);letter-spacing:.08em}
    .hero-path h2{font-size:20px;margin:8px 0 12px}
    .mini-flow{display:flex;align-items:center;gap:6px;flex-wrap:wrap;position:relative;z-index:1}
    .mini-flow span{padding:7px 9px;border:1px solid color-mix(in srgb,var(--path) 48%,var(--line));border-radius:8px;background:#071421;font:750 11px ui-monospace,monospace}
    .mini-flow i{font-style:normal;color:var(--path);font-size:18px}
    .facts{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin:28px 0}
    .fact{background:#091827d9;border:1px solid var(--line);border-radius:13px;padding:14px}
    .fact b{display:block;color:var(--cyan);font-size:17px}.fact span{display:block;color:var(--muted);font-size:12px;margin-top:4px}
    .section-nav{position:sticky;top:10px;z-index:30;display:flex;gap:7px;overflow-x:auto;padding:8px;background:#071421e8;backdrop-filter:blur(12px);border:1px solid var(--line);border-radius:13px;margin:18px 0 24px}
    .section-nav a{text-decoration:none;white-space:nowrap;color:var(--muted);font-size:12px;font-weight:800;padding:7px 10px;border-radius:8px}
    .section-nav a:hover{color:var(--text);background:#13304b}
    section.card{scroll-margin-top:78px;margin-top:20px;border:1px solid var(--line);border-radius:20px;padding:25px;background:linear-gradient(145deg,rgba(16,36,59,.97),rgba(7,20,34,.98));box-shadow:0 18px 54px #0003}
    .section-head{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:20px}
    .section-no{font:900 11px ui-monospace,monospace;color:var(--cyan);letter-spacing:.15em}
    h2{font-size:26px;letter-spacing:-.6px;margin:5px 0 4px}h3{font-size:17px;margin:0 0 9px}
    .muted,.section-head p{color:var(--muted);line-height:1.65;margin:0;max-width:760px}
    .pill{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:99px;padding:6px 10px;color:#bfd1e2;font-size:11px;font-weight:800;background:#071421}
    .pill.live{border-color:#326f5a;color:var(--green)}.pill.legacy{border-color:#705c36;color:var(--orange)}
    .grid2{display:grid;grid-template-columns:1fr 1fr;gap:14px}.grid3{display:grid;grid-template-columns:repeat(3,1fr);gap:13px}
    .sensor{border:1px solid var(--line);border-top:3px solid var(--sensor);border-radius:14px;background:#081725;padding:16px}
    .sensor-head{display:flex;justify-content:space-between;gap:9px;align-items:center}.sensor code,.shape{font:750 12px ui-monospace,monospace;color:var(--sensor,var(--cyan))}
    .sensor p{font-size:13px;color:var(--muted);line-height:1.55;margin:11px 0}
    .frame-strip{display:grid;grid-template-columns:repeat(5,1fr);gap:5px}.frame-strip span{height:36px;display:grid;place-items:center;border:1px solid color-mix(in srgb,var(--sensor) 50%,var(--line));border-radius:7px;background:#0a1b2d;font:750 10px ui-monospace,monospace;color:#cfe0ef}
    .cnn{display:grid;grid-template-columns:repeat(9,auto);align-items:center;gap:7px;padding:17px;border:1px solid var(--line);border-radius:14px;background:#071421;margin-top:14px;overflow-x:auto}
    .box{min-width:120px;min-height:82px;display:flex;flex-direction:column;justify-content:center;text-align:center;border:1px solid var(--line);border-radius:10px;background:#0c1d30;padding:10px}
    .box b{font-size:13px}.box code{color:var(--cyan);font-size:11px;margin-top:7px}.arr{color:var(--cyan);font-size:22px;text-align:center}
    .tiny{font-size:11px;color:var(--muted);line-height:1.55}.callout{border-left:4px solid var(--cyan);border-radius:9px;padding:13px 15px;background:#081725;color:#c9dbea;line-height:1.65;margin-top:14px}
    .callout.orange{border-color:var(--orange)}.callout.red{border-color:var(--red);background:#24151e}.callout.green{border-color:var(--green)}
    .arch-tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
    button{border:1px solid var(--line);border-radius:9px;background:#102943;color:var(--text);font-weight:820;padding:9px 13px;cursor:pointer}
    button:hover,button.active{border-color:var(--cyan);color:var(--cyan)}
    .all-arch{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:14px}
    .all-arch-card{border:1px solid var(--line);border-top:3px solid var(--arch);border-radius:14px;background:#071421;padding:16px}
    .all-arch-card .arch-label{font:900 11px ui-monospace,monospace;color:var(--arch);letter-spacing:.08em}
    .all-arch-card h3{margin:7px 0 6px}.all-arch-card>p{color:var(--muted);font-size:12px;line-height:1.55;min-height:57px}
    .overview-flow{display:grid;gap:6px;margin-top:13px}.overview-step{position:relative;border:1px solid color-mix(in srgb,var(--arch) 38%,var(--line));border-radius:9px;background:#0b1b2c;padding:9px 10px}
    .overview-step:not(:last-child):after{content:"↓";position:absolute;bottom:-15px;left:50%;z-index:2;color:var(--arch);font-weight:900}
    .overview-step b{display:block;font-size:12px}.overview-step code{display:block;color:var(--arch);font-size:10px;line-height:1.45;margin-top:3px}
    .expand-title{display:flex;align-items:center;gap:10px;margin:4px 0 12px}.expand-title:after{content:"";height:1px;flex:1;background:var(--line)}
    .arch-summary{display:grid;grid-template-columns:.76fr 1.24fr;gap:14px}
    .read-card{background:#071421;border:1px solid var(--line);border-radius:13px;padding:17px}
    .read-card .big{font-size:23px;font-weight:900;color:var(--accent,var(--cyan));margin:9px 0}
    .read-card p{color:var(--muted);line-height:1.65;margin:0}
    .arch-flow{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;align-items:stretch}
    .arch-box{position:relative;border:1px solid var(--line);border-radius:12px;background:#091827;padding:14px;min-height:128px}
    .arch-box b{display:block;font-size:13px;margin-bottom:9px}.arch-box code{font-size:11px;color:var(--accent,var(--cyan));line-height:1.55}.arch-box p{font-size:11px;color:var(--muted);line-height:1.45;margin:8px 0 0}
    .arch-box:after{content:"→";position:absolute;right:-13px;top:43%;z-index:2;color:var(--accent,var(--cyan));font-size:19px}
    .arch-box:last-child:after{display:none}
    .formula{font:750 14px ui-monospace,SFMono-Regular,monospace;line-height:1.8;padding:16px;border:1px solid var(--line);background:#06131f;border-radius:11px;color:#dceeff}
    .formula em{font-style:normal;color:var(--cyan)}
    .interactive-grid{display:grid;grid-template-columns:.78fr 1.22fr;gap:14px;margin-top:14px}
    .controls{border:1px solid var(--line);border-radius:13px;background:#071421;padding:17px}
    .control{margin-bottom:16px}.control:last-child{margin-bottom:0}.control label{display:flex;justify-content:space-between;gap:8px;color:#c9d9e8;font-size:12px;font-weight:800;margin-bottom:7px}
    input[type=range]{width:100%;accent-color:var(--cyan)}.readout{font:850 12px ui-monospace,monospace;color:var(--cyan)}
    .gate-viz{display:grid;grid-template-columns:1fr 42px 1fr 42px 1fr;align-items:center;gap:7px}
    .gate-box{min-height:106px;border:1px solid var(--line);border-radius:11px;display:flex;align-items:center;justify-content:center;text-align:center;padding:11px;background:#0a1a2b;transition:.2s}
    .gate-box.sensor-residual{border-color:var(--purple)}.plus{font-size:24px;color:var(--cyan);text-align:center}
    .gate-badge{display:inline-block;padding:5px 8px;border-radius:7px;background:#152c45;color:var(--cyan);font:850 12px ui-monospace,monospace;margin-top:7px}
    .token-legend{display:flex;gap:12px;flex-wrap:wrap;margin:10px 0;color:var(--muted);font-size:11px}.dot{display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:4px}
    .tokens{display:grid;grid-template-columns:repeat(16,1fr);gap:5px}.tokens.sensors{grid-template-columns:repeat(15,1fr)}
    .token{height:42px;display:grid;place-items:center;border:1px solid var(--line);border-radius:7px;background:#081725;color:#758ca3;font:750 10px ui-monospace,monospace;transition:.18s}
    .token.radar{--tok:var(--orange)}.token.camera{--tok:var(--blue)}.token.lidar{--tok:var(--green)}.token.channel{--tok:var(--cyan)}
    .token.visible{border-color:var(--tok);background:color-mix(in srgb,var(--tok) 18%,#081725);color:#e9f7ff}
    .token.current{background:var(--tok);color:#06111c;box-shadow:0 0 20px color-mix(in srgb,var(--tok) 38%,transparent);transform:translateY(-2px)}
    .token.dim{opacity:.18}.seq-wrap{border:1px solid var(--line);border-radius:13px;background:#06131f;padding:15px;overflow-x:auto}.seq{display:flex;gap:5px;min-width:840px}
    .seq .token{width:42px;min-width:42px}.brace{display:grid;grid-template-columns:15fr 16fr;min-width:840px;margin-top:6px;color:var(--muted);font:10px ui-monospace,monospace;text-align:center}
    .variant{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:13px 0}.variant-card{border:1px solid var(--line);background:#081725;border-radius:12px;padding:14px;cursor:pointer}.variant-card.active{border-color:var(--purple);box-shadow:0 0 0 1px #b981ff44}.variant-card p{font-size:12px;color:var(--muted);line-height:1.55;margin:6px 0 0}
    .compare-table,.result-table{width:100%;border-collapse:collapse;font-size:13px;min-width:850px}.table-wrap{overflow-x:auto}
    .compare-table th,.compare-table td,.result-table th,.result-table td{padding:11px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}
    .compare-table th,.result-table th{color:#c7d8e8;font-size:11px;text-transform:uppercase;letter-spacing:.04em;background:#071421}
    .compare-table td:first-child{color:var(--muted);width:18%}.compare-table td:nth-child(2){border-left:2px solid #b981ff42}.compare-table td:nth-child(3){border-left:2px solid #3dd9eb42}
    .yes{color:var(--green);font-weight:850}.no{color:var(--orange);font-weight:850}
    .target-flow{display:grid;grid-template-columns:1fr 45px 1fr;gap:10px;align-items:stretch}.target{border:1px solid var(--line);border-radius:14px;padding:18px;background:#081725}.target.a{border-top:3px solid var(--cyan)}.target.b{border-top:3px solid var(--orange)}
    .target h3 span{font:850 11px ui-monospace,monospace;color:var(--cyan);margin-right:6px}.target.b h3 span{color:var(--orange)}.target p{color:var(--muted);line-height:1.6;font-size:13px}.target ul{padding-left:18px;color:#cbdbe9;line-height:1.65;font-size:13px}
    .chart-tabs{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:10px}.chart-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:11px}.chart{border:1px solid var(--line);border-radius:12px;background:#06131f;padding:11px}.chart h3{font-size:12px;color:#c8d8e7}.chart canvas{display:block;width:100%;height:210px}
    .result-table{text-align:right}.result-table th,.result-table td{text-align:right;white-space:nowrap}.result-table th:first-child,.result-table td:first-child{text-align:left}
    .status-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.status{border:1px solid var(--line);border-radius:12px;background:#081725;padding:15px}.status h3{font-size:14px}.status p{font-size:12px;color:var(--muted);line-height:1.55;margin:6px 0 0}
    .code-map{display:grid;grid-template-columns:1fr 1fr;gap:10px}.code-line{border:1px solid var(--line);border-radius:10px;background:#06131f;padding:12px}.code-line code{font-size:11px;color:var(--cyan);word-break:break-all}.code-line p{font-size:12px;color:var(--muted);line-height:1.5;margin:6px 0 0}
    footer{margin-top:24px;color:#72889f;font-size:12px;text-align:center}
    @media(max-width:960px){
      .hero,.arch-summary,.interactive-grid{grid-template-columns:1fr}.facts{grid-template-columns:repeat(3,1fr)}
      .grid3,.status-grid,.chart-grid{grid-template-columns:1fr}.all-arch{grid-template-columns:1fr}.all-arch-card>p{min-height:0}.arch-flow{grid-template-columns:1fr}.arch-box:after{content:"↓";right:auto;left:49%;top:auto;bottom:-18px}
      .target-flow,.gate-viz{grid-template-columns:1fr}.target-flow>.arr,.gate-viz>.plus{transform:rotate(90deg)}
      .cnn{grid-template-columns:1fr}.cnn>.arr{transform:rotate(90deg)}.box{min-width:0}
    }
    @media(max-width:640px){
      main{padding:20px 14px 60px}.toplink{display:none}.facts{grid-template-columns:1fr 1fr}section.card{padding:18px}.grid2,.variant,.code-map{grid-template-columns:1fr}
      h1{letter-spacing:-1.7px}.lead{font-size:16px}.tokens{grid-template-columns:repeat(8,1fr)}.tokens.sensors{grid-template-columns:repeat(5,1fr)}
    }
  </style>
</head>
<body>
<main>
  <nav class="top">
    <a class="home" href="../">← 전체 시각화</a>
    <a class="toplink" href="#code">실제 코드 위치 보기 ↓</a>
  </nav>

  <header class="hero">
    <div>
      <div class="eyebrow">CARLA · MULTIMODAL CHANNEL FORECASTING</div>
      <h1>같은 센서, 전혀 다른<br><span>두 가지 fusion</span></h1>
      <p class="lead">과거 CSI와 BS Camera·Radar·LiDAR를 사용해 미래 복소수 CSI 4개를 예측합니다. 아래 그림은 실제 구현을 기준으로 EGRP와 MLLM-B가 센서를 어디서, 어떤 attention으로 결합하는지 처음부터 출력까지 보여줍니다.</p>
      <div class="hero-note"><strong>한 문장으로 구분:</strong> EGRP는 <b>채널이 센서를 직접 조회하는 cross-attention</b>에 차폐 확률 gate를 곱합니다. MLLM-B는 <b>센서와 채널 토큰을 한 줄로 이어 붙인 뒤 frozen GPT-2의 causal self-attention</b>으로 섞습니다.</div>
    </div>
    <div class="hero-visual">
      <div class="hero-path" style="--path:var(--cyan)">
        <div class="label">EGRP · EXPLICIT FUSION</div><h2>Cross-attention + event gate</h2>
        <div class="mini-flow"><span>Q: CSI</span><i>↘</i><span>CrossAttn</span><i>→</i><span>× α·P(onset)</span><i>→</i><span>CSI residual</span></div>
      </div>
      <div class="hero-path" style="--path:var(--purple)">
        <div class="label">MLLM-B · TOKEN FUSION</div><h2>Concatenation + causal GPT-2</h2>
        <div class="mini-flow"><span>[R,C,L,H]</span><i>→</i><span>GPT-2 ×12</span><i>→</i><span>P=4 queries</span><i>→</i><span>CSI residual</span></div>
      </div>
    </div>
  </header>

  <div class="facts">
    <div class="fact"><b>K = 16</b><span>CSI history frames</span></div>
    <div class="fact"><b>W = 5</b><span>frames / sensor</span></div>
    <div class="fact"><b>15</b><span>full sensor tokens</span></div>
    <div class="fact"><b>P = 4</b><span>future CSI frames</span></div>
    <div class="fact"><b>16×64×2</b><span>one CSI frame</span></div>
  </div>

  <nav class="section-nav" aria-label="페이지 섹션">
    <a href="#inputs">입력·FrameCNN</a><a href="#map">전체 지도</a><a href="#egrp">EGRP</a>
    <a href="#mllm">MLLM-B</a><a href="#compare">직접 비교</a><a href="#targets">Target A/B</a>
    <a href="#evidence">실험 결과</a><a href="#code">코드 대응</a>
  </nav>

  <section class="card" id="inputs">
    <div class="section-head"><div><div class="section-no">01 · SHARED INPUTS</div><h2>모델에 실제로 들어가는 데이터</h2><p>배치 크기를 B라고 할 때 CSI는 16개 과거 프레임, 각 센서는 최근 5개 프레임입니다. 센서마다 원본 해상도와 채널 수가 달라도 FrameCNN 뒤에는 모두 128차원 토큰이 됩니다.</p></div><span class="pill">code-faithful shapes</span></div>
    <div class="grid3">
      <article class="sensor" style="--sensor:var(--orange)"><div class="sensor-head"><h3>Radar</h3><code>(B,5,2,16,16)</code></div><p>각 프레임은 2-channel range–angle feature map입니다. W=5이므로 샘플당 radar 토큰은 5개입니다.</p><div class="frame-strip"><span>R₁</span><span>R₂</span><span>R₃</span><span>R₄</span><span>R₅</span></div></article>
      <article class="sensor" style="--sensor:var(--blue)"><div class="sensor-head"><h3>BS Camera</h3><code>(B,5,3,64,64)</code></div><p>기지국 시점 RGB 이미지입니다. 현재 C4 캠페인에서 사용하는 BS-camera 입력만 표시했습니다.</p><div class="frame-strip"><span>C₁</span><span>C₂</span><span>C₃</span><span>C₄</span><span>C₅</span></div></article>
      <article class="sensor" style="--sensor:var(--green)"><div class="sensor-head"><h3>LiDAR</h3><code>(B,5,1,16,16)</code></div><p>한 채널의 BEV/grid 표현입니다. 동일한 W=5 window를 사용해 LiDAR 토큰 5개를 만듭니다.</p><div class="frame-strip"><span>L₁</span><span>L₂</span><span>L₃</span><span>L₄</span><span>L₅</span></div></article>
    </div>
    <div class="cnn" aria-label="FrameCNN 구조">
      <div class="box"><b>Sensor frame</b><code>Cin×H×W</code></div><div class="arr">→</div>
      <div class="box"><b>Conv 5×5</b><code>stride 2<br>GN + SiLU</code></div><div class="arr">→</div>
      <div class="box"><b>Conv 3×3 ×2</b><code>stride 2<br>GN + SiLU</code></div><div class="arr">→</div>
      <div class="box"><b>Global pool</b><code>H×W → 1×1</code></div><div class="arr">→</div>
      <div class="box"><b>Projection</b><code>Linear → 128<br>LayerNorm</code></div>
    </div>
    <div class="callout"><strong>FrameCNN은 무엇을 하나?</strong> 한 장의 2D 센서 프레임에서 공간 패턴을 압축해 128개 숫자로 된 벡터 하나를 만듭니다. 이 연산을 W=5 각 프레임에 독립적으로 적용한 뒤, 시간 embedding과 modality embedding을 더해 “언제 수집된 어느 센서인가”를 구분합니다.</div>
  </section>

  <section class="card" id="map">
    <div class="section-head"><div><div class="section-no">02 · ARCHITECTURE MAP</div><h2>두 가지 핵심 fusion 구조를 한 화면에서 비교</h2><p>EGRP와 MLLM-B의 입력부터 미래 CSI 출력까지를 동시에 표시했습니다. 아래 확대 보기에서는 버튼을 눌러 각 tensor 흐름을 더 자세히 볼 수 있습니다.</p></div></div>
    <div class="all-arch">
      <article class="all-arch-card" style="--arch:var(--cyan)">
        <div class="arch-label">MODEL 1 · EVENT-GUIDED FUSION</div><h3>EGRP</h3>
        <p>GatedFusion의 scalar를 frozen onset predictor가 만든 샘플별 차폐 확률 gate로 바꿉니다.</p>
        <div class="overview-flow">
          <div class="overview-step"><b>CSI + Sensors</b><code>Target A inputs + Target B sensor window</code></div>
          <div class="overview-step"><b>Backbone + FrameCNN</b><code>channel z · sensor KV</code></div>
          <div class="overview-step"><b>Cross-attention</b><code>att = MHA(z, sensor, sensor)</code></div>
          <div class="overview-step"><b>Per-sample event gate</b><code>gᵢ = α·sigmoid(onset logitᵢ)</code></div>
          <div class="overview-step"><b>P=4 residual head</b><code>z′ → ΔĤ → future CSI</code></div>
        </div>
      </article>
      <article class="all-arch-card" style="--arch:var(--purple)">
        <div class="arch-label">MODEL 2 · TOKEN FUSION</div><h3>MLLM-B</h3>
        <p>센서와 CSI를 31개 token sequence로 연결하고 frozen GPT-2의 causal self-attention으로 융합합니다.</p>
        <div class="overview-flow">
          <div class="overview-step"><b>CSI + Sensors</b><code>CSI 16 · sensor 15 tokens</code></div>
          <div class="overview-step"><b>LWM + FrameCNN</b><code>all modalities → d768</code></div>
          <div class="overview-step"><b>Token concatenation</b><code>[R₁…R₅,C₁…C₅,L₁…L₅,H₁…H₁₆]</code></div>
          <div class="overview-step"><b>Frozen GPT-2 ×12</b><code>causal self-attention · no event gate</code></div>
          <div class="overview-step"><b>P=4 residual head</b><code>Ŷ = Hlast + ΔĤ</code></div>
        </div>
      </article>
    </div>
    <div class="callout orange"><strong>GatedFusion의 위치:</strong> 세 번째 핵심 모델이 아니라 EGRP의 기반이 되는 baseline/ablation입니다. EGRP는 GatedFusion의 channel-to-sensor cross-attention을 그대로 사용하면서, 전역 scalar gate를 Target B onset 확률 기반의 샘플별 gate로 교체한 구조입니다.</div>
    <h3 class="expand-title">선택한 구조 확대 보기</h3>
    <div class="arch-tabs" id="archTabs"></div>
    <div class="arch-summary">
      <div class="read-card" id="archRead"></div>
      <div class="arch-flow" id="archFlow"></div>
    </div>
  </section>

  <section class="card" id="egrp">
    <div class="section-head"><div><div class="section-no">03 · EGRP / GATEDFUSION</div><h2>채널 token이 센서 token을 직접 조회</h2><p>LWM 또는 DTCN 같은 channel backbone이 만든 시간 token을 Query로 사용합니다. 센서 15개 token은 Key/Value가 되고, 4-head cross-attention의 결과를 residual로 채널 표현에 더합니다.</p></div><span class="pill live">현재 BS-camera C4 평가</span></div>
    <div class="arch-flow" style="--accent:var(--cyan)">
      <div class="arch-box"><b>① CSI backbone</b><code>LWM/DTCN full:<br>z ∈ ℝ(B,16,256)</code><p>과거 CSI의 시간 변화 표현</p></div>
      <div class="arch-box"><b>② Sensor tokens</b><code>[R₁…R₅,C₁…C₅,L₁…L₅]<br>(B,15,128) → (B,15,256)</code><p>concatenate 후 차원 projection</p></div>
      <div class="arch-box"><b>③ Cross-attention</b><code>Q=z<br>K=V=sensor<br>4 heads</code><p>각 채널 token이 모든 센서 token 조회</p></div>
      <div class="arch-box"><b>④ Event gate</b><code>gᵢ=α·σ(event_logitᵢ)<br>(B,1,1)</code><p>샘플별 하나의 gate 값</p></div>
      <div class="arch-box"><b>⑤ Residual fusion</b><code>z′=z+gᵢ·LN(att)<br>(B,16,256)</code><p>CSI backbone 정보는 항상 보존</p></div>
    </div>
    <div class="interactive-grid">
      <div class="controls">
        <h3>EGRP gate를 직접 바꿔 보기</h3>
        <div class="control"><label><span>Target B onset 확률 e</span><span class="readout" id="eventRead">0.70</span></label><input id="eventCtl" type="range" min="0" max="100" value="70"></div>
        <div class="control"><label><span>학습 scale α</span><span class="readout" id="alphaRead">1.00</span></label><input id="alphaCtl" type="range" min="-100" max="200" value="100"></div>
        <div class="formula">eᵢ = σ(frozen onset head)<br>gᵢ = α · eᵢ = <em id="gateRead">0.70</em><br>z′ᵢ = zᵢ + gᵢ · LN(attᵢ)</div>
      </div>
      <div>
        <div class="gate-viz">
          <div class="gate-box"><div><strong>Channel backbone</strong><br><span class="shape">(B,16,256)</span></div></div>
          <div class="plus">+</div>
          <div class="gate-box sensor-residual" id="sensorResidual"><div><strong>Sensor attention residual</strong><br><span class="shape">g × LN(att)</span><br><span class="gate-badge" id="gateBadge">gate 0.70</span></div></div>
          <div class="plus">→</div>
          <div class="gate-box"><div><strong>Fused channel tokens</strong><br><span class="shape">(B,16,256)</span></div></div>
        </div>
        <div class="callout green" id="gateExplain"></div>
      </div>
    </div>
    <div class="grid2" style="margin-top:14px">
      <div class="read-card" style="--accent:var(--orange)"><div class="eyebrow">BASIC GATEDFUSION</div><div class="big">g = one learned scalar</div><p>전체 데이터와 모든 샘플이 같은 scalar gate를 공유합니다. g=0으로 초기화되므로 학습 시작은 channel-only와 완전히 같습니다.</p></div>
      <div class="read-card" style="--accent:var(--cyan)"><div class="eyebrow">EGRP</div><div class="big">gᵢ = α × event probability</div><p>미리 학습해 고정한 onset predictor의 확률로 샘플별 gate를 만듭니다. 다만 시간별·센서별 gate가 아니라, 한 샘플의 sensor residual 전체에 곱하는 단일 값입니다.</p></div>
    </div>
  </section>

  <section class="card" id="mllm">
    <div class="section-head"><div><div class="section-no">04 · MLLM-B</div><h2>센서와 CSI를 한 sequence로 만든 뒤 GPT-2에서 융합</h2><p>MLLM-B에는 fusion 단계의 명시적인 channel-to-sensor cross-attention이 없습니다. 센서 token을 먼저, CSI token을 나중에 연결하고 frozen GPT-2의 causal self-attention이 결합합니다.</p></div><span class="pill legacy">과거 구조 탐색 · 현재 C4 직접 비교 아님</span></div>
    <div class="variant">
      <article class="variant-card" data-variant="a"><h3>Variant A · 문헌형</h3><p>CSI 한 프레임 2,048값을 Linear 2,048→768로 직접 token화합니다. GPT-2가 시간 모델링과 멀티모달 fusion을 모두 담당합니다.</p></article>
      <article class="variant-card active" data-variant="b"><h3>Variant B · 현재 설명 대상</h3><p>LWM이 먼저 CSI 시간 특징을 학습한 다음 768차원으로 맞춥니다. GPT-2는 이미 가공된 CSI와 sensor의 fusion에 집중합니다.</p></article>
    </div>
    <div class="arch-flow" style="--accent:var(--purple)" id="mllmFlow"></div>
    <div class="callout orange" id="variantExplain"></div>
    <div class="interactive-grid">
      <div class="controls">
        <h3>Causal attention이 볼 수 있는 token</h3>
        <p class="tiny">아래에서 CSI token H 위치를 이동해 보세요. 선택한 H는 왼쪽에 놓인 모든 sensor token과 자신까지의 이전 CSI token만 볼 수 있습니다.</p>
        <div class="control"><label><span>선택한 CSI token</span><span class="readout" id="mllmRead">H₁</span></label><input id="mllmCtl" type="range" min="1" max="16" value="1"></div>
        <div class="formula" id="mllmFormula"></div>
      </div>
      <div class="seq-wrap">
        <div class="token-legend"><span><i class="dot" style="background:var(--orange)"></i>Radar 5</span><span><i class="dot" style="background:var(--blue)"></i>Camera 5</span><span><i class="dot" style="background:var(--green)"></i>LiDAR 5</span><span><i class="dot" style="background:var(--cyan)"></i>CSI 16</span></div>
        <div class="seq" id="mllmTokens"></div>
        <div class="brace"><span>sensor tokens · 15</span><span>channel tokens · 16</span></div>
      </div>
    </div>
    <div class="grid3" style="margin-top:14px">
      <div class="read-card" style="--accent:var(--purple)"><div class="eyebrow">FUSION</div><div class="big">Concatenate</div><p>실제 순서는 [Radar 5, Camera 5, LiDAR 5, Channel 16]입니다. Full이면 총 31개 token입니다.</p></div>
      <div class="read-card" style="--accent:var(--purple)"><div class="eyebrow">GPT-2</div><div class="big">Frozen 124M</div><p>GPT-2 backbone parameter는 고정하고 LayerNorm만 학습합니다. CSI encoder, sensor encoder, projection, prediction head는 학습합니다.</p></div>
      <div class="read-card" style="--accent:var(--purple)"><div class="eyebrow">IMPORTANT</div><div class="big">No fusion gate</div><p>EGRP의 α·e gate가 없습니다. 센서 영향은 GPT-2 causal self-attention 안에서 암묵적으로 결정됩니다.</p></div>
    </div>
  </section>

  <section class="card" id="compare">
    <div class="section-head"><div><div class="section-no">05 · DIRECT COMPARISON</div><h2>EGRP와 MLLM-B를 같은 질문으로 비교</h2><p>두 모델 모두 마지막에는 미래 P=4 query head로 CSI를 예측하지만, 센서가 channel 표현에 들어오는 경로가 다릅니다.</p></div></div>
    <div class="table-wrap"><table class="compare-table">
      <thead><tr><th>비교 항목</th><th>MLLM-B</th><th>EGRP</th></tr></thead>
      <tbody>
        <tr><td>융합 핵심</td><td><strong>Token concatenation + GPT-2 causal self-attention</strong></td><td><strong>Channel→sensor cross-attention + event gate</strong></td></tr>
        <tr><td>Fusion 입력</td><td>[sensor 15, channel 16] → 총 31×768</td><td>Q=channel 16×256, K/V=sensor 15×256</td></tr>
        <tr><td>명시적 cross-attention</td><td><span class="no">Fusion에는 없음</span><br><span class="tiny">단, 최종 P-query decoder에는 있음</span></td><td><span class="yes">Fusion에 있음</span><br><span class="tiny">4-head MultiheadAttention</span></td></tr>
        <tr><td>인과성</td><td>GPT-2 causal mask: 오른쪽 미래 token은 볼 수 없음</td><td>Q/KV 분리: 관측된 CSI와 과거 sensor만 사용</td></tr>
        <tr><td>Target B 사용</td><td>사용하지 않음</td><td>Frozen onset head가 샘플별 gate 생성</td></tr>
        <tr><td>Gate</td><td>없음</td><td>gᵢ=α·sigmoid(event_logitᵢ)</td></tr>
        <tr><td>학습되는 부분</td><td>LWM, FrameCNN, projections, GPT-2 LN, P-head</td><td>channel backbone, FrameCNN, cross-attention, α, P-head</td></tr>
        <tr><td>현재 결과 범위</td><td>Legacy 구조 탐색 결과만 존재</td><td>현재 BS-camera C4와 sensor audit에 사용</td></tr>
      </tbody>
    </table></div>
  </section>

  <section class="card" id="targets">
    <div class="section-head"><div><div class="section-no">06 · TWO TARGETS, ONE FINAL TASK</div><h2>Target A와 Target B는 같은 출력이 아닙니다</h2><p>EGRP는 Target B 결과를 최종 정답으로 내는 모델이 아니라, Target B 확률을 이용해 Target A 채널 예측을 조절하는 모델입니다.</p></div></div>
    <div class="target-flow">
      <article class="target b"><h3><span>TARGET B</span>미래 차폐 시작 예측</h3><p>Radar/Camera/LiDAR 최근 5프레임으로 일정 horizon 안에 blockage onset이 시작될지를 이진 분류합니다.</p><ul><li>출력: future-onset logit → probability e</li><li>학습: BCE loss</li><li>평가: AUC / F1</li><li>EGRP에서는 checkpoint를 불러와 freeze</li></ul></article>
      <div class="arr">→</div>
      <article class="target a"><h3><span>TARGET A</span>미래 복소수 CSI 예측</h3><p>과거 CSI 16개와 센서 정보를 이용해 미래 CSI 4개를 직접 회귀합니다.</p><ul><li>출력: Ŷ ∈ ℝ(B,4,16,64,2)</li><li>학습: RMS-normalized weighted MSE</li><li>평가: raw-space NMSE dB</li><li>최종 출력은 Hlast + ΔĤ residual prediction</li></ul></article>
    </div>
    <div class="callout"><strong>최종 prediction head는 두 구조에 공통입니다.</strong> 미래 시점마다 하나씩 총 P=4 learnable query가 fusion token 전체를 cross-attention으로 읽고, shared MLP가 각 미래의 16×64×2 변화량 ΔH를 만듭니다. 여기의 cross-attention은 <em>출력 decoding</em>이며, MLLM-B의 multimodal fusion 방식과 혼동하면 안 됩니다.</div>
  </section>

  <section class="card" id="evidence">
    <div class="section-head"><div><div class="section-no">07 · STORED EXPERIMENT EVIDENCE</div><h2>현재 BS-camera C4 결과와 sensor-content audit</h2><p>C4는 50–200 ms Target A 개발 validation 결과입니다. 아래 수치는 저장된 result와 epoch log에서 그대로 읽었습니다.</p></div><span class="pill live">6 train seeds · 2 validation seeds</span></div>
    <div class="chart-tabs" id="curveTabs"></div><p class="tiny" id="curveProtocol"></p>
    <div class="chart-grid">
      <div class="chart"><h3>Train loss · normalized weighted MSE</h3><canvas id="trainChart"></canvas></div>
      <div class="chart"><h3>Validation loss · same weighted MSE</h3><canvas id="valChart"></canvas></div>
      <div class="chart"><h3>Validation NMSE · dB ↓</h3><canvas id="nmseChart"></canvas></div>
    </div>
    <h3 style="margin-top:22px">C4 best checkpoint 결과</h3>
    <div class="table-wrap"><table class="result-table"><thead><tr><th>Model</th><th>Mean NMSE dB ↓</th><th>Median dB ↓</th><th>Copy-last dB</th><th>Gain dB ↑</th><th>50 / 100 / 150 / 200 ms</th></tr></thead><tbody>__RESULT_ROWS__</tbody></table></div>
    <div class="callout orange"><strong>모델 성능과 센서 기여는 다른 질문입니다.</strong> DTCN full의 1.38 dB가 가장 낮지만, 이것만으로 Camera·Radar·LiDAR 덕분이라고 말할 수 없습니다. 같은 입력을 zero 또는 sample shuffle해 성능이 무너지는지 확인해야 합니다.</div>
    <h3 style="margin-top:22px">Zero / Shuffle 인과 audit</h3>
    <div class="table-wrap"><table class="result-table"><thead><tr><th>Model</th><th>Original</th><th>Gate</th><th>Radar Zero / Shuffle</th><th>Camera Zero / Shuffle</th><th>LiDAR Zero / Shuffle</th></tr></thead><tbody>__AUDIT_ROWS__</tbody></table></div>
    <div class="callout red"><strong>현재 확인된 결과:</strong> Original = Zero = Shuffle로 사실상 동일합니다. 기본 GatedFusion의 gate도 0 부근에 머물렀습니다. 따라서 이 C4 결과에서 성능 향상은 channel backbone이 만들었으며, 센서 내용의 추가 이득은 입증되지 않았습니다. EGRP gate가 0이 아니어도 audit 결과가 같으므로 “gate가 열렸다”와 “센서 내용이 유효했다”는 같은 말이 아닙니다.</div>
  </section>

  <section class="card" id="status">
    <div class="section-head"><div><div class="section-no">08 · EXPERIMENT STATUS</div><h2>구현됨과 현재 근거를 구분</h2><p>코드에 모델이 존재한다는 사실과 동일한 데이터·분할·학습 조건에서 비교가 끝났다는 사실은 구분해야 합니다.</p></div></div>
    <div class="status-grid">
      <article class="status"><span class="pill live">IMPLEMENTED + CURRENT</span><h3>EGRP 계열</h3><p>GatedFusion baseline을 포함해 BS Camera·Radar·LiDAR를 사용하는 C4 50–200 ms 결과와 Zero/Shuffle audit가 저장돼 있습니다.</p></article>
      <article class="status"><span class="pill legacy">IMPLEMENTED + LEGACY RUN</span><h3>MLLM-A / MLLM-B</h3><p>구조와 과거 실행 결과는 존재하지만, 현재 BS-camera C4 프로토콜에서 GatedFusion/EGRP와 matched rerun한 결과는 아닙니다.</p></article>
      <article class="status"><span class="pill">NEXT FAIR TEST</span><h3>동일 조건 비교</h3><p>같은 seed split, sensor source, initialization, epoch에서 EGRP vs MLLM-B를 비교하고 각 센서 Zero/Shuffle을 반복해야 융합 방식의 차이를 주장할 수 있습니다.</p></article>
    </div>
  </section>

  <section class="card" id="code">
    <div class="section-head"><div><div class="section-no">09 · CODE-TO-DIAGRAM MAP</div><h2>그림의 각 블록이 실제 어느 코드인가?</h2><p>아래 상대 경로는 CARLA 실험 저장소를 기준으로 합니다. 웹 그림은 이 구현을 읽어 shape와 연산 순서를 옮긴 것입니다.</p></div></div>
    <div class="code-map">
      <div class="code-line"><code>scenario_pilot/channel_prediction/models_mm.py · sensor_encoders()</code><p>Radar 2ch, Camera 3ch, LiDAR 1ch FrameCNN을 생성합니다.</p></div>
      <div class="code-line"><code>amber_beam_prediction/model.py · FrameCNN</code><p>Conv–GN–SiLU 세 단계, global pooling, Linear, LayerNorm의 실제 센서 encoder입니다.</p></div>
      <div class="code-line"><code>scenario_pilot/channel_prediction/models_mm.py · GatedFusion.forward()</code><p>Q=channel, K/V=sensor cross-attention과 z + g·norm(att)을 구현합니다.</p></div>
      <div class="code-line"><code>scenario_pilot/channel_prediction/models_mm.py · EGRP.forward()</code><p>Frozen onset head의 sigmoid 확률과 학습 scale α로 per-sample gate를 만듭니다.</p></div>
      <div class="code-line"><code>scenario_pilot/channel_prediction/models_mm.py · MLLMFusion.forward()</code><p>sensor-first concatenation, GPT-2 inputs_embeds, variant A/B channel tokenization을 구현합니다.</p></div>
      <div class="code-line"><code>models/chiron_channel.py · ChannelPredictionHead</code><p>P=4 query cross-attention, shared MLP, zero-init residual output을 구현합니다.</p></div>
    </div>
  </section>
  <footer>CARLA · Sionna RT · Multimodal Wireless Channel Forecasting · generated from the experiment source of record</footer>
</main>
<script>
const DATA=__DATA__;

const ARCH={
  egrp:{
    label:'EGRP',accent:'#3dd9eb',question:'차폐가 다가오는 샘플에서만 센서를 더 쓸 수 있는가?',
    answer:'Frozen onset probability로 sample별 cross-attention gate 생성',
    desc:'Target B onset predictor는 고정하고, eᵢ=σ(logit)와 학습 scale α로 gᵢ를 만듭니다. 최종 학습 목표와 출력은 Target A 미래 CSI입니다.',
    flow:[['Channel backbone','z: (B,K,D)'],['Sensor KV','15 tokens → D'],['Cross-attention','att=MHA(z,kv,kv)'],['Event gate','gᵢ=α·eᵢ'],['Residual + P head','z′→ future CSI']]
  },
  mllm:{
    label:'MLLM-B',accent:'#b981ff',question:'모든 modality를 하나의 token 언어처럼 처리할 수 있는가?',
    answer:'Sensor-first concatenation 뒤 frozen GPT-2 causal self-attention',
    desc:'LWM이 CSI를 먼저 시간 encoding하고 모든 token을 768차원으로 맞춥니다. Fusion 전용 cross-attention이나 event gate는 없습니다.',
    flow:[['LWM CSI encoder','16×2048 → 16×256'],['All → d768','sensor 15 + CSI 16'],['Concatenate','[R,C,L,H] = 31'],['Frozen GPT-2','causal self-attn ×12'],['P=4 CSI head','(B,4,16,64,2)']]
  }
};
const archTabs=document.getElementById('archTabs'),archRead=document.getElementById('archRead'),archFlow=document.getElementById('archFlow');
Object.entries(ARCH).forEach(([key,value])=>{const b=document.createElement('button');b.textContent=value.label;b.onclick=()=>selectArch(key);b.dataset.key=key;archTabs.appendChild(b)});
function selectArch(key){
  const a=ARCH[key];[...archTabs.children].forEach(x=>x.classList.toggle('active',x.dataset.key===key));
  archRead.style.setProperty('--accent',a.accent);
  archRead.innerHTML=`<div class="eyebrow">핵심 질문</div><div class="big">${a.question}</div><p><strong style="color:#eef7ff">${a.answer}</strong><br><br>${a.desc}</p>`;
  archFlow.style.setProperty('--accent',a.accent);
  archFlow.innerHTML=a.flow.map(x=>`<div class="arch-box"><b>${x[0]}</b><code>${x[1]}</code></div>`).join('');
}selectArch('egrp');

const eventCtl=document.getElementById('eventCtl'),alphaCtl=document.getElementById('alphaCtl');
function drawGate(){
  const e=+eventCtl.value/100,a=+alphaCtl.value/100,g=e*a;
  document.getElementById('eventRead').textContent=e.toFixed(2);
  document.getElementById('alphaRead').textContent=a.toFixed(2);
  document.getElementById('gateRead').textContent=g.toFixed(2);
  document.getElementById('gateBadge').textContent=`gate ${g.toFixed(2)}`;
  document.getElementById('sensorResidual').style.opacity=Math.max(.17,Math.min(1,.20+Math.abs(g)*.62));
  const msg=Math.abs(g)<.08?'gate가 거의 닫혀 sensor residual이 사실상 주입되지 않습니다. channel-only 경로가 지배합니다.':g>0?'onset 확률과 α에 비례해 sensor attention residual을 같은 방향으로 주입합니다.':'α가 음수이므로 sensor attention residual을 반대 방향으로 더합니다.';
  document.getElementById('gateExplain').innerHTML=`현재 샘플: <strong>e=${e.toFixed(2)}, α=${a.toFixed(2)}, g=${g.toFixed(2)}</strong><br>${msg}`;
}eventCtl.oninput=drawGate;alphaCtl.oninput=drawGate;drawGate();

const VARIANTS={
  a:[
    ['CSI direct token','2048 → 768'],['Sensor tokens','15×128 → 15×768'],['Concatenate','[R,C,L,H]'],['GPT-2 ×12','time + fusion'],['P=4 head','future CSI']
  ],
  b:[
    ['LWM ×12','2048 → 64'],['Adapter','64 → 256 → 768'],['Sensor → 768','15 tokens'],['GPT-2 ×12','fusion'],['P=4 head','future CSI']
  ]
};
function selectVariant(key){
  document.querySelectorAll('.variant-card').forEach(x=>x.classList.toggle('active',x.dataset.variant===key));
  document.getElementById('mllmFlow').innerHTML=VARIANTS[key].map(x=>`<div class="arch-box"><b>${x[0]}</b><code>${x[1]}</code></div>`).join('');
  document.getElementById('variantExplain').innerHTML=key==='b'?'<strong>Variant B를 선택한 이유:</strong> 동일한 LWM channel encoder를 사용하는 gated 계열과 비교할 때 fusion 방식의 차이를 더 잘 분리할 수 있습니다. 다만 GPT-2가 크기 때문에 완전한 capacity matching은 별도 검증이 필요합니다.':'<strong>Variant A의 의미:</strong> 문헌에 더 가까운 shallow channel tokenizer를 사용해 GPT-2가 시간 모델링과 fusion을 동시에 담당합니다.';
}
document.querySelectorAll('.variant-card').forEach(x=>x.onclick=()=>selectVariant(x.dataset.variant));selectVariant('b');

const mllmCtl=document.getElementById('mllmCtl'),mllmTokens=document.getElementById('mllmTokens');
function drawMLLM(){
  const h=+mllmCtl.value;document.getElementById('mllmRead').textContent=`H${h}`;
  const specs=[];
  ['R','C','L'].forEach((prefix,mi)=>{for(let i=1;i<=5;i++)specs.push([`${prefix}${i}`,['radar','camera','lidar'][mi],true,false])});
  for(let i=1;i<=16;i++)specs.push([`H${i}`,'channel',i<=h,i===h]);
  mllmTokens.innerHTML=specs.map(x=>`<div class="token ${x[1]} ${x[2]?'visible':'dim'} ${x[3]?'current':''}">${x[0]}</div>`).join('');
  document.getElementById('mllmFormula').innerHTML=`Query = H${h}<br>볼 수 있음: sensor 15 + CSI ${h}<br>총 visible = <em>${15+h} tokens</em><br>볼 수 없음: H${h+1}…H16`;
}mllmCtl.oninput=drawMLLM;drawMLLM();

function drawChart(id,xs,ys,color){
  const canvas=document.getElementById(id),rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,w=Math.max(270,rect.width),h=210;
  canvas.width=w*dpr;canvas.height=h*dpr;const c=canvas.getContext('2d');c.scale(dpr,dpr);c.fillStyle='#06131f';c.fillRect(0,0,w,h);
  const vals=ys.filter(Number.isFinite);let lo=Math.min(...vals),hi=Math.max(...vals);if(Math.abs(hi-lo)<1e-9){lo-=.5;hi+=.5}const extra=(hi-lo)*.1;lo-=extra;hi+=extra;
  const L=43,R=10,T=13,B=28;c.font='10px sans-serif';
  for(let i=0;i<5;i++){const y=T+(h-T-B)*i/4,v=hi-(hi-lo)*i/4;c.strokeStyle='#28445f';c.beginPath();c.moveTo(L,y);c.lineTo(w-R,y);c.stroke();c.fillStyle='#8096ac';c.textAlign='right';c.fillText(v.toFixed(2),L-5,y+3)}
  c.textAlign='center';c.fillText(xs[0],L,h-8);c.fillText(xs[xs.length-1],w-R,h-8);c.strokeStyle=color;c.lineWidth=2.4;c.beginPath();
  ys.forEach((v,i)=>{const x=L+(w-L-R)*i/(ys.length-1),y=T+(h-T-B)*(hi-v)/(hi-lo);i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();
  const bi=ys.indexOf(Math.min(...ys)),bx=L+(w-L-R)*bi/(ys.length-1),by=T+(h-T-B)*(hi-ys[bi])/(hi-lo);c.fillStyle=color;c.beginPath();c.arc(bx,by,4,0,Math.PI*2);c.fill();
}
let selectedCurve=0;
function selectCurve(i){
  selectedCurve=i;const cv=DATA.evidence.curves[i];[...document.getElementById('curveTabs').children].forEach((x,j)=>x.classList.toggle('active',i===j));
  document.getElementById('curveProtocol').textContent=cv.protocol;drawChart('trainChart',cv.epochs,cv.train_loss,'#59dfa0');drawChart('valChart',cv.epochs,cv.val_loss,'#ffad5c');drawChart('nmseChart',cv.epochs,cv.val_nmse_db,'#6595ff');
}
DATA.evidence.curves.forEach((cv,i)=>{const b=document.createElement('button');b.textContent=cv.label;b.onclick=()=>selectCurve(i);document.getElementById('curveTabs').appendChild(b)});
selectCurve(0);window.addEventListener('resize',()=>selectCurve(selectedCurve));
</script>
</body>
</html>
"""
