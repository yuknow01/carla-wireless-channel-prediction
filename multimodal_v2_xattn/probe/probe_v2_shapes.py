"""mm2:chiron 프로브 (학습 없음, 합성 입력, 기본 CPU). 실행: python probe/probe_v2_shapes.py [--device cuda:0]
검사 항목
  1. 구성별 파라미터 분해(백본·헤드·융합·센서 인코더)와 단계별 shape (forward hook)
  2. 초기 항등: fuse_zero_init=True 이면 mm2:chiron(센서 입력) 출력 == 같은 백본 가중치의 채널 전용 ChironChannelPredictor 출력 (max|diff|)
  3. 식(21) 어텐션 가중치: 모달별 평균(합 1)·프레임별 [K,M] — 초기값은 질의 R_k 가 randn·0.02 라 거의 균등이어야 함
  4. fuse_where=input / output 두 위치, no_fuse_zero_init, fuse_no_channel_token, 센서 부분집합(cam / radar / cam,lidar,radar)
  5. 역전파: 손실 → 융합 질의·out_proj·센서 인코더에 grad 가 흐르는지
결과: probe/probe_v2_shapes.json + .log (같은 이름)
"""
import os, sys, json, time, argparse, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "cp")); sys.path.insert(1, ROOT)
from cp_models import build_model, count_params
from models.chiron_channel import ChironChannelPredictor

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu"); ap.add_argument("--B", type=int, default=2); a = ap.parse_args()
dev = torch.device(a.device); torch.manual_seed(0)
K, H, Nt, Ksc = 16, 4, 64, 64
log_lines = []
def log(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log_lines.append(s)

def synth_sensors(B, mods):
    g = torch.Generator().manual_seed(1)
    d = {}
    if "cam" in mods: d["cam"] = torch.randn(B, K, 196, 768, generator=g).half()
    if "lidar" in mods: d["lidar"] = torch.randn(B, K, 64, 384, generator=g).half()
    if "radar" in mods: d["radar"] = (torch.rand(B, K, 2, 64, 64, generator=g) * 3).half()
    if "pos" in mods: d["pos"] = torch.randn(B, K, 10, generator=g) * 20
    return {k: v.to(dev) for k, v in d.items()}

def hook_shapes(model):
    rec = []
    def mk(name):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            rec.append((name, list(o.shape) if torch.is_tensor(o) else str(type(o))))
        return f
    hs = [model.backbone.patch_embed.register_forward_hook(mk("backbone.patch_embed")),
          model.backbone.blocks[0].register_forward_hook(mk("backbone.blocks[0]")),
          model.backbone.final_norm.register_forward_hook(mk("backbone.final_norm")),
          model.backbone.head.register_forward_hook(mk("backbone.head"))]
    for m, e in model.enc.items(): hs.append(e.register_forward_hook(mk(f"enc.{m}")))
    for i, blk in enumerate(model.fusion):
        hs.append(blk.register_forward_pre_hook(lambda mod, inp, i=i: rec.append((f"fusion[{i}].U(in)", list(inp[0].shape)))))
        hs.append(blk.register_forward_hook(mk(f"fusion[{i}].f(out)")))
    return rec, hs

results = {}
configs = [
    dict(tag="full_input", sensors="cam,lidar,radar,pos", fuse_where="input"),
    dict(tag="full_output", sensors="cam,lidar,radar,pos", fuse_where="output"),
    dict(tag="full_input_no_zero_init", sensors="cam,lidar,radar,pos", fuse_where="input", fuse_zero_init=False),
    dict(tag="full_input_no_channel_token", sensors="cam,lidar,radar,pos", fuse_where="input", fuse_no_channel_token=True),
    dict(tag="cam_lidar_radar_input", sensors="cam,lidar,radar", fuse_where="input"),
    dict(tag="radar_input", sensors="radar", fuse_where="input"),
    dict(tag="cam_input", sensors="cam", fuse_where="input"),
    dict(tag="lidar16_input", sensors="lidar", fuse_where="input", lidar_tokens=16),
]
ref = ChironChannelPredictor(num_bs_antennas=Nt, num_subcarriers=Ksc, embed_dim=256, depth=6, num_heads=4, history_len=K, prediction_horizon=H, patch_h=4, patch_w=32, dropout=0.1).to(dev).eval()
X = torch.randn(a.B, K, Nt, Ksc, 2, device=dev)
with torch.no_grad(): y_ref = ref(X)
log(f"[ref] 채널 전용 ChironChannelPredictor params {count_params(ref):,} | X {list(X.shape)} → {list(y_ref.shape)}")

for cfg in configs:
    tag = cfg.pop("tag"); t0 = time.time()
    kw = dict(dropout=0.1, pos_source="predicted", pos_mode="token", fuse_layers=3, causal_mask=False, backbone_init="")
    kw.update({k: v for k, v in cfg.items()})
    model = build_model("mm2:chiron", K, H, Nt, Ksc, **kw).to(dev)
    model.backbone.load_state_dict(ref.state_dict(), strict=True)                       # 같은 백본 가중치 → 항등 검사 가능
    model.eval()
    sens = synth_sensors(a.B, model.sensors)
    rec, hs = hook_shapes(model)
    with torch.no_grad(): y = model(X, sens)
    for h in hs: h.remove()
    diff = float((y - y_ref).abs().max()); att = model.attention_over_modalities()
    pg = model.param_groups()
    # 역전파 점검(train 모드, 손실 = MSE to 0)
    model.train(); model.zero_grad(); loss = model(X, sens).pow(2).mean(); loss.backward()
    fz = model.fusion[0] if len(model.fusion) else None
    grads = dict(query=float(fz.query.grad.abs().sum()) if fz is not None and fz.query.grad is not None else None,
                 out_proj=float(fz.attn.out_proj.weight.grad.abs().sum()) if fz is not None and fz.attn.out_proj.weight.grad is not None else None,
                 in_proj=float(fz.attn.in_proj_weight.grad.abs().sum()) if fz is not None and fz.attn.in_proj_weight.grad is not None else None,
                 enc={m: float(sum(p.grad.abs().sum() for p in e.parameters() if p.grad is not None)) for m, e in model.enc.items()})
    results[tag] = dict(config=cfg, sensors=model.sensors, n_tok=model.n_tok, fuse_where=model.fuse_where, include_channel_token=model.include_channel_token,
                        params=pg, out_shape=list(y.shape), max_abs_diff_vs_channel_only=diff, attention=att, shapes=rec, grads=grads, sec=round(time.time() - t0, 2))
    log(f"[{tag}] sensors {model.sensors} n_tok {model.n_tok} fuse_where {model.fuse_where} | total {pg['total']:,} (fusion {pg['fusion']:,}, enc {pg['sensor_encoders']:,}) "
        f"| max|diff| vs 채널전용 {diff:.3e} | attn {json.dumps({k: round(v, 4) for k, v in att['per_modal'].items()}) if att else None} | {time.time() - t0:.1f}s")
    for name, shp in rec: log(f"      {name:28s} {shp}")
    log(f"      grads: query {grads['query']}, in_proj {grads['in_proj']}, out_proj {grads['out_proj']}, enc {json.dumps({k: round(v, 3) for k, v in grads['enc'].items()})}")

json.dump(results, open(os.path.join(HERE, "probe_v2_shapes.json"), "w"), indent=1, ensure_ascii=False)
open(os.path.join(HERE, "probe_v2_shapes.log"), "w").write("\n".join(log_lines) + "\n")
log(f"저장: {HERE}/probe_v2_shapes.json, .log")
