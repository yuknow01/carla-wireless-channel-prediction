"""13: 멀티모달 채널 예측 모델 mm:chiron(cp/cp_multimodal.py) + 센서 로더(cp/cp_sensor_data.py) 프로브 — MULTIMODAL_CP_ARCHITECTURE.md 근거 (2026-09-07).
10_probe_multimodal_shapes.py 와 같은 방식(forward hook 으로 shape·호출 수 기록)이되, 센서 입력은 합성이 아니라 **실제 캐시**(ViT 패치·LiDAR 풀링·레이더 래스터·CAV 위치)다.
  (1) B1 train 창 1개 × RX 2행(B=2)으로 8개 구성(cam / lidar / radar / cam+lidar / cam+lidar+radar / pos(token) / pos(broadcast) / cam+lidar+radar+pos) forward hook 프로브
      → 파라미터 분해(전체·센서 인코더·융합·백본·헤드), 출력 shape, hook 수, fwd 시간, 피크 메모리, 게이트 초기 평균, 채널 전용 chiron 과의 max|diff|(같은 백본 가중치).
  (2) 인과 마스크 검증(out_proj 를 무작위로 채운 사본에서 센서 프레임 15 만 교란 → 채널 프레임 0..14 토큰 불변인지).
  (3) 로더 실측: B=32 콜드/웜 로딩 시간, 창 1개 센서 메모리, 디스크 읽기량, 학습식 무작위 배치 20개의 시간 추이.
  (4) 학습 비용: 구성별 B=32 학습 step(fwd+bwd+opt, 워밍업 2 + 측정 10) 시간·피크 메모리 → epoch 추정(21,561 step + val 1,220 배치), 채널 전용 chiron(repo:chiron) 대조.
  (5) 커버리지 표(cp_sensor_data.coverage_report) + 위치 변환 자체 점검(true_ego_pos φ vs derived/aod).
GPU 0 만(CUDA_VISIBLE_DEVICES=0), AMP 없음, 학습 launch 없음. 기존 코드 무수정.
실행: CUDA_VISIBLE_DEVICES=0 /home/dlghdbs200/anaconda3/envs/hoyun_312/bin/python scripts/13_probe_multimodal_cp.py --device cuda:0
출력: scripts/13_probe_multimodal_cp.json, scripts/13_probe_multimodal_cp.log
"""
import os, sys, json, time, argparse, math, traceback, numpy as np, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); FEAS = os.path.dirname(HERE); CP = os.path.join(FEAS, "cp")
REPO = os.environ.get("CP_REPO", "/mnt/ssd_7t_2/carla-wireless-dataset/multimodal_code_index")
for p in (CP, REPO):
    if p not in sys.path: sys.path.insert(0, p)
from cp_data import load_index, make_windows
from cp_models import build_model, count_params
from cp_sensor_data import MMWindowSet, coverage_report, DEFAULT_CACHE_ROOT, ALL_SENSORS
from cp_multimodal import MMChiron
from models.chiron_channel import ChironChannelPredictor

ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--B", type=int, default=2); ap.add_argument("--train_B", type=int, default=32)
ap.add_argument("--steps", type=int, default=10); ap.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT)
ap.add_argument("--s1_ckpt", default="/mnt/ssd_7t_2/carla-wireless-dataset/mmw_reproduction/outputs_cp/s1_chiron_lr3e-4/best.pt")
ap.add_argument("--out", default=os.path.join(HERE, "13_probe_multimodal_cp.json")); a = ap.parse_args()
dev = torch.device(a.device); K, H = 16, 4; B = a.B
log_f = open(os.path.join(HERE, "13_probe_multimodal_cp.log"), "w")
def lp(*s):
    t = " ".join(str(x) for x in s); print(t, flush=True); log_f.write(t + "\n"); log_f.flush()
lp(f"torch {torch.__version__} cuda_visible {os.environ.get('CUDA_VISIBLE_DEVICES')} device {dev} B {B} train_B {a.train_B}")
def shp(x):
    if torch.is_tensor(x): return list(x.shape)
    if isinstance(x, (tuple, list)): return [shp(v) for v in x]
    if isinstance(x, dict): return {k: shp(v) for k, v in x.items()}
    return None
def peak(): return round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
CONFIGS = [("cam", "token"), ("lidar", "token"), ("radar", "token"), ("cam,lidar", "token"), ("cam,lidar,radar", "token"), ("pos", "token"), ("pos", "broadcast"), ("cam,lidar,radar,pos", "token")]
def cname(s, pm): return s.replace(",", "+") + ("(bcast)" if pm == "broadcast" else "")
results = dict(configs={}, loader={}, train_cost={}, coverage={}, env=dict(torch=torch.__version__, cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), device=str(dev)))

# ---------------- 실제 로더: 채널 memmap + 센서 캐시 ----------------
t0 = time.time(); idx, mm = load_index(); F = torch.from_numpy(np.asarray(mm)); Nr = F.shape[1]; Nt, Ksc = F.shape[2], F.shape[3]
win, is_val = make_windows(idx, K, H, split="B1")
ws = MMWindowSet(F, win["train"], K, H, Nr, idx, ",".join(ALL_SENSORS), pos_source="predicted", cache_root=a.cache_root, log=lp)
s0, ti = win["train"][0]
X, Y, scale, meta, SENS = ws.batch([0, 1], dev)                       # 창 0 × RX 행 0,1
data = dict(X=list(X.shape), Y=list(Y.shape), scale=[float(v) for v in scale], window0=dict(start=int(s0), traj=idx["trajs"][ti], frames=[int(v) for v in ws.frame_num[s0:s0 + K]]),
            sensors={k: list(v.shape) + [str(v.dtype)] for k, v in SENS.items()}, X_rms=[float(torch.sqrt((X[b] ** 2).sum(-1).mean())) for b in range(B)],
            sensor_stats={k: dict(mean=float(v.float().mean()), std=float(v.float().std()), absmax=float(v.float().abs().max())) for k, v in SENS.items()},
            pos_row0=[round(float(v), 3) for v in SENS["pos"][0, 0]], radar_count_sum_frame0=float(SENS["radar"][0, 0, 0].float().sum()), load_sec=round(time.time() - t0, 1))
lp("DATA", json.dumps(data)); results["data"] = data

# ---------------- (1) 구성별 hook 프로브 (B=2, eval) ----------------
def probe(model, fwd):
    recs, order = {}, []
    def mk(name):
        def hook(mod, args, kwargs, out):
            r = dict(cls=type(mod).__name__, inp=[shp(x) for x in args if torch.is_tensor(x) or isinstance(x, (tuple, list))], out=shp(out)); key = name or "<root>"
            if key not in recs: recs[key] = []; order.append(key)
            L = recs[key]
            if L and L[-1]["rec"] == r: L[-1]["n"] += 1
            else: L.append(dict(rec=r, n=1))
        return hook
    hs = [m.register_forward_hook(mk(n), with_kwargs=True) for n, m in model.named_modules() if not n.startswith("backbone.blocks.") or n.startswith("backbone.blocks.0")]   # 블록은 0번만
    model.eval()
    with torch.no_grad():
        torch.cuda.synchronize(); t = time.time(); out = fwd(); torch.cuda.synchronize(); dt = time.time() - t
    for h in hs: h.remove()
    return out, {k: [dict(cls=v["rec"]["cls"], n=v["n"], inp=v["rec"]["inp"], out=v["rec"]["out"]) for v in recs[k]] for k in order}, dt

def ref_chiron(m):
    r = ChironChannelPredictor(num_bs_antennas=Nt, num_subcarriers=Ksc, embed_dim=256, depth=6, num_heads=4, history_len=K, prediction_horizon=H, patch_h=4, patch_w=32, dropout=0.1).to(dev).eval()
    r.load_state_dict(m.backbone.state_dict()); return r

for sens, pm in CONFIGS:
    name = cname(sens, pm); torch.manual_seed(0); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); rec = dict(sensors=sens, pos_mode=pm)
    try:
        m = build_model("mm:chiron", K, H, Nt, Ksc, sensors=sens, pos_mode=pm, fuse_layers=3, causal_mask=False).to(dev)
        rec.update(params=count_params(m), groups=m.param_groups())
        sub = {k: v for k, v in SENS.items() if k in sens.split(",")}
        out, calls, dt = probe(m, lambda: m(X, sub, diag=True))
        ref = ref_chiron(m)
        with torch.no_grad(): yr = ref(X)
        rec.update(out_shape=list(out.shape), fwd_sec=round(dt, 4), peak_mem_mb=peak(), n_hooked=len(calls), gate_means_init=m.gate_means(), attn_mass_init=m.attention_mass(),
                   n_tok_per_frame=int(sum(m.n_tok[x] for x in m.token_mods)), sensor_tokens=int(K * sum(m.n_tok[x] for x in m.token_mods)),
                   max_abs_diff_vs_channel_only_chiron=float((out - yr).abs().max()), out_absmax=float(out.abs().max()), calls=calls)
        # 채널 전용과 같은지: 센서를 0으로 바꿔도 초기엔 같아야 함(항등)
        with torch.no_grad(): out0 = m(X, {k: torch.zeros_like(v) for k, v in sub.items()})
        rec["max_abs_diff_sensor_zero_init"] = float((out0 - out).abs().max())
        lp(f"OK  {name:26s} params {rec['params']/1e6:7.3f}M (enc {rec['groups']['sensor_encoders']/1e6:.3f}M fusion {rec['groups']['fusion']/1e6:.3f}M backbone {rec['groups']['backbone_no_head']/1e6:.3f}M head {rec['groups']['head']/1e6:.3f}M) "
           f"out {tuple(out.shape)} hooks {len(calls)} fwd {dt:.3f}s peak {rec['peak_mem_mb']:.0f}MB gate {rec['gate_means_init']} sensor_tok {rec['sensor_tokens']} max|diff| vs chiron {rec['max_abs_diff_vs_channel_only_chiron']:.2e}")
        del ref, m
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"; rec["tb"] = traceback.format_exc()[-1500:]; lp(f"ERR {name}: {rec['error']}")
    results["configs"][name] = rec; torch.cuda.empty_cache()

# S1 가중치로 백본 초기화 → RepoWrap chiron(S1 best.pt) 과 동일 출력인지
try:
    torch.manual_seed(0); m = build_model("mm:chiron", K, H, Nt, Ksc, sensors="cam,lidar,radar,pos", backbone_init=a.s1_ckpt).to(dev).eval()
    r = build_model("repo:chiron", K, H, Nt, Ksc, D=256, L=6).to(dev).eval(); sd = torch.load(a.s1_ckpt, map_location="cpu", weights_only=False)["model"]; r.load_state_dict(sd, strict=True)
    with torch.no_grad(): y1 = m(X, SENS); y2 = r(X)
    from cp_data import nmse_per_sample
    raw1, _, _ = nmse_per_sample(y1, Y); raw2, _, _ = nmse_per_sample(y2, Y)
    results["s1_init_check"] = dict(ckpt=a.s1_ckpt, epoch=int(torch.load(a.s1_ckpt, map_location="cpu", weights_only=False)["epoch"]), max_abs_diff=float((y1 - y2).abs().max()),
                                    nmse_db_mm=[round(float(10 * torch.log10(v)), 2) for v in raw1.mean(0)], nmse_db_s1=[round(float(10 * torch.log10(v)), 2) for v in raw2.mean(0)], gate=m.gate_means())
    lp("S1_INIT_CHECK", json.dumps(results["s1_init_check"])); del m, r
except Exception as e:
    results["s1_init_check"] = dict(error=f"{type(e).__name__}: {e}"); lp("S1_INIT_CHECK ERR", e)

# ---------------- (2) 인과 마스크 검증 ----------------
try:
    torch.manual_seed(0); m = build_model("mm:chiron", K, H, Nt, Ksc, sensors="cam,lidar,radar,pos", causal_mask=True).to(dev).eval()
    for blk in m.fusion: nn.init.normal_(blk.attn.out_proj.weight, std=0.02)     # 항등 초기화를 깨서 센서 영향이 보이게(사본 전용)
    cap = {}
    h = m.fusion[0].register_forward_hook(lambda mod, i, o: cap.__setitem__("y", o.detach().clone()))
    with torch.no_grad():
        m(X, SENS); y_ref = cap["y"]
        S2 = {k: v.clone() for k, v in SENS.items()}
        for k in S2: S2[k][:, K - 1] = S2[k][:, K - 1] * 3 + 1              # 마지막 센서 프레임(15)만 교란
        m(X, S2); y_pert = cap["y"]
        S3 = {k: v.clone() for k, v in SENS.items()}
        for k in S3: S3[k][:, 0] = S3[k][:, 0] * 3 + 1                      # 첫 센서 프레임(0)만 교란
        m(X, S3); y_pert0 = cap["y"]
    h.remove(); S = m.S
    d = (y_pert - y_ref).abs().view(B, K, S, -1).amax(dim=(0, 2, 3)); d0 = (y_pert0 - y_ref).abs().view(B, K, S, -1).amax(dim=(0, 2, 3))
    results["causal_mask_check"] = dict(perturb_sensor_frame=K - 1, max_abs_change_per_channel_frame=[float(v) for v in d], frames_lt_15_unchanged=bool(d[:K - 1].max() == 0), frame_15_changed=bool(d[K - 1] > 0),
                                        perturb_frame0_max_abs_change_per_channel_frame=[float(v) for v in d0], all_frames_changed_when_frame0_perturbed=bool((d0 > 0).all()),
                                        mask_shape=list(m._mask(K, sum(m.n_tok[x] for x in m.token_mods), dev).shape), mask_blocked_frac=float(m._mask(K, sum(m.n_tok[x] for x in m.token_mods), dev).float().mean()))
    lp("CAUSAL_MASK_CHECK", json.dumps(results["causal_mask_check"])); del m
except Exception as e:
    results["causal_mask_check"] = dict(error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:]); lp("CAUSAL_MASK_CHECK ERR", e)
torch.cuda.empty_cache()

# ---------------- (3) 로더 실측 ----------------
try:
    rng = np.random.RandomState(0); ids_cold = rng.choice(len(ws), a.train_B, replace=False).tolist()
    st0 = dict(ws.store.stats); w0 = dict(ws.stats)
    torch.cuda.synchronize(); t = time.time(); _ = ws.batch(ids_cold, dev); torch.cuda.synchronize(); cold = time.time() - t
    st1 = dict(ws.store.stats)
    torch.cuda.synchronize(); t = time.time(); _ = ws.batch(ids_cold, dev); torch.cuda.synchronize(); warm = time.time() - t
    times = []; bytes_before = ws.store.stats["disk_bytes"]
    for it in range(20):
        ids = rng.choice(len(ws), a.train_B, replace=False).tolist(); torch.cuda.synchronize(); t = time.time(); _ = ws.batch(ids, dev); torch.cuda.synchronize(); times.append(round(time.time() - t, 4))
    ch_only_t = []
    for it in range(5):
        ids = rng.choice(len(ws), a.train_B, replace=False).tolist(); torch.cuda.synchronize(); t = time.time(); _ = super(MMWindowSet, ws).batch(ids, dev); torch.cuda.synchronize(); ch_only_t.append(round(time.time() - t, 4))
    # 정상상태(프레임 전부 RAM, 창 LRU 미적중 = 학습 중 무작위 배치의 실제 경로): 궤적 0~2(같은 RSU 프레임 1,100개 공유) 창만으로 40배치 워밍업 후 20배치 측정, 창 LRU 우회
    wc_save = ws.window_cache; ws.window_cache = 1; ws._wlru.clear(); tr_ids = [i for i, (w_, r_) in enumerate(ws.items) if ws.windows[w_][1] in (0, 1, 2)]
    for it in range(40): _ = ws.batch(rng.choice(tr_ids, a.train_B, replace=False).tolist(), dev)
    d_ss = ws.store.stats["disk_bytes"]; ss = []; ss_ch = []
    for it in range(20):
        ids = rng.choice(tr_ids, a.train_B, replace=False).tolist(); torch.cuda.synchronize(); t = time.time(); _ = ws.batch(ids, dev); torch.cuda.synchronize(); ss.append(round(time.time() - t, 4))
    for it in range(10):
        ids = rng.choice(tr_ids, a.train_B, replace=False).tolist(); torch.cuda.synchronize(); t = time.time(); _ = super(MMWindowSet, ws).batch(ids, dev); torch.cuda.synchronize(); ss_ch.append(round(time.time() - t, 4))
    steady = dict(batch_sec_mean=round(float(np.mean(ss)), 4), batch_sec_std=round(float(np.std(ss)), 4), channel_part_sec_mean=round(float(np.mean(ss_ch)), 4), extra_disk_bytes=int(ws.store.stats["disk_bytes"] - d_ss), note="frames in RAM(궤적 0~2), window LRU bypass")
    ws.window_cache = wc_save
    wb = ws.window_bytes(); n_frames = len(ws.store._lru)
    results["loader"] = dict(steady_state=steady,train_B=a.train_B, cold_batch_sec=round(cold, 4), warm_batch_sec=round(warm, 4), random_batches_sec=times, random_batches_mean_sec=round(float(np.mean(times)), 4),
                             channel_only_batch_sec=ch_only_t, disk_bytes_cold_batch=int(st1["disk_bytes"] - st0["disk_bytes"]), disk_bytes_20_random_batches=int(ws.store.stats["disk_bytes"] - bytes_before),
                             window_bytes=wb, window_bytes_total=int(sum(wb.values())), frame_bytes=ws.store.row_bytes, frames_in_ram=n_frames, ram_bytes_frames=int(n_frames * sum(ws.store.row_bytes.values())),
                             rsu_frames_total=sum(len(v) for (mm_, _), v in ws.store.row.items() if mm_ == "cam"), full_ram_estimate_bytes=int(sum(len(v) for (mm_, _), v in ws.store.row.items() if mm_ == "cam") * sum(ws.store.row_bytes.values())),
                             stats=ws.stat_line(), n_train_items=len(ws), n_train_windows=len(win["train"]))
    lp("LOADER", json.dumps(results["loader"]))
except Exception as e:
    results["loader"] = dict(error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:]); lp("LOADER ERR", e)

# ---------------- (4) 학습 비용 (B=32) ----------------
ids32 = np.random.RandomState(1).choice(len(ws), a.train_B, replace=False).tolist()
X32, Y32, sc32, _, S32 = ws.batch(ids32, dev)
def loss_fn(pred, Yt): return (((pred - Yt) ** 2).sum(dim=(2, 3, 4)).sum(1) / (Yt ** 2).sum(dim=(2, 3, 4)).sum(1).clamp_min(1e-12)).mean()
steps_per_epoch = math.ceil(len(ws) / a.train_B); val_batches = math.ceil(len(win["val"]) * Nr / a.train_B)
def train_cost(name, build, fwd_fn, grad_check=False):
    torch.manual_seed(0); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); rec = {}
    try:
        m = build().to(dev); m.train(); opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4); ts = []; gc = []
        for it in range(2 + a.steps):
            torch.cuda.synchronize(); t = time.time()
            pred = fwd_fn(m); loss = loss_fn(pred, Y32); opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            if grad_check and it < 3:
                g = {n: float(p.grad.norm()) for n, p in m.named_parameters() if p.grad is not None and (n.startswith("fusion.0.attn.out_proj.weight") or n.startswith("fusion.0.gate.weight") or n.startswith("enc.cam.proj.weight") or n.startswith("enc.lidar.proj.weight") or n.startswith("enc.radar.out.weight") or n.startswith("enc.pos.mlp.2.weight"))}
                gc.append(dict(step=it, loss=float(loss), grad_norms={k: round(v, 6) for k, v in g.items()}))
            opt.step(); torch.cuda.synchronize(); ts.append(time.time() - t)
        m.eval(); torch.cuda.synchronize(); t = time.time()
        with torch.no_grad(): fwd_fn(m)
        torch.cuda.synchronize(); ev = time.time() - t
        step = float(np.mean(ts[2:])); rec = dict(params=count_params(m), step_sec_mean=round(step, 4), step_sec_std=round(float(np.std(ts[2:])), 4), eval_fwd_sec=round(ev, 4), peak_mem_mb=peak(), loss_first=round(float(loss), 4),
                                                   epoch_est_sec=round(step * steps_per_epoch + ev * val_batches, 0), epoch_est_hr=round((step * steps_per_epoch + ev * val_batches) / 3600, 2), steps_per_epoch=steps_per_epoch, val_batches=val_batches)
        if isinstance(m, MMChiron): rec["gate_means_after_steps"] = m.gate_means()
        if gc: rec["grad_wakeup"] = gc
        lp(f"TRAIN {name:26s} params {rec['params']/1e6:7.3f}M step {step*1000:.0f}±{np.std(ts[2:])*1000:.0f} ms eval fwd {ev*1000:.0f} ms peak {rec['peak_mem_mb']:.0f}MB → epoch ≈ {rec['epoch_est_hr']:.2f} h" + (f" gate {rec['gate_means_after_steps']}" if 'gate_means_after_steps' in rec else ""))
        del m, opt
    except Exception as e:
        rec = dict(error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:]); lp(f"TRAIN ERR {name}: {rec['error']}")
    torch.cuda.empty_cache(); return rec
results["train_cost"]["chiron_channel_only(repo:chiron)"] = train_cost("chiron_channel_only", lambda: build_model("repo:chiron", K, H, Nt, Ksc, D=256, L=6), lambda m: m(X32))
for sens, pm in CONFIGS:
    sub = {k: v for k, v in S32.items() if k in sens.split(",")}
    results["train_cost"][cname(sens, pm)] = train_cost(cname(sens, pm), lambda sens=sens, pm=pm: build_model("mm:chiron", K, H, Nt, Ksc, sensors=sens, pos_mode=pm, fuse_layers=3), lambda m, sub=sub: m(X32, sub), grad_check=(sens == "cam,lidar,radar,pos"))
results["train_cost"]["cam+lidar+radar+pos(causal)"] = train_cost("full(causal_mask)", lambda: build_model("mm:chiron", K, H, Nt, Ksc, sensors="cam,lidar,radar,pos", fuse_layers=3, causal_mask=True), lambda m: m(X32, S32))
results["train_cost"]["loader_per_batch_sec"] = dict(warm_same_batch=results["loader"].get("warm_batch_sec"), random_with_disk=results["loader"].get("random_batches_mean_sec"), steady_state=results["loader"].get("steady_state", {}).get("batch_sec_mean"))

# ---------------- (5) 커버리지 + 위치 변환 점검 ----------------
try:
    rep, chk = coverage_report(idx, a.cache_root, log=lp); du = {d: sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(f"{a.cache_root}/{d}") for f in fs) for d in ("lidar_pool8", "radar_raster64", "pos") if os.path.isdir(f"{a.cache_root}/{d}")}
    metas = {}
    for d in ("lidar_pool8", "radar_raster64"):
        f = f"{a.cache_root}/{d}/Town03/Town03_5wayroad/meta.json"
        if os.path.exists(f): metas[d] = json.load(open(f))
    results["coverage"] = dict(coverage=rep, pos_check=chk, cache_bytes=du, cache_root=a.cache_root, meta_example=metas, pos_rsu_info=ws.pos.rsu_info)
except Exception as e:
    results["coverage"] = dict(error=f"{type(e).__name__}: {e}", tb=traceback.format_exc()[-800:]); lp("COVERAGE ERR", e)

results["summary"] = {k: dict(params=v.get("params"), enc=v.get("groups", {}).get("sensor_encoders"), fusion=v.get("groups", {}).get("fusion"), sensor_tokens=v.get("sensor_tokens"), fwd_sec=v.get("fwd_sec"), peak_mb=v.get("peak_mem_mb"),
                              gate=v.get("gate_means_init"), max_diff=v.get("max_abs_diff_vs_channel_only_chiron"), step_sec=results["train_cost"].get(k, {}).get("step_sec_mean"), train_peak_mb=results["train_cost"].get(k, {}).get("peak_mem_mb"),
                              epoch_hr=results["train_cost"].get(k, {}).get("epoch_est_hr"), err=v.get("error")) for k, v in results["configs"].items()}
results["env"]["peak_mem_mb_overall"] = peak()
json.dump(results, open(a.out, "w"), indent=1, default=str); lp("saved", a.out, "| overall peak MB", results["env"]["peak_mem_mb_overall"])
