"""멀티모달 채널 예측 run(`mm:chiron`) 의 **센서 기여 절제 평가** — 학습 없음, best.pt 재평가만 (2026-09-07, 신규 파일).
계획 EXPERIMENT_PLAN_MULTIMODAL_20260907.md §3(절제 프로토콜)·§4(판정 규칙), 진단 훅 MULTIMODAL_CP_ARCHITECTURE.md §3.

실행: CUDA_VISIBLE_DEVICES=0 python eval_sensor_ablation_cp.py --run mm_all_lr3e-4 --device cuda:0 --batch 32 [--max_windows N] [--per_modal] [--check_full_real] [--dry_run]
출력: <out_root>/<run>/sensor_ablation.json (이 파일 1개만 씀) + 콘솔 요약 표(조건 × 지평 median dB, real 대비 Δ).

조건(입력만 바꿈, 모델·가중치·타깃 Y 는 모든 조건에서 동일; 지표 = cp_data.nmse_per_sample / summarize 그대로):
  real           정상 입력. result.json val.median_db/pooled_db 와 일치해야 함(자체 대조해 diff 출력; --max_windows 부분집합이면 --check_full_real 로 전체 val 을 real 만 다시 평가해 대조)
  zero           모든 센서 텐서 0(cam/lidar/radar 캐시 0, pos 특징 0). 인코더는 상수 토큰을 내고 융합 블록은 실행됨
  shuffle        센서만 다른 창의 것으로 교체(채널·타깃 유지). **창 단위** seed 고정 derangement(val 전체 창, 상대 ≠ 자기 창) — 같은 창의 RX 16행은
                 센서가 동일하므로 표본(행) 단위 in-batch randperm 은 같은 창끼리 맞바꿔 무효가 될 수 있음. 창 단위 매핑이면 RX 16행 전부가 같은 상대 창의 센서를 받음
  foreign        센서만 같은 val split 의 **다른 시나리오** 창의 것으로 교체(seed 고정 매핑)
  sensor_only    채널 이력 X = 0, 센서만 실제
  hist_shuffle   채널 이력 X 를 다른 창의 것으로 교체(타깃·센서 유지) — eval_sanity 의 정의("이력을 다른 창의 것으로 교체")를 shuffle 과 같은 상대 창·같은 RX 행으로 구현
                 (eval_sanity 의 in-batch randperm 은 batch ≤ 16 이면 같은 창의 RX 행이 상대가 되어 검사가 무력해지므로 창 단위로 보장)
  last_zero      X 의 마지막 프레임만 0                      [eval_sanity 동일]
  hist_only_last X 의 마지막 프레임만 남기고 나머지 0          [eval_sanity 동일]
  none           [extra] sensors dict 비움 → 융합 블록(및 pos 가산) 미실행 = 공동 학습된 백본 단독 경로
  shuffle_scene  [extra] 센서만 **같은 시나리오**의 다른 창의 것으로 교체(seed 고정 derangement) — 시간 정렬만 깨고 장면 통계는 유지(shuffle 은 2/3 이 타 장면)
  zero_<m>/shuffle_<m>  (--per_modal, 모달 2개 이상일 때) 모달 m 만 0 / 상대 창 것으로 교체, 나머지 모달은 실제
진단(real 입력, diag=True forward 1회 추가): 층별 게이트 평균 g(model.gate_means), 센서 토큰이 받은 어텐션 질량(model.attention_mass: 모달별 합, 토큰당 정규화, 정규화 점유율),
  프레임별 질량(last_attn 재집계), forward hook 으로 ‖g⊙A‖/‖Z‖·‖A‖/‖Z‖(항등성 지표, 문서 §3 주의 항목). 게이트 평균은 모든 조건에서 무료로 기록.
"""
import os, sys, json, time, argparse, datetime
from collections import OrderedDict
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_data import load_index, make_windows, WindowSet, nmse_per_sample, summarize, db
from cp_models import build_model

COND_DEFS = OrderedDict([
    ("real", "정상 입력(채널 이력 X + 창의 실제 센서). result.json val 과 일치해야 함"),
    ("zero", "모든 센서 텐서 0(cam/lidar/radar 캐시 0, pos 특징 0); 인코더 상수 토큰, 융합 블록 실행됨"),
    ("shuffle", "센서만 다른 창의 것으로 교체(val 전체 창의 seed 고정 derangement, 상대≠자기 창, 창 단위 → RX 16행 동일 상대), 채널·타깃 유지"),
    ("foreign", "센서만 같은 val split 의 다른 시나리오 창의 것으로 교체(seed 고정 매핑), 채널·타깃 유지"),
    ("sensor_only", "채널 이력 X=0, 센서만 실제"),
    ("hist_shuffle", "채널 이력 X 를 다른 창(shuffle 과 같은 상대 창, 같은 RX 행)의 것으로 교체, 타깃·센서 유지 [eval_sanity 정의]"),
    ("last_zero", "X 의 마지막 프레임만 0 [eval_sanity 동일]"),
    ("hist_only_last", "X 의 마지막 프레임만 남기고 나머지 0 [eval_sanity 동일]"),
    ("none", "[extra] sensors dict 비움 → 융합 블록·pos 가산 미실행(공동 학습된 백본 단독 경로)"),
    ("shuffle_scene", "[extra] 센서만 같은 시나리오의 다른 창의 것으로 교체(seed 고정 derangement), 채널·타깃 유지"),
])
PER_MODAL_DEFS = {"zero_": "모달 {m} 만 0, 나머지 모달 실제", "shuffle_": "모달 {m} 만 상대 창(shuffle 매핑) 것으로 교체, 나머지 모달 실제"}
BASE_ORDER = list(COND_DEFS.keys())

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--out_root", default=os.environ.get("CP_OUT", "/mnt/ssd_7t_2/carla-wireless-dataset/mmw_reproduction/outputs_cp"))
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--max_windows", type=int, default=0, help="val 창 부분집합(등간격 추출, 0=전체). 상대 창 매핑은 항상 val 전체 창 기준(seed 고정)")
ap.add_argument("--per_modal", action="store_true", help="모달 2개 이상이면 zero_<m>/shuffle_<m> 도 계산")
ap.add_argument("--conditions", default="all", help="쉼표 목록으로 조건 제한(기본 all)")
ap.add_argument("--check_full_real", action="store_true", help="--max_windows 사용 시 real 만 val 전체로 다시 평가해 result.json 과 대조")
ap.add_argument("--seed", type=int, default=0, help="shuffle/foreign/shuffle_scene 상대 창 매핑 seed")
ap.add_argument("--dry_run", action="store_true", help="조건 정의·실행 계획만 출력")
a = ap.parse_args()

d = f"{a.out_root}/{a.run}"; cfg = json.load(open(f"{d}/config.json"))
assert cfg["model"].startswith("mm:"), f"mm: 모델 전용 (config model={cfg['model']}); 채널 전용 run 은 eval_sanity.py"
sensors_cfg = [s for s in cfg.get("sensors", "").split(",") if s]
pos_mode = cfg.get("pos_mode", "token")
token_mods = [m for m in sensors_cfg if not (m == "pos" and pos_mode == "broadcast")]   # 융합 kv 토큰을 내는 모달(pos broadcast 는 가산 경로)

def active_conditions():
    conds = list(BASE_ORDER)
    if a.per_modal and len(sensors_cfg) > 1:
        conds += [f"zero_{m}" for m in sensors_cfg] + [f"shuffle_{m}" for m in sensors_cfg]
    if a.conditions != "all":
        want = [c.strip() for c in a.conditions.split(",") if c.strip()]; bad = [c for c in want if c not in conds]; assert not bad, f"unknown conditions {bad}; available {conds}"
        conds = [c for c in conds if c in want]
    return conds

def cond_def(c):
    if c in COND_DEFS: return COND_DEFS[c]
    for pre, t in PER_MODAL_DEFS.items():
        if c.startswith(pre): return t.format(m=c[len(pre):])
    raise KeyError(c)

CONDS = active_conditions()
if a.dry_run:
    print(f"[dry_run] run {a.run} model {cfg['model']} sensors {sensors_cfg} pos_mode {pos_mode} pos_source {cfg.get('pos_source')} fuse_layers {cfg.get('fuse_layers')} causal_mask {cfg.get('causal_mask')}")
    print(f"[dry_run] val split {cfg['split']} n_val(result) {cfg.get('n_val')} | batch {a.batch} max_windows {a.max_windows or 'all'} seed {a.seed} per_modal {a.per_modal}")
    for c in CONDS: print(f"  {c:16s} {cond_def(c)}")
    print(f"[dry_run] 진단(real): gate_means(층별), attention_mass(모달별 합·토큰당·점유율), 프레임별 질량, ‖g⊙A‖/‖Z‖, ‖A‖/‖Z‖ | 출력 {d}/sensor_ablation.json"); sys.exit(0)

# ---------------------------------------------------------------------------------------------------------------- data / model
t_start = time.time(); dev = torch.device(a.device)
ck = torch.load(f"{d}/best.pt", map_location="cpu", weights_only=False)
idx, mm = load_index(cfg["setting"], cfg["Ksc"]); F = torch.from_numpy(np.asarray(mm)); Nr = F.shape[1]; K, H = cfg["K_hist"], cfg["H_pred"]
win, _ = make_windows(idx, K, H, split=cfg["split"]); val_win = win["val"]; n_win_all = len(val_win)
from cp_sensor_data import MMWindowSet
va = MMWindowSet(F, val_win, K, H, Nr, idx, cfg["sensors"], pos_source=cfg.get("pos_source", "predicted"), cache_root=cfg.get("sensor_cache_root") or None, log=lambda *s: print(*s, flush=True))
kw = dict(D=cfg["D"], L=cfg["L"], heads=cfg["heads"], dropout=cfg["dropout"], pos_emb=not cfg["no_pos_emb"], c_hid=cfg["c_hid"], c_enc=cfg["c_enc"], residual=cfg.get("residual", False), patch_h=cfg.get("patch_h", 4), patch_w=cfg.get("patch_w", 16),
          lwm11_init=cfg.get("lwm11_init", "pretrained"), lwm11_freeze=cfg.get("lwm11_freeze", "none"), in_scale=cfg.get("in_scale", 1.0), t_layers=cfg.get("t_layers", 2))
kw.update(sensors=cfg["sensors"], pos_source=cfg.get("pos_source", "predicted"), pos_mode=pos_mode, fuse_layers=cfg.get("fuse_layers", 3), causal_mask=cfg.get("causal_mask", False),
          backbone_init="")   # 백본 초기 ckpt 는 best.pt 가 덮어쓰므로 로드 생략(파일 이동에도 안전)
model = build_model(cfg["model"], K, H, F.shape[2], F.shape[3], **kw).to(dev); missing = model.load_state_dict(ck["model"], strict=True); model.eval()
speed = np.asarray(idx["speed"], np.float32); trajs = idx["trajs"]
scene_of_win = np.array([trajs[ti]["scen"] for _, ti in val_win]); traj_of_win = np.array([ti for _, ti in val_win]); start_of_win = np.array([s for s, _ in val_win])
print(f"[{a.run}] model {cfg['model']} sensors {model.sensors} token_mods {model.token_mods} n_tok {model.n_tok} pos_mode {pos_mode} fuse_layers {len(model.fusion)} | val windows {n_win_all} (x{Nr}) scenes {sorted(map(str, set(scene_of_win)))} | ckpt ep {ck.get('epoch')} | load {time.time()-t_start:.0f}s", flush=True)

# ---------------------------------------------------------------------------------------------------------------- 상대 창 매핑(val 전체 창 기준, seed 고정)
def derangement(n, rng):
    p = rng.permutation(n); fp = np.where(p == np.arange(n))[0]
    if len(fp) == 1: i = fp[0]; j = int(rng.integers(n - 1)); j += (j >= i); p[i], p[j] = p[j], p[i]
    elif len(fp) > 1: p[fp] = p[np.roll(fp, 1)]
    assert (p != np.arange(n)).all(); return p

rng = np.random.default_rng(a.seed)
perm_shuffle = derangement(n_win_all, rng)
perm_scene = np.full(n_win_all, -1)
for sc in sorted(set(scene_of_win)):
    g = np.where(scene_of_win == sc)[0]
    if len(g) >= 2: perm_scene[g] = g[derangement(len(g), rng)]
perm_foreign = np.full(n_win_all, -1); others = {sc: np.where(scene_of_win != sc)[0] for sc in set(scene_of_win)}
for w in range(n_win_all):
    o = others[scene_of_win[w]]
    if len(o): perm_foreign[w] = o[int(rng.integers(len(o)))]
def map_stats(p):
    ok = p >= 0; q = p[ok]; w = np.arange(n_win_all)[ok]; same_tr = traj_of_win[q] == traj_of_win[w]
    return dict(n_mapped=int(ok.sum()), n_unmapped=int((~ok).sum()), frac_same_scene=float((scene_of_win[q] == scene_of_win[w]).mean()) if ok.any() else None, frac_same_traj=float(same_tr.mean()) if ok.any() else None,
                median_abs_frame_gap_same_traj=float(np.median(np.abs(start_of_win[q][same_tr] - start_of_win[w][same_tr]))) if same_tr.any() else None, seed=a.seed)
partner_stats = dict(shuffle=map_stats(perm_shuffle), shuffle_scene=map_stats(perm_scene), foreign=map_stats(perm_foreign))
assert partner_stats["foreign"]["n_mapped"] == 0 or partner_stats["foreign"]["frac_same_scene"] == 0.0
assert partner_stats["shuffle_scene"]["n_mapped"] == 0 or partner_stats["shuffle_scene"]["frac_same_scene"] == 1.0
unavailable = {c for c, p in (("foreign", perm_foreign), ("shuffle_scene", perm_scene)) if (p < 0).any()}
if unavailable: print(f"[warn] 상대 창 없는 창 존재 → 조건 제외: {sorted(unavailable)} {partner_stats}", flush=True); CONDS = [c for c in CONDS if c not in unavailable]

# ---------------------------------------------------------------------------------------------------------------- 평가 대상 창(부분집합은 등간격)
sel_win = np.arange(n_win_all) if not a.max_windows or a.max_windows >= n_win_all else np.unique(np.linspace(0, n_win_all - 1, a.max_windows).round().astype(int))
def item_ids_of(wins): return [int(w) * Nr + r for w in wins for r in range(Nr)]

# ---------------------------------------------------------------------------------------------------------------- 진단 hook(융합 블록 입력 Z, 어텐션 출력 A, 게이트 logit)
class FusionProbe:
    def __init__(self, model):
        self.cap = [dict() for _ in model.fusion]; self.h = []
        for li, blk in enumerate(model.fusion):
            self.h.append(blk.register_forward_pre_hook(lambda mod, args, li=li: self.cap[li].__setitem__("x", args[0].detach())))
            self.h.append(blk.attn.register_forward_hook(lambda mod, args, out, li=li: self.cap[li].__setitem__("a", out[0].detach())))
            self.h.append(blk.gate.register_forward_hook(lambda mod, args, out, li=li: self.cap[li].__setitem__("logit", out.detach())))
    def ratios(self):
        out = []
        for c in self.cap:
            x, a_, g = c["x"], c["a"], torch.sigmoid(c["logit"]); nx = x.flatten(1).norm(dim=1).clamp_min(1e-12)
            out.append(dict(upd_over_z=float(((g * a_).flatten(1).norm(dim=1) / nx).mean()), attn_over_z=float((a_.flatten(1).norm(dim=1) / nx).mean())))
        return out
    def remove(self):
        for h in self.h: h.remove()

# ---------------------------------------------------------------------------------------------------------------- 조건별 입력 구성
def stack_sensors(dicts, dev):
    if not dicts or not dicts[0]: return {}
    return {m: torch.stack([dd[m] for dd in dicts]).to(dev, non_blocking=True) for m in dicts[0]}

def make_inputs(cond, X, S, Ssh, Sfo, Ssc, Xp):
    if cond == "real": return X, S
    if cond == "zero": return X, {m: torch.zeros_like(t) for m, t in S.items()}
    if cond == "shuffle": return X, Ssh
    if cond == "foreign": return X, Sfo
    if cond == "shuffle_scene": return X, Ssc
    if cond == "sensor_only": return torch.zeros_like(X), S
    if cond == "none": return X, {}
    if cond == "hist_shuffle": return Xp, S
    if cond == "last_zero": Xc = X.clone(); Xc[:, -1] = 0; return Xc, S
    if cond == "hist_only_last": Xc = X.clone(); Xc[:, :-1] = 0; return Xc, S
    if cond.startswith("zero_"): m = cond[5:]; dd = dict(S); dd[m] = torch.zeros_like(S[m]); return X, dd
    if cond.startswith("shuffle_"): m = cond[8:]; dd = dict(S); dd[m] = Ssh[m]; return X, dd
    raise KeyError(cond)

def sync():
    if dev.type == "cuda": torch.cuda.synchronize(dev)

@torch.no_grad()
def run_eval(wins, conds, diag, tag):
    ids = item_ids_of(wins); n = len(ids)
    need_sh = any(c in ("shuffle", "hist_shuffle") or c.startswith("shuffle_") and c != "shuffle_scene" for c in conds); need_fo = "foreign" in conds; need_sc = "shuffle_scene" in conds; need_xp = "hist_shuffle" in conds
    acc = {c: dict(raw=[], align=[], rho=[], err=np.zeros(H), gate=None, sec=0.0) for c in conds}; pw_sum = np.zeros(H); metas = []
    dg = dict(n=0, gate=None, mass=None, frame=None, ratio=None); probe = FusionProbe(model) if (diag and len(model.fusion)) else None; selfcheck = None; t0 = time.time(); load_sec = 0.0
    for bi, b in enumerate(range(0, n, a.batch)):
        bid = ids[b:b + a.batch]; tl = time.time()
        X, Y, scale, meta = WindowSet.batch(va, bid, dev); ws = [va.items[i][0] for i in bid]; rs = [va.items[i][1] for i in bid]; B = len(bid)
        S = stack_sensors([va.window_sensors(w) for w in ws], dev)
        Ssh = stack_sensors([va.window_sensors(int(perm_shuffle[w])) for w in ws], dev) if need_sh else None
        Sfo = stack_sensors([va.window_sensors(int(perm_foreign[w])) for w in ws], dev) if need_fo else None
        Ssc = stack_sensors([va.window_sensors(int(perm_scene[w])) for w in ws], dev) if need_sc else None
        Xp = WindowSet.batch(va, [int(perm_shuffle[w]) * Nr + r for w, r in zip(ws, rs)], dev)[0] if need_xp else None
        if bi == 0:   # 로더 자체 점검: 직접 조립한 (X, Y, sensors) 가 MMWindowSet.batch(train_cp 경로) 와 동일한가
            X2, Y2, _, _, S2 = va.batch(bid, dev)
            selfcheck = dict(X_equal=bool(torch.equal(X, X2)), Y_equal=bool(torch.equal(Y, Y2)), sensors_equal={m: bool(torch.equal(S[m], S2[m])) for m in S}, modals=list(S.keys()), shapes={m: list(S[m].shape) for m in S})
            assert selfcheck["X_equal"] and selfcheck["Y_equal"] and all(selfcheck["sensors_equal"].values()), selfcheck
        load_sec += time.time() - tl; metas += meta; pw_sum += (Y ** 2).sum(dim=(0, 2, 3, 4)).cpu().numpy()
        for c in conds:
            Xc, Sc = make_inputs(c, X, S, Ssh, Sfo, Ssc, Xp); sync(); tc = time.time()
            pred = model(Xc, Sc); raw, align, rho = nmse_per_sample(pred, Y); sync(); acc[c]["sec"] += time.time() - tc
            acc[c]["raw"].append(raw.cpu().numpy()); acc[c]["align"].append(align.cpu().numpy()); acc[c]["rho"].append(rho.cpu().numpy()); acc[c]["err"] += ((pred - Y) ** 2).sum(dim=(0, 2, 3, 4)).cpu().numpy()
            gm = model.gate_means()
            if gm and all(g is not None for g in gm): acc[c]["gate"] = (acc[c]["gate"] if acc[c]["gate"] is not None else np.zeros(len(gm))) + np.array(gm) * B
        if probe is not None:   # real 입력 진단 forward(diag=True: 어텐션 가중치 저장; need_weights 경로라 real 지표와 분리)
            model(X, S, diag=True); gm = np.array(model.gate_means(), dtype=np.float64); mass = model.attention_mass(); rat = probe.ratios()
            fr = [blk.last_attn.mean(dim=(0, 1)).view(K, -1).sum(1).cpu().numpy() if blk.last_attn is not None else None for blk in model.fusion]
            if dg["n"] == 0: dg.update(gate=np.zeros(len(gm)), mass=[{m: 0.0 for m in (ms or {})} for ms in mass], frame=[np.zeros(K) for _ in fr], ratio=[dict(upd_over_z=0.0, attn_over_z=0.0) for _ in rat])
            dg["gate"] += gm * B
            for li in range(len(mass)):
                if mass[li]:
                    for m, v in mass[li].items(): dg["mass"][li][m] += v * B
                if fr[li] is not None: dg["frame"][li] += fr[li] * B
                for k2 in dg["ratio"][li]: dg["ratio"][li][k2] += rat[li][k2] * B
            dg["n"] += B
        if bi % 50 == 0 or b + a.batch >= n: print(f"  [{tag}] {min(b + a.batch, n)}/{n} {time.time()-t0:.0f}s (load {load_sec:.0f}s) mem {torch.cuda.max_memory_allocated()/2**20:.0f} MB", flush=True)
    if probe is not None: probe.remove()
    sp_last = np.array([speed[s + K - 1] for _, s in metas]); out = OrderedDict()
    for c in conds:
        raw = np.concatenate(acc[c]["raw"]); summ = summarize(raw, np.concatenate(acc[c]["align"]), np.concatenate(acc[c]["rho"]), acc[c]["err"], pw_sum, metas, idx, sp_last)
        summ["def"] = cond_def(c); summ["gate_mean"] = (acc[c]["gate"] / n).tolist() if acc[c]["gate"] is not None else None; summ["forward_sec"] = round(acc[c]["sec"], 2); out[c] = summ
    diag_out = None
    if dg["n"]:
        nt = model.n_tok; mass = [{m: v / dg["n"] for m, v in ms.items()} for ms in dg["mass"]]
        per_tok = [{m: v / nt[m] for m, v in ms.items()} for ms in mass]; share = [{m: v / max(1e-12, sum(ms.values())) for m, v in ms.items()} for ms in per_tok]
        diag_out = dict(n=dg["n"], gate_mean=(dg["gate"] / dg["n"]).tolist(), attention_mass=mass, attention_mass_per_token=per_tok, attention_share_per_token=share, n_tok={m: nt[m] for m in model.token_mods},
                        attention_frame_mass=[(f / dg["n"]).tolist() for f in dg["frame"]], fusion_norm_ratio=[{k2: v / dg["n"] for k2, v in r.items()} for r in dg["ratio"]],
                        notes="attention_mass: 층별 평균 어텐션(head 평균, 채널 질의 512·배치 평균)을 모달 토큰 구획별로 합산(합 1, 토큰 수에 비례하므로 per_token·share 로 비교). frame_mass: 이력 프레임 k 의 센서 토큰이 받은 질량(K개, 합 1). "
                              "fusion_norm_ratio: upd_over_z=‖g⊙A‖/‖Z‖(블록 입력 대비 게이트된 갱신량, 표본별 Frobenius 비 평균; 0 이면 항등), attn_over_z=‖A‖/‖Z‖")
    return out, diag_out, selfcheck, dict(n=n, n_windows=len(wins), sec=round(time.time() - t0, 1), load_sec=round(load_sec, 1))

# ---------------------------------------------------------------------------------------------------------------- 실행
torch.cuda.reset_peak_memory_stats(dev)
res, diag, selfcheck, meta_main = run_eval(sel_win, CONDS, diag=True, tag="main")
rj = json.load(open(f"{d}/result.json")) if os.path.exists(f"{d}/result.json") else dict(val=ck.get("val", {}))
def compare_real(r):
    rv = rj.get("val", {}); md = np.array(r["median_db"]) - np.array(rv.get("median_db", [np.nan] * H)); pd_ = np.array(r["pooled_db"]) - np.array(rv.get("pooled_db", [np.nan] * H))
    comparable = int(r["n"]) == int(rv.get("n", -1)); mx = float(np.nanmax(np.abs(np.concatenate([md, pd_]))))
    return dict(comparable=comparable, n_eval=int(r["n"]), n_result_json=rv.get("n"), result_median_db=rv.get("median_db"), result_pooled_db=rv.get("pooled_db"), median_diff_db=md.tolist(), pooled_diff_db=pd_.tolist(), max_abs_diff_db=mx,
                ok_within_0p01=(bool(mx <= 0.01) if comparable else None), note=("전체 val 이므로 정확 비교" if comparable else "부분집합(--max_windows) → 표본이 달라 정확 비교 불가; --check_full_real 결과 참조"))
check = {"main_real_vs_result_json": compare_real(res["real"])} if "real" in res else {}
if a.check_full_real and len(sel_win) < n_win_all:
    full, _, _, meta_full = run_eval(np.arange(n_win_all), ["real"], diag=False, tag="full_real"); check["full_real_vs_result_json"] = compare_real(full["real"]); check["full_real"] = full["real"]; check["full_real_meta"] = meta_full
peak_mb = torch.cuda.max_memory_allocated(dev) / 2**20; peak_res_mb = torch.cuda.max_memory_reserved(dev) / 2**20

# ---------------------------------------------------------------------------------------------------------------- 콘솔 표
hs = [f"h{i+1}" for i in range(H)]; rm = np.array(res["real"]["median_db"]) if "real" in res else None; rp = np.array(res["real"]["pooled_db"]) if "real" in res else None
print(f"\n== sensor_ablation {a.run} | n {meta_main['n']} ({meta_main['n_windows']}/{n_win_all} windows x {Nr} RX) | median NMSE dB (Δ vs real) ==")
print(f"{'condition':16s} " + " ".join(f"{h:>7s}" for h in hs) + " | " + " ".join(f"{'Δ'+h:>7s}" for h in hs) + " | pooled " + " ".join(f"{h:>6s}" for h in hs) + " | gate")
for c, r in res.items():
    m = np.array(r["median_db"]); p = np.array(r["pooled_db"]); dm = (m - rm) if rm is not None else np.full(H, np.nan)
    print(f"{c:16s} " + " ".join(f"{v:7.2f}" for v in m) + " | " + " ".join(f"{v:+7.2f}" for v in dm) + " | " + " " * 7 + " ".join(f"{v:6.2f}" for v in p) + " | " + (" ".join(f"{g:.3f}" for g in r["gate_mean"]) if r["gate_mean"] else "-"))
if diag:
    print("\n[diag real] gate_mean per layer", np.round(diag["gate_mean"], 4).tolist(), "| ‖g⊙A‖/‖Z‖", [round(r["upd_over_z"], 4) for r in diag["fusion_norm_ratio"]], "| ‖A‖/‖Z‖", [round(r["attn_over_z"], 4) for r in diag["fusion_norm_ratio"]])
    for li, (ms, pt, sh) in enumerate(zip(diag["attention_mass"], diag["attention_mass_per_token"], diag["attention_share_per_token"])):
        print(f"  L{li+1} mass {json.dumps({m: round(v, 4) for m, v in ms.items()})} per_token {json.dumps({m: round(v, 4) for m, v in pt.items()})} share {json.dumps({m: round(v, 3) for m, v in sh.items()})} frame(last4) {np.round(diag['attention_frame_mass'][li][-4:], 4).tolist()}")
for k2, v in check.items():
    if k2.endswith("vs_result_json"): print(f"[check {k2}] comparable {v['comparable']} n {v['n_eval']} vs {v['n_result_json']} | median diff {np.round(v['median_diff_db'], 4).tolist()} pooled diff {np.round(v['pooled_diff_db'], 4).tolist()} | max|diff| {v['max_abs_diff_db']:.4f} dB ok≤0.01 {v['ok_within_0p01']}")
print(f"[mem] peak allocated {peak_mb:.0f} MB reserved {peak_res_mb:.0f} MB | total {time.time()-t_start:.0f}s | loader {json.dumps(va.stat_line())}")

# ---------------------------------------------------------------------------------------------------------------- 저장(run 폴더에 이 파일 1개만)
for c, r in res.items():
    r["delta_median_vs_real_db"] = (np.array(r["median_db"]) - rm).tolist() if rm is not None else None; r["delta_pooled_vs_real_db"] = (np.array(r["pooled_db"]) - rp).tolist() if rp is not None else None
out = OrderedDict(run_id=a.run, model=cfg["model"], ckpt=f"{d}/best.pt", best_epoch=ck.get("epoch"), created=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), script=os.path.abspath(__file__),
                  args=vars(a), config_mm=dict(sensors=sensors_cfg, token_mods=model.token_mods, n_tok=model.n_tok, pos_mode=pos_mode, pos_source=cfg.get("pos_source"), fuse_layers=len(model.fusion), causal_mask=cfg.get("causal_mask"), split=cfg["split"], K_hist=K, H_pred=H),
                  n_samples=meta_main["n"], n_windows=meta_main["n_windows"], n_val_windows_total=n_win_all, subset=bool(len(sel_win) < n_win_all), selected_windows=(sel_win.tolist() if len(sel_win) < n_win_all else "all"),
                  condition_defs={c: cond_def(c) for c in res}, conditions=res, diag=diag, checks=check, loader_selfcheck=selfcheck, partner_stats=partner_stats,
                  param_groups=model.param_groups(), timing=dict(total_sec=round(time.time() - t_start, 1), main=meta_main, peak_gpu_alloc_mb=round(peak_mb, 1), peak_gpu_reserved_mb=round(peak_res_mb, 1), device=str(dev)), loader_stats=va.stat_line())
json.dump(out, open(f"{d}/sensor_ablation.json", "w"), indent=1, ensure_ascii=False); print("saved", f"{d}/sensor_ablation.json", flush=True)
