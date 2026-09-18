"""채널 예측 트레이너 (계획 EXPERIMENT_PLAN_FINAL_20260904.md).
예: python train_cp.py --model transformer --lr 1e-4 --D 512 --L 6 --split B1 --seed 42 --device cuda:0 --run_id c1_tf_lr1e-4_s42
공통: K=16→H=4, RX 행 표본, 창별 RMS 정규화, 손실 = 4지평 합산 NMSE, AdamW(wd 1e-4) warmup 1ep + cosine, clip 1.0, patience(기본 5), best-val ckpt, AMP 없음.
"""
import os, sys, json, time, math, argparse, random, csv, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_data import load_index, make_windows, WindowSet, nmse_per_sample, summarize, db
from cp_models import build_model, count_params

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--run_id", required=True)
ap.add_argument("--setting", default="Nt_1_64_Nr_1_16_fc_28GHz"); ap.add_argument("--Ksc", type=int, default=64)
ap.add_argument("--K_hist", type=int, default=16); ap.add_argument("--H_pred", type=int, default=4)
ap.add_argument("--split", default="B1"); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--patience", type=int, default=5)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--wd", type=float, default=1e-4); ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--clip", type=float, default=1.0); ap.add_argument("--warmup_epochs", type=float, default=1.0)
ap.add_argument("--D", type=int, default=512); ap.add_argument("--L", type=int, default=6); ap.add_argument("--heads", type=int, default=8); ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--no_pos_emb", action="store_true"); ap.add_argument("--residual", action="store_true", help="copy-last + delta 예측 (transformer/lstm)"); ap.add_argument("--c_hid", type=int, default=128); ap.add_argument("--c_enc", type=int, default=32)
ap.add_argument("--norm", default="rms", choices=["rms", "global", "none"]); ap.add_argument("--patch_h", type=int, default=4); ap.add_argument("--patch_w", type=int, default=16)
ap.add_argument("--lwm11_init", default="pretrained", choices=["pretrained", "scratch"]); ap.add_argument("--lwm11_freeze", default="none", choices=["none", "all", "first8"]); ap.add_argument("--in_scale", type=float, default=1.0); ap.add_argument("--t_layers", type=int, default=2)
ap.add_argument("--device", default="cuda:0"); ap.add_argument("--workers", type=int, default=0)
ap.add_argument("--max_train_windows", type=int, default=0, help="디버그: 학습 창 수 제한")
ap.add_argument("--eval_every", type=int, default=1)
ap.add_argument("--out_root", default=os.environ.get("CP_OUT", "/mnt/ssd_7t_2/carla-wireless-dataset/mmw_reproduction/outputs_cp"))
# --- 멀티모달(mm:) 전용 인자 (2026-09-07). mm: 이 아닌 모델에서는 어디에도 쓰이지 않는다 ---
ap.add_argument("--sensors", default="", help="mm: 모델 센서 목록(쉼표): cam,lidar,radar,pos"); ap.add_argument("--pos_source", default="predicted", choices=["predicted", "true", "gps"])
ap.add_argument("--pos_mode", default="token", choices=["token", "broadcast"]); ap.add_argument("--fuse_layers", type=int, default=3); ap.add_argument("--causal_mask", action="store_true")
ap.add_argument("--sensor_cache_root", default="", help="기본 derived_cp/sensor_cache"); ap.add_argument("--backbone_init", default="", help="mm: 백본 초기 가중치 ckpt(S1 chiron best.pt), 기본 scratch")
args = ap.parse_args()
IS_MM = args.model.startswith("mm:")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
set_seed(args.seed)
out = f"{args.out_root}/{args.run_id}"; os.makedirs(out, exist_ok=True)
log_f = open(f"{out}/train.log", "a")
def lprint(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
dev = torch.device(args.device)

# ---------- data ----------
t0 = time.time(); idx, mm = load_index(args.setting, args.Ksc)
F = torch.from_numpy(np.asarray(mm))  # RAM 상주 fp16 [N,Nr,Nt,Ksc,2]
N, Nr, Nt, Ksc, _ = F.shape
win, is_val = make_windows(idx, args.K_hist, args.H_pred, split=args.split)
if args.max_train_windows: win["train"] = win["train"][:args.max_train_windows]
tr = WindowSet(F, win["train"], args.K_hist, args.H_pred, Nr); va = WindowSet(F, win["val"], args.K_hist, args.H_pred, Nr)
if IS_MM:  # 센서 로더(창 단위 + 프레임 단위 LRU, val 은 train 과 캐시 공유). batch() 가 sensors dict 를 5번째로 돌려준다
    from cp_sensor_data import MMWindowSet
    tr = MMWindowSet(F, win["train"], args.K_hist, args.H_pred, Nr, idx, args.sensors, pos_source=args.pos_source, cache_root=args.sensor_cache_root or None, log=lprint)
    va = MMWindowSet(F, win["val"], args.K_hist, args.H_pred, Nr, idx, args.sensors, pos_source=args.pos_source, cache_root=args.sensor_cache_root or None, share=tr, log=lprint)
    lprint(f"[mm] sensors {tr.sensors} pos_source {args.pos_source} window_bytes {tr.window_bytes()} coverage {json.dumps(tr.coverage())}")
def get_batch(ws, ids):  # (X, Y, scale, meta, sensors|None)
    b = ws.batch(ids, dev); return b[0], b[1], b[2], b[3], (b[4] if IS_MM else None)
def fwd(X, Y, sens):
    if args.model == "convlstm_ae": return model(X, Y)
    return model(X, sens) if IS_MM else model(X)
speed = np.asarray(idx["speed"], np.float32)
lprint(f"[{args.run_id}] frames {N} Nr {Nr} Nt {Nt} Ksc {Ksc} | split {args.split}: train windows {len(win['train'])} (x{Nr} = {len(tr)}), val windows {len(win['val'])} (x{Nr} = {len(va)}) | load {time.time()-t0:.0f}s")
if args.norm == "global":
    g = float(torch.sqrt((F[::97].float() ** 2).sum(-1).mean())); lprint(f"global scale {g:.4e}")

# ---------- model ----------
kw = dict(D=args.D, L=args.L, heads=args.heads, dropout=args.dropout, pos_emb=not args.no_pos_emb, c_hid=args.c_hid, c_enc=args.c_enc, residual=args.residual, patch_h=args.patch_h, patch_w=args.patch_w, lwm11_init=args.lwm11_init, lwm11_freeze=args.lwm11_freeze, in_scale=args.in_scale, t_layers=args.t_layers)
if IS_MM: kw.update(sensors=args.sensors, pos_source=args.pos_source, pos_mode=args.pos_mode, fuse_layers=args.fuse_layers, causal_mask=args.causal_mask, backbone_init=args.backbone_init)
model = build_model(args.model, args.K_hist, args.H_pred, Nt, Ksc, **kw).to(dev)
n_params = count_params(model); lprint(f"model {args.model} params {n_params/1e6:.2f}M | args {json.dumps(vars(args))}")
if IS_MM: lprint(f"[mm] param_groups {json.dumps(model.param_groups())}")
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
steps_per_epoch = math.ceil(len(tr) / args.batch); total_steps = steps_per_epoch * args.epochs; warm = int(args.warmup_epochs * steps_per_epoch)
def lr_at(step):
    if step < warm: return args.lr * (step + 1) / max(1, warm)
    p = (step - warm) / max(1, total_steps - warm); return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
json.dump(dict(vars(args), n_params=n_params, Nt=Nt, Ksc=Ksc, Nr=Nr, n_train=len(tr), n_val=len(va), val_scenes=[t["scen"] for t, v in zip(idx["trajs"], is_val) if v]), open(f"{out}/config.json", "w"), indent=1)

def apply_norm(X, Y, scale):
    if args.norm == "rms": return X, Y  # WindowSet.batch 가 이미 RMS 로 나눔
    X = X * scale[:, None, None, None, None]; Y = Y * scale[:, None, None, None, None]  # 되돌림
    if args.norm == "global": return X / g, Y / g
    return X, Y

def loss_fn(pred, Y):
    return (((pred - Y) ** 2).sum(dim=(2, 3, 4)).sum(1) / (Y ** 2).sum(dim=(2, 3, 4)).sum(1).clamp_min(1e-12)).mean()

@torch.no_grad()
def evaluate(ws, tag):
    model.eval(); raws, aligns, rhos, metas = [], [], [], []; err_sum = np.zeros(args.H_pred); pw_sum = np.zeros(args.H_pred); cl_err = np.zeros(args.H_pred)
    ids = list(range(len(ws)))
    for b in range(0, len(ids), args.batch):
        X, Y, sc, meta, sens = get_batch(ws, ids[b:b + args.batch]); X, Y = apply_norm(X, Y, sc)
        pred = fwd(X, None, sens)
        raw, align, rho = nmse_per_sample(pred, Y)
        raws.append(raw.cpu().numpy()); aligns.append(align.cpu().numpy()); rhos.append(rho.cpu().numpy()); metas += meta
        err_sum += ((pred - Y) ** 2).sum(dim=(0, 2, 3, 4)).cpu().numpy(); pw_sum += (Y ** 2).sum(dim=(0, 2, 3, 4)).cpu().numpy()
        cl = X[:, -1:].expand(-1, args.H_pred, -1, -1, -1); cl_err += ((cl - Y) ** 2).sum(dim=(0, 2, 3, 4)).cpu().numpy()
    raw = np.concatenate(raws); align = np.concatenate(aligns); rho = np.concatenate(rhos)
    sp_last = np.array([speed[s + args.K_hist - 1] for _, s in metas])
    summ = summarize(raw, align, rho, err_sum, pw_sum, metas, idx, sp_last); summ["copy_last_pooled_db"] = db(cl_err / pw_sum).tolist()
    model.train(); return summ

# ---------- train ----------
best = float("inf"); best_ep = -1; bad = 0; step = 0
csv_f = open(f"{out}/metrics.csv", "a", newline=""); cw = csv.writer(csv_f)
if os.path.getsize(f"{out}/metrics.csv") == 0: cw.writerow(["epoch", "lr", "train_loss", "val_pooled_db_w1", "val_pooled_db_w2", "val_pooled_db_w3", "val_pooled_db_w4", "val_median_db_w1", "val_median_db_w4", "val_align_median_db_w1", "val_rho_w1", "copy_last_pooled_db_w1", "epoch_sec", "gpu_mem_mb"])
model.train()
for ep in range(1, args.epochs + 1):
    te = time.time(); perm = torch.randperm(len(tr)).tolist(); tot = 0.0; nb = 0
    for b in range(0, len(perm), args.batch):
        for gph in opt.param_groups: gph["lr"] = lr_at(step)
        X, Y, sc, _, sens = get_batch(tr, perm[b:b + args.batch]); X, Y = apply_norm(X, Y, sc)
        pred = fwd(X, Y, sens)
        loss = loss_fn(pred, Y); opt.zero_grad(set_to_none=True); loss.backward()
        if args.clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt.step(); tot += loss.item(); nb += 1; step += 1
        if nb % 500 == 0: lprint(f"  ep{ep} step {nb}/{steps_per_epoch} loss {tot/nb:.4f} lr {lr_at(step):.2e} {time.time()-te:.0f}s")
    v = evaluate(va, "val"); vsel = float(np.mean(v["pooled_db"]))  # 조기종료 기준: 4지평 평균 pooled dB
    row = dict(ep=ep, loss=round(tot / max(1, nb), 5), val_pooled_db=[round(x, 2) for x in v["pooled_db"]], val_median_db=[round(x, 2) for x in v["median_db"]], val_align_db=[round(x, 2) for x in v["align_median_db"]],
               rho=[round(x, 3) for x in v["rho_median"]], copy_last_db=[round(x, 2) for x in v["copy_last_pooled_db"]], lr=lr_at(step), sec=round(time.time() - te, 1))
    lprint(json.dumps(row))
    cw.writerow([ep, lr_at(step), tot / max(1, nb)] + v["pooled_db"] + [v["median_db"][0], v["median_db"][-1], v["align_median_db"][0], v["rho_median"][0], v["copy_last_pooled_db"][0], time.time() - te, torch.cuda.max_memory_allocated() / 2**20]); csv_f.flush()
    if vsel < best - 1e-4:
        best, best_ep, bad = vsel, ep, 0; torch.save({"model": model.state_dict(), "epoch": ep, "val": v, "args": vars(args)}, f"{out}/best.pt")
        json.dump(dict(run_id=args.run_id, model=args.model, n_params=n_params, best_epoch=ep, val=v, args=vars(args)), open(f"{out}/result.json", "w"), indent=1)
    else:
        bad += 1
        if args.patience > 0 and bad >= args.patience: lprint(f"early stopping at ep {ep} (best ep {best_ep}, {best:.2f} dB)"); break
lprint(f"DONE best_ep {best_ep} val_mean_pooled_db {best:.2f}")
open(f"{out}/DONE", "w").write(f"{best_ep} {best:.3f}\n")
