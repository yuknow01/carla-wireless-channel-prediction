"""완료 run 의 best.pt 로 val 재평가 + 누수/의존성 검사.
조건: real(정상) / hist_shuffle(이력을 다른 창의 것으로 교체, 타깃 유지 → 0 dB 근처여야 정상) / last_zero(마지막 프레임만 0) / hist_only_last(마지막 프레임만 남기고 나머지 0).
실행: python eval_sanity.py --run s1_lwm_lr3e-4 --device cuda:1
"""
import os, sys, json, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_data import load_index, make_windows, WindowSet, nmse_per_sample, db
from cp_models import build_model
ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:1"); ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--out_root", default=os.environ.get("CP_OUT", "/mnt/ssd_7t_2/carla-wireless-dataset/mmw_reproduction/outputs_cp")); a = ap.parse_args()
d = f"{a.out_root}/{a.run}"; cfg = json.load(open(f"{d}/config.json")); ck = torch.load(f"{d}/best.pt", map_location="cpu", weights_only=False)
dev = torch.device(a.device); idx, mm = load_index(cfg["setting"], cfg["Ksc"]); F = torch.from_numpy(np.asarray(mm)); Nr = F.shape[1]
win, _ = make_windows(idx, cfg["K_hist"], cfg["H_pred"], split=cfg["split"]); va = WindowSet(F, win["val"], cfg["K_hist"], cfg["H_pred"], Nr)
kw = dict(D=cfg["D"], L=cfg["L"], heads=cfg["heads"], dropout=cfg["dropout"], pos_emb=not cfg["no_pos_emb"], c_hid=cfg["c_hid"], c_enc=cfg["c_enc"], residual=cfg.get("residual", False), patch_h=cfg.get("patch_h", 4), patch_w=cfg.get("patch_w", 16), lwm11_init=cfg.get("lwm11_init", "pretrained"), lwm11_freeze=cfg.get("lwm11_freeze", "none"), in_scale=cfg.get("in_scale", 1.0), t_layers=cfg.get("t_layers", 2))
model = build_model(cfg["model"], cfg["K_hist"], cfg["H_pred"], F.shape[2], F.shape[3], **kw).to(dev); model.load_state_dict(ck["model"]); model.eval()
K = cfg["K_hist"]; res = {}
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    for cond in ("real", "hist_shuffle", "last_zero", "hist_only_last"):
        raws = []; ids = list(range(len(va)))
        for b in range(0, len(ids), a.batch):
            X, Y, _, _ = va.batch(ids[b:b + a.batch], dev)
            if cond == "hist_shuffle": X = X[torch.randperm(X.shape[0], generator=g).to(dev)]
            elif cond == "last_zero": X = X.clone(); X[:, -1] = 0
            elif cond == "hist_only_last": X = X.clone(); X[:, :-1] = 0
            pred = model(X); raw, _, _ = nmse_per_sample(pred, Y); raws.append(raw.cpu().numpy())
        raw = np.concatenate(raws); res[cond] = dict(median_db=np.median(db(raw), 0).tolist(), pooled_like_mean_db=db(raw.mean(0)).tolist())
        print(f"{cond:15s} median dB {np.round(res[cond]['median_db'], 2)}", flush=True)
json.dump(res, open(f"{d}/sanity.json", "w"), indent=1); print("saved", f"{d}/sanity.json")
