"""채널 예측 공용 데이터 모듈: memmap 로드, 분할(B1 시나리오 holdout), 창 인덱스, RX 행 표본, RMS 정규화, 지표.
표본: X [K, Nt, Ksc, 2] (과거 K 프레임, RX 행 1개) -> Y [H, Nt, Ksc, 2].
"""
import os, json, numpy as np, torch

DERIVED = os.environ.get("CP_DERIVED", "/mnt/ssd_7t_2/carla-wireless-dataset/mmw_reproduction/derived_cp")  # 로컬 실행: CP_DERIVED 로 덮어씀
VAL_SCENES_B1 = ("Town05_ringroad", "Town03_gastation", "Town10_crossroad")
SPEED_BINS = [(0.0, 1.0), (1.0, 4.0), (4.0, 6.0), (6.0, 9.0)]


def load_index(setting="Nt_1_64_Nr_1_16_fc_28GHz", K=64):
    tag = f"H_{setting}_K{K}"
    idx = json.load(open(f"{DERIVED}/{tag}_index.json"))
    shape = tuple(idx["shape"])
    mm = np.memmap(f"{DERIVED}/{tag}.f16", dtype=np.float16, mode="r", shape=shape)
    return idx, mm


def make_windows(idx, K_hist, H_pred, split="B1", val_scenes=VAL_SCENES_B1, train_stride=1, val_stride=None):
    """(traj별 연속 프레임) -> 창 시작 전역 인덱스. 반환: dict(train=[...], val=[...]) 각 원소 = (start_global, traj_id)."""
    val_stride = val_stride or H_pred
    ft = np.asarray(idx["frame_traj"]); fp = np.asarray(idx["frame_pos_in_traj"]); tl = np.asarray(idx["traj_len"])
    trajs = idx["trajs"]
    is_val = np.array([t["scen"] in val_scenes for t in trajs])
    if split == "A1":  # 궤적 단위(기존 seed0 val 5개)
        VAL = {("Town07_grainsilos", "cav_3"), ("Town05_CBDcrossroad", "cav_3"), ("Town05_CBDcrossroad", "cav_1"), ("Town03_crossroad", "cav_3"), ("Town10_Hroad", "cav_3")}
        is_val = np.array([(t["scen"], t["cav"]) in VAL for t in trajs])
    win = {"train": [], "val": []}
    L = K_hist + H_pred
    starts = np.where(fp == 0)[0]  # 각 궤적의 첫 전역 인덱스
    if split.startswith("T1"):  # 시간(프레임) 기준 분할 (2026-09-14, EXPERIMENT_PLAN_FRAMESPLIT_20260914.md §2): 궤적마다 앞 80% train / 뒤 20% val, 사이 G 프레임 비움
        ratio, G = 0.8, (int(split[3:]) if split.startswith("T1g") else L)  # "T1" -> G = 창 길이(프레임 공유 0), "T1g100" -> G = 100
        for ti, s0 in enumerate(starts):
            n = int(tl[s0]); T = int(np.floor(ratio * n))
            for s in range(0, (T - G) - L + 1, train_stride): win["train"].append((int(s0 + s), ti))   # 창 [s, s+L) ⊂ [0, T-G)
            for s in range(T, n - L + 1, val_stride): win["val"].append((int(s0 + s), ti))            # 창 [s, s+L) ⊂ [T, n)
        return win, np.ones(len(trajs), dtype=bool)  # 모든 궤적(장면 16개)이 val 에 포함
    for ti, s0 in enumerate(starts):
        n = int(tl[s0]); key = "val" if is_val[ti] else "train"; stride = val_stride if is_val[ti] else train_stride
        for s in range(0, n - L + 1, stride):
            win[key].append((int(s0 + s), ti))
    return win, is_val


class WindowSet:
    """창 × RX 행 표본 집합. 프레임 데이터는 RAM 상주 fp16 torch 텐서."""
    def __init__(self, frames_fp16: torch.Tensor, windows, K_hist, H_pred, n_rx=16):
        self.F = frames_fp16  # [N, Nr, Nt, Ksc, 2] float16 (CPU)
        self.windows = windows; self.K = K_hist; self.H = H_pred; self.n_rx = n_rx
        self.items = [(w, r) for w in range(len(windows)) for r in range(n_rx)]

    def __len__(self): return len(self.items)

    def batch(self, item_ids, device):
        """item_ids -> X [B,K,Nt,Ksc,2], Y [B,H,Nt,Ksc,2] float32 on device (RMS 정규화 적용), scale [B], meta(traj, start)."""
        xs, ys, meta = [], [], []
        for i in item_ids:
            w, r = self.items[i]; s, ti = self.windows[w]
            blk = self.F[s:s + self.K + self.H, r]  # [K+H, Nt, Ksc, 2]
            xs.append(blk[:self.K]); ys.append(blk[self.K:]); meta.append((ti, s))
        X = torch.stack(xs).to(device, non_blocking=True).float(); Y = torch.stack(ys).to(device, non_blocking=True).float()
        scale = torch.sqrt((X ** 2).sum(-1).mean(dim=(1, 2, 3))).clamp_min(1e-12)  # per-sample RMS over K,Nt,Ksc of |h|
        X = X / scale[:, None, None, None, None]; Y = Y / scale[:, None, None, None, None]
        return X, Y, scale, meta


def nmse_per_sample(pred, tgt):
    """pred,tgt [B,H,Nt,Ksc,2] -> raw NMSE [B,H], align NMSE(=1-rho^2) [B,H], rho [B,H]."""
    err = ((pred - tgt) ** 2).sum(dim=(2, 3, 4)); pw = (tgt ** 2).sum(dim=(2, 3, 4)).clamp_min(1e-12)
    raw = err / pw
    pc = torch.complex(pred[..., 0], pred[..., 1]); tc = torch.complex(tgt[..., 0], tgt[..., 1])
    inner = (pc.conj() * tc).sum(dim=(2, 3)).abs()
    rho = inner / (torch.sqrt((pc.abs() ** 2).sum(dim=(2, 3))) * torch.sqrt((tc.abs() ** 2).sum(dim=(2, 3)))).clamp_min(1e-12)
    align = (1 - rho ** 2).clamp_min(1e-12)
    return raw, align, rho


def db(x): return 10 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1e-12))


def summarize(raw, align, rho, err_sum, pw_sum, metas, idx, speeds_last):
    """per-window 지표 배열 -> dict(pooled dB, median dB, per-scene, per-speed)."""
    trajs = idx["trajs"]; H = raw.shape[1]
    scen = np.array([trajs[t]["scen"] for t, _ in metas])
    out = {"pooled_db": db(err_sum / pw_sum).tolist(), "median_db": np.median(db(raw), axis=0).tolist(), "q25_db": np.percentile(db(raw), 25, axis=0).tolist(),
           "q75_db": np.percentile(db(raw), 75, axis=0).tolist(), "align_median_db": np.median(db(align), axis=0).tolist(), "rho_median": np.median(rho, axis=0).tolist(), "n": int(raw.shape[0])}
    out["per_scene"] = {s: {"n": int((scen == s).sum()), "median_db": np.median(db(raw[scen == s]), axis=0).tolist()} for s in sorted(set(scen))}
    out["per_speed"] = {}
    for lo, hi in SPEED_BINS:
        m = (speeds_last >= lo) & (speeds_last < hi)
        if m.any(): out["per_speed"][f"{lo}-{hi}"] = {"n": int(m.sum()), "median_db": np.median(db(raw[m]), axis=0).tolist()}
    return out
