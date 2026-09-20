"""G0-1: 경로 파라미터 -> 채널 H 합성 memmap (fp16, [N_frames, Nr=16, Nt=64, K, 2]) + 프레임 인덱스(궤적/프레임/속도/시나리오).
H_t[r,n,k] = sum_l a_l[r,n] exp(-j 2π f_k τ_l), f_k=(k-K/2)Δf. τ<0 더미 경로(a=0)는 자동 무시.
실행: python prep_h_memmap.py --setting Nt_1_64_Nr_1_16_fc_28GHz --K 64 --df 120e3
"""
import os, re, glob, json, argparse, time, numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--setting", default="Nt_1_64_Nr_1_16_fc_28GHz"); ap.add_argument("--K", type=int, default=64); ap.add_argument("--df", type=float, default=120e3)
ap.add_argument("--out", default="/mnt/ssd_7t_2/carla-wireless-dataset/mmw_reproduction/derived_cp")
a = ap.parse_args()
ROOT = "/mnt/ssd_7t_2/carla-wireless-dataset/mmw_reproduction/sunny"; CH = f"{ROOT}/channel_data/v2i/{a.setting}"; SD = f"{ROOT}/sensor_data"
FREQ = ((np.arange(a.K) - a.K / 2) * a.df).astype(np.float64)
SPEED_RE = re.compile(r"vehicle_speed:\s*\n\s*speed:\s*\n\s*x:\s*([-\d.eE+]+)\s*\n\s*y:\s*([-\d.eE+]+)\s*\n\s*z:\s*([-\d.eE+]+)")
POS_RE = re.compile(r"vehicle_pose:\s*\n\s*location:\s*\n\s*x:\s*([-\d.eE+]+)\s*\n\s*y:\s*([-\d.eE+]+)\s*\n\s*z:\s*([-\d.eE+]+)")
def fnum(p): return int(re.findall(r"\d+", os.path.basename(p))[0])
def sensor_dir(town, scen):
    c = sorted(p for p in glob.glob(f"{SD}/{town}/*_seed*") if os.path.basename(p).startswith(scen) or scen.startswith(re.sub(r"_seed\d+$", "", os.path.basename(p))))
    return c[0]
trajs = sorted(glob.glob(f"{CH}/Town*/*/cav_*"))
files_all, meta = [], []
for ti, t in enumerate(trajs):
    town, scen, cav = t.split("/")[-3:]
    fs = sorted(glob.glob(t + "/*_paths.npz"), key=fnum)
    for i, f in enumerate(fs): files_all.append(f); meta.append((ti, town, scen, cav, fnum(f), i, len(fs)))
N = len(files_all); Nr, Nt = 16, int(a.setting.split("_")[2])
tag = f"H_{a.setting}_K{a.K}"; mm_path = f"{a.out}/{tag}.f16"; idx_path = f"{a.out}/{tag}_index.json"
print(f"frames {N}, trajs {len(trajs)}, shape [{N},{Nr},{Nt},{a.K},2] fp16 = {N*Nr*Nt*a.K*2*2/1e9:.1f} GB -> {mm_path}", flush=True)
mm = np.memmap(mm_path, dtype=np.float16, mode="w+", shape=(N, Nr, Nt, a.K, 2))
speed = np.zeros(N, np.float32); pos = np.zeros((N, 3), np.float32); t0 = time.time(); sd_cache = {}
for i, f in enumerate(files_all):
    d = np.load(f); A = d["a"][0, 0, :, 0, :, :, 0]; tau = d["tau"].ravel()
    ph = np.exp(-2j * np.pi * np.outer(FREQ, tau)).astype(np.complex64)          # [K,P]
    H = np.einsum("rtp,kp->rtk", A, ph)                                            # [Nr,Nt,K] complex64
    mm[i, ..., 0] = H.real.astype(np.float16); mm[i, ..., 1] = H.imag.astype(np.float16)
    ti, town, scen, cav, fr, _, _ = meta[i]
    if (town, scen) not in sd_cache: sd_cache[(town, scen)] = sensor_dir(town, scen)
    y = open(f"{sd_cache[(town, scen)]}/{cav}/{fr:06d}.yaml").read()
    m = SPEED_RE.search(y); speed[i] = np.linalg.norm([float(m.group(1)), float(m.group(2)), float(m.group(3))]) if m else np.nan
    m = POS_RE.search(y); pos[i] = [float(m.group(1)), float(m.group(2)), float(m.group(3))] if m else np.nan
    if i % 2000 == 0: print(f"  {i}/{N} {time.time()-t0:.0f}s", flush=True)
mm.flush(); del mm
json.dump(dict(setting=a.setting, K=a.K, df=a.df, shape=[N, Nr, Nt, a.K, 2], dtype="float16", layout="[frame, rx, tx, subcarrier, re/im]",
               trajs=[dict(id=ti, town=t.split("/")[-3], scen=t.split("/")[-2], cav=t.split("/")[-1]) for ti, t in enumerate(trajs)],
               frame_traj=[m[0] for m in meta], frame_num=[m[4] for m in meta], frame_pos_in_traj=[m[5] for m in meta], traj_len=[m[6] for m in meta],
               speed=speed.tolist(), pos=pos.tolist()), open(idx_path, "w"))
# 검증: 무작위 5프레임 재합성 비교 (fp16 반올림 오차만 있어야 함)
mm = np.memmap(mm_path, dtype=np.float16, mode="r", shape=(N, Nr, Nt, a.K, 2)); rng = np.random.default_rng(0); errs = []
for i in rng.choice(N, 5, replace=False):
    d = np.load(files_all[i]); A = d["a"][0, 0, :, 0, :, :, 0]; tau = d["tau"].ravel(); H = np.einsum("rtp,kp->rtk", A, np.exp(-2j*np.pi*np.outer(FREQ, tau)))
    Hm = mm[i, ..., 0].astype(np.float32) + 1j * mm[i, ..., 1].astype(np.float32)
    errs.append(10*np.log10(np.sum(np.abs(H-Hm)**2)/np.sum(np.abs(H)**2)))
print("fp16 roundtrip NMSE dB (5 random frames):", np.round(errs, 1), "| speed nan:", int(np.isnan(speed).sum()), f"| done {time.time()-t0:.0f}s", flush=True)
