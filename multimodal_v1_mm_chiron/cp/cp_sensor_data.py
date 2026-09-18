"""채널 예측(K=16→H=4, Δ=10 ms, RX 행 표본) 멀티모달용 RSU 센서 로더·캐시 (2026-09-07, 신규 파일).

설계 근거: MULTIMODAL_ARCHITECTURES.md §5.0(센서·채널 10 ms 1:1 → 프레임 번호로 직접 정렬, 시나리오당 RSU 1벌을 CAV 3~4대가 공유),
           §5.1 안 (a)(캐시된 인코더 출력 + 프레임당 수십 KB 풀링 요약), §5.2 안 (b)(CAV 위치 → RSU 로컬 특징 10차원).
채널 index(`derived_cp/*_index.json`: trajs[ti]={town,scen,cav}, frame_num[i], frame_traj[i]) 와 RSU 캐시(`derived/feat_patch_rsu224/<town>/<scen>/rsu.npy`
+ `rsu_frames.npy`, 프레임 번호가 채널 6자리 프레임 번호와 동일 — dataset_faithful.py:81-93·precompute_patch_rsu.py 와 같은 규약)를 (town, scen, frame_num) 으로 잇는다.

모달·캐시:
  cam   : derived/feat_patch_rsu224/<town>/<scen>/rsu.npy  [Nf,1,196,768] fp16 memmap (기존, [B] 재현용 ViT-B/16 패치)  → 창당 [16,196,768] fp16 (모델 입구에서 float)
  lidar : derived_cp/sensor_cache/lidar_pool8/<town>/<scen>/rsu.npy [Nf,64,384] fp16 (신규, --build_lidar_pool: feat_pp_rsu [384,100,176] → AdaptiveAvgPool2d((8,8)) → 셀 64 × 채널 384)
  radar : derived_cp/sensor_cache/radar_raster64/<town>/<scen>/rsu.npy [Nf,2,64,64] fp16 (신규, --build_radar: rsu_1/<frame>.json 검출 리스트 → 거리 64 bin × 방위 64 bin, ch0=점유 카운트, ch1=평균 velocity)
  pos   : derived_cp/sensor_cache/pos/pos_features.npz (신규, --build_pos: CAV yaml predicted_ego_pos/true_ego_pos/GPS + vehicle_speed → RSU 로컬 10차원 특징, 전역 프레임 순서 [53800,10] × 3 소스)
창 단위 LRU + 프레임 단위 LRU(기본 무제한 ≈ 17,200 RSU 프레임 × 0.37 MB ≈ 6.3 GB RAM)로 디스크 읽기를 줄인다(창의 RX 16행·stride-1 이웃 창이 프레임을 공유).
실행(캐시 생성):
  CUDA_VISIBLE_DEVICES=0 python cp_sensor_data.py --build_lidar_pool --device cuda:0     # 약 217 GB 순차 읽기
  python cp_sensor_data.py --build_radar
  python cp_sensor_data.py --build_pos
  python cp_sensor_data.py --coverage                                                     # 커버리지 표 + 위치 변환 vs derived/aod 자체 점검
"""
import os, sys, re, json, glob, time, math, argparse
from collections import OrderedDict
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cp_data import DERIVED, load_index, WindowSet

ROOT = "/mnt/ssd_7t_2/carla-wireless-dataset/mmw_reproduction"
SENSOR_ROOT = f"{ROOT}/sunny/sensor_data"
PATCH_RSU224 = f"{ROOT}/derived/feat_patch_rsu224"      # [Nf,1,196,768] fp16 + rsu_frames.npy (precompute_patch_rsu.py --resize 224, [B] 재현 캐시)
PP_RSU = f"{ROOT}/derived/feat_pp_rsu"                   # [Nf,384,100,176] fp16 + rsu_frames.npy + meta.json (precompute_pointpillar_rsu.py, every=1)
AOD_DIR = f"{ROOT}/derived/aod/Nt_1_64_Nr_1_16_fc_28GHz"  # 지배 경로 AoD phi_t [rad] (gen_aod.py) — 위치 변환 자체 점검용
DEFAULT_CACHE_ROOT = f"{DERIVED}/sensor_cache"
ALL_SENSORS = ("cam", "lidar", "radar", "pos")
LIDAR_POOL_HW = (8, 8); LIDAR_CELLS = LIDAR_POOL_HW[0] * LIDAR_POOL_HW[1]; LIDAR_CH = 384
RADAR_R_BINS, RADAR_AZ_BINS, RADAR_R_MAX = 64, 64, 120.0   # 거리 0~120 m(지시), 방위 = config.yaml radar horizontal_fov(110°) 대칭
POS_DIM = 10; POS_SOURCES = ("predicted", "true", "gps")
CAM_SHAPE, LIDAR_SHAPE, RADAR_SHAPE = (196, 768), (LIDAR_CELLS, LIDAR_CH), (2, RADAR_R_BINS, RADAR_AZ_BINS)


def frame_num(p): return int(re.findall(r"\d+", os.path.basename(p))[0])


def sensor_dir(town, scen):
    """채널 시나리오명 → 센서 폴더(시드 접미사 포함). 16/16 시나리오가 정확히 1개씩 대응(2026-09-07 실측)."""
    c = sorted(p for p in glob.glob(f"{SENSOR_ROOT}/{town}/{scen}*") if os.path.isdir(p))
    assert len(c) == 1, (town, scen, c)
    return c[0]


def scen_list(idx): return sorted({(t["town"], t["scen"]) for t in idx["trajs"]})


# ----------------------------------------------------------------------------------------------------------------
# 위치 변환: CARLA 월드 → RSU 배열(Sionna) 로컬.  REPRO_V2_PREREQUISITES.md §5.3 규약(CARLA→Sionna = y 부호반전, Tx 배열 방위 = −(RSU LiDAR yaw)).
# 본 파일은 그 규약을 그대로 옮긴 것이며 --coverage 가 derived/aod 의 phi_t 와 대조해 자체 점검한다(문서 §4 참조).
# ----------------------------------------------------------------------------------------------------------------
def rsu_local(p_world, rsu_xyz, lidar_yaw_deg):
    """p_world [N,3] (CARLA) → RSU 로컬 [N,3]: d = p − p_rsu; (x, −y, z); R_z(+yaw_lidar) 회전(= R_z(−az), az = −yaw_lidar)."""
    d = np.asarray(p_world, np.float64) - np.asarray(rsu_xyz, np.float64)[None]
    xs, ys, zs = d[:, 0], -d[:, 1], d[:, 2]
    a = math.radians(lidar_yaw_deg); c, s = math.cos(a), math.sin(a)
    return np.stack([c * xs - s * ys, s * xs + c * ys, zs], 1)


def rsu_local_vec(v_world, lidar_yaw_deg):
    return rsu_local(v_world, (0.0, 0.0, 0.0), lidar_yaw_deg)


def pos_features(p_local, v_local):
    """[dx,dy,dz,r,sinφ,cosφ,vx,vy,vz,|v|] (m, m/s; φ = atan2(y', x') = 빔각 기준)."""
    r = np.linalg.norm(p_local, axis=1); phi = np.arctan2(p_local[:, 1], p_local[:, 0]); sp = np.linalg.norm(v_local, axis=1)
    return np.concatenate([p_local, r[:, None], np.sin(phi)[:, None], np.cos(phi)[:, None], v_local, sp[:, None]], 1).astype(np.float32)


_XYZ = r"{name}:\s*\n\s*{sub}:\s*\n\s*x:\s*([-\d.eE+]+)\s*\n\s*y:\s*([-\d.eE+]+)\s*\n\s*z:\s*([-\d.eE+]+)"
_RE = {k: re.compile(_XYZ.format(name=n, sub=s)) for k, (n, s) in
       dict(true=("vehicle_pose", "location"), pred=("predicted_ego_pos", "location"), gps=("GPS", "location"), vel=("vehicle_speed", "speed")).items()}


def parse_cav_yaml(text):
    out = {}
    for k, rx in _RE.items():
        m = rx.search(text); out[k] = [float(m.group(1)), float(m.group(2)), float(m.group(3))] if m else [np.nan] * 3
    return out


def read_rsu_pose(sdir):
    """rsu_1 첫 yaml → dict(rsu_xyz, lidar_xyz, lidar_yaw, radar_yaw). yaml 은 프레임 간 동일(STEP3 §3-0 `rsu_pose_unique_over_frames: 1`)."""
    import yaml
    f = sorted(glob.glob(f"{sdir}/rsu_1/[0-9]*.yaml"))[0]; y = yaml.safe_load(open(f))["sensors"]
    L = lambda b: [y[b]["location"][k] for k in "xyz"]
    return dict(rsu_xyz=L("rsu_pose"), lidar_xyz=L("lidar_pose"), lidar_yaw=float(y["lidar_pose"]["rotation"]["yaw"]),
                radar_yaw=float(y["radar_pose"]["rotation"]["yaw"]), rsu_yaw=float(y["rsu_pose"]["rotation"]["yaw"]), file=f)


def read_radar_fov(sdir):
    """config.yaml 의 rsu radar horizontal_fov(도). 없으면 None (호출측이 ±60° 가정 표기)."""
    try:
        import yaml
        cfg = yaml.safe_load(open(f"{sdir}/config.yaml"))
        def find(o):
            if isinstance(o, dict):
                if o.get("blueprint") == "sensor.other.radar": return o
                for v in o.values():
                    r = find(v)
                    if r is not None: return r
            elif isinstance(o, list):
                for v in o:
                    r = find(v)
                    if r is not None: return r
            return None
        r = find(cfg); a = r["attributes"]
        return dict(horizontal_fov=float(a["horizontal_fov"]), vertical_fov=float(a.get("vertical_fov", "nan")), range=float(a.get("range", "nan")))
    except Exception as e:
        return None


# ----------------------------------------------------------------------------------------------------------------
# 캐시 생성
# ----------------------------------------------------------------------------------------------------------------
def _write_scen(out_dir, arr, frames, meta):
    os.makedirs(out_dir, exist_ok=True)
    np.save(f"{out_dir}/rsu.npy", arr); np.save(f"{out_dir}/frames.npy", np.asarray(frames, np.int32))
    json.dump(meta, open(f"{out_dir}/meta.json", "w"), indent=1)


def build_lidar_pool(idx, cache_root, device="cuda:0", batch=32, log=print):
    """feat_pp_rsu [Nf,384,100,176] fp16 → AdaptiveAvgPool2d((8,8)) → [Nf,64,384] fp16 (셀 = row*8+col, row=y(배열 프레임), col=x)."""
    dev = torch.device(device); pool = torch.nn.AdaptiveAvgPool2d(LIDAR_POOL_HW); t_all = time.time(); tot_bytes = 0; tot_read = 0
    for town, scen in scen_list(idx):
        od = f"{cache_root}/lidar_pool8/{town}/{scen}"
        if os.path.exists(f"{od}/rsu.npy"): log(f"[lidar_pool] skip {town}/{scen}"); continue
        src = f"{PP_RSU}/{town}/{scen}"; arr = np.load(f"{src}/rsu.npy", mmap_mode="r"); frames = np.load(f"{src}/rsu_frames.npy"); meta_src = json.load(open(f"{src}/meta.json"))
        Nf, C, Hh, Ww = arr.shape; fb = C * Hh * Ww * arr.dtype.itemsize; out = np.zeros((Nf, LIDAR_CELLS, C), np.float16); t0 = time.time()
        buf = np.empty((batch, C, Hh, Ww), np.float16)
        with open(arr.filename, "rb", buffering=0) as fh:
            for b0 in range(0, Nf, batch):
                n = min(batch, Nf - b0); fh.seek(arr.offset + b0 * fb); fh.readinto(memoryview(buf[:n])); tot_read += n * fb
                with torch.no_grad():
                    p = pool(torch.from_numpy(buf[:n]).to(dev).float())                      # [n,384,8,8]
                    out[b0:b0 + n] = p.permute(0, 2, 3, 1).reshape(n, LIDAR_CELLS, C).cpu().numpy().astype(np.float16)
        meta = dict(source=f"{src}/rsu.npy", source_shape=[Nf, C, Hh, Ww], pool="AdaptiveAvgPool2d((8,8)) on [384,100,176] (100/8·176/8 비정수 → torch adaptive bin)",
                    layout="[Nf, cell=row*8+col (row=y 배열프레임·CARLA y 반전, col=x), 384ch] fp16", source_meta={k: meta_src[k] for k in ("every", "z_shift", "frame_axes", "n_frames")},
                    lidar_range=meta_src["pointpillars"]["lidar_range"], created=time.strftime("%Y-%m-%d %H:%M"), sec=round(time.time() - t0, 1))
        _write_scen(od, out, frames, meta); tot_bytes += out.nbytes
        log(f"[lidar_pool] {town}/{scen} {Nf}f {out.nbytes/2**20:.1f} MB {time.time()-t0:.0f}s (read {Nf*fb/2**30:.1f} GB)")
    log(f"[lidar_pool] done {time.time()-t_all:.0f}s written {tot_bytes/2**20:.0f} MB read {tot_read/2**30:.1f} GB")


def radar_raster(dets, fov_deg, r_max=RADAR_R_MAX, nr=RADAR_R_BINS, na=RADAR_AZ_BINS):
    """검출 리스트 [{velocity, azimuth(rad), altitude, depth(m)}] → [2,nr,na] float32 (ch0 점유 카운트, ch1 평균 velocity m/s; 범위 밖 검출은 버림). 반환 (raster, n_dropped)."""
    out = np.zeros((2, nr, na), np.float32)
    if not dets: return out, 0
    a = np.array([[d["velocity"], d["azimuth"], d["depth"]] for d in dets], np.float64)
    half = math.radians(fov_deg) / 2
    ri = np.floor(a[:, 2] / r_max * nr).astype(int); ai = np.floor((a[:, 1] + half) / (2 * half) * na).astype(int)
    ok = (ri >= 0) & (ri < nr) & (ai >= 0) & (ai < na); ri, ai, v = ri[ok], ai[ok], a[ok, 0]
    flat = ri * na + ai; cnt = np.bincount(flat, minlength=nr * na).astype(np.float32); sv = np.bincount(flat, weights=v, minlength=nr * na).astype(np.float32)
    out[0] = cnt.reshape(nr, na); out[1] = np.where(cnt > 0, sv / np.maximum(cnt, 1), 0).reshape(nr, na)
    return out, int((~ok).sum())


def build_radar(idx, cache_root, log=print):
    t_all = time.time(); tot_bytes = 0
    for town, scen in scen_list(idx):
        od = f"{cache_root}/radar_raster64/{town}/{scen}"
        if os.path.exists(f"{od}/rsu.npy"): log(f"[radar] skip {town}/{scen}"); continue
        sdir = sensor_dir(town, scen); fov = read_radar_fov(sdir); fov_deg = fov["horizontal_fov"] if fov else 120.0
        files = sorted(glob.glob(f"{sdir}/rsu_1/[0-9]*.json"), key=frame_num); frames = [frame_num(f) for f in files]
        out = np.zeros((len(files), 2, RADAR_R_BINS, RADAR_AZ_BINS), np.float16); t0 = time.time(); n_det = 0; n_drop = 0; max_cnt = 0
        for i, f in enumerate(files):
            dets = json.load(open(f)); r, nd = radar_raster(dets, fov_deg); out[i] = r.astype(np.float16); n_det += len(dets); n_drop += nd; max_cnt = max(max_cnt, float(r[0].max()))
        meta = dict(source=f"{sdir}/rsu_1/<frame>.json", fov_deg=fov_deg, fov_source=("config.yaml radar horizontal_fov" if fov else "확인 불가 → ±60° 가정"), config_radar=fov,
                    r_max_m=RADAR_R_MAX, r_bins=RADAR_R_BINS, az_bins=RADAR_AZ_BINS, layout="[Nf, ch(0=count,1=mean velocity m/s), range bin(0..120 m), azimuth bin(−fov/2..+fov/2, CARLA 부호 그대로)] fp16",
                    n_frames=len(files), mean_detections=n_det / max(1, len(files)), dropped_frac=n_drop / max(1, n_det), max_cell_count=max_cnt, created=time.strftime("%Y-%m-%d %H:%M"), sec=round(time.time() - t0, 1))
        _write_scen(od, out, frames, meta); tot_bytes += out.nbytes
        log(f"[radar] {town}/{scen} {len(files)}f fov {fov_deg}° mean det {n_det/max(1,len(files)):.0f} dropped {n_drop/max(1,n_det):.4f} max cell {max_cnt:.0f} {out.nbytes/2**20:.1f} MB {time.time()-t0:.0f}s")
    log(f"[radar] done {time.time()-t_all:.0f}s written {tot_bytes/2**20:.0f} MB")


def build_pos(idx, cache_root, log=print):
    """전역 프레임 순서(index 와 동일)로 CAV yaml 을 읽어 소스별 10차원 특징을 만든다. 출력 pos/pos_features.npz."""
    od = f"{cache_root}/pos"; os.makedirs(od, exist_ok=True); outp = f"{od}/pos_features.npz"
    if os.path.exists(outp): log(f"[pos] exists {outp}"); return outp
    ft = np.asarray(idx["frame_traj"]); fn = np.asarray(idx["frame_num"]); N = len(ft); trajs = idx["trajs"]
    raw = {k: np.full((N, 3), np.nan, np.float32) for k in ("true", "pred", "gps", "vel")}; feats = {k: np.zeros((N, POS_DIM), np.float32) for k in POS_SOURCES}
    rsu_info = {}; missing = []; t0 = time.time(); phi_true = np.full(N, np.nan, np.float32)
    for ti, t in enumerate(trajs):
        sdir = sensor_dir(t["town"], t["scen"]); key = f"{t['town']}/{t['scen']}"
        if key not in rsu_info: rsu_info[key] = read_rsu_pose(sdir)
        ii = np.where(ft == ti)[0]
        for i in ii:
            f = f"{sdir}/{t['cav']}/{int(fn[i]):06d}.yaml"
            if not os.path.exists(f): missing.append(f); continue
            d = parse_cav_yaml(open(f).read())
            for k in raw: raw[k][i] = d[k]
        rp = rsu_info[key]; v_loc = rsu_local_vec(raw["vel"][ii], rp["lidar_yaw"])
        for src, k in (("true", "true"), ("predicted", "pred"), ("gps", "gps")):
            p_loc = rsu_local(raw[k][ii], rp["lidar_xyz"], rp["lidar_yaw"]); feats[src][ii] = pos_features(p_loc, v_loc)
            if src == "true": phi_true[ii] = np.arctan2(p_loc[:, 1], p_loc[:, 0])
        if ti % 10 == 0: log(f"[pos] traj {ti}/{len(trajs)} {time.time()-t0:.0f}s")
    np.savez(outp, frames=fn.astype(np.int32), frame_traj=ft.astype(np.int32), **{f"raw_{k}": v for k, v in raw.items()}, **{f"feat_{k}": v for k, v in feats.items()}, phi_true=phi_true,
             rsu_info=json.dumps(rsu_info), missing=np.array(missing), feature_names=np.array(["dx", "dy", "dz", "r", "sin_phi", "cos_phi", "vx", "vy", "vz", "speed"]),
             convention="d=p−p_lidar(CARLA); (x,−y,z); R_z(+yaw_lidar_CARLA) — REPRO_V2_PREREQUISITES §5.3 규약, 원점 = lidar_pose(z 4 m) [가정]; 속도 = vehicle_speed(시뮬레이터 참값, 소스 무관) [가정]")
    log(f"[pos] done N {N} missing yaml {len(missing)} nan rows true {int(np.isnan(raw['true'][:,0]).sum())} pred {int(np.isnan(raw['pred'][:,0]).sum())} gps {int(np.isnan(raw['gps'][:,0]).sum())} {time.time()-t0:.0f}s → {outp} ({os.path.getsize(outp)/2**20:.1f} MB)")
    return outp


# ----------------------------------------------------------------------------------------------------------------
# 로더
# ----------------------------------------------------------------------------------------------------------------
class RSUSensorStore:
    """시나리오별 memmap + 프레임→행 매핑 + 프레임 단위 LRU. 모달 cam/lidar/radar 만(pos 는 PosStore)."""
    def __init__(self, idx, mods, cache_root, frame_cache=0, log=print):
        self.mods = [m for m in mods if m in ("cam", "lidar", "radar")]; self.cache_root = cache_root; self.log = log
        self.arr, self.row, self.shape = {}, {}, {}
        for key in scen_list(idx):
            for m in self.mods:
                d = {"cam": f"{PATCH_RSU224}/{key[0]}/{key[1]}", "lidar": f"{cache_root}/lidar_pool8/{key[0]}/{key[1]}", "radar": f"{cache_root}/radar_raster64/{key[0]}/{key[1]}"}[m]
                fr_f = f"{d}/rsu_frames.npy" if m == "cam" else f"{d}/frames.npy"
                if not (os.path.exists(f"{d}/rsu.npy") and os.path.exists(fr_f)):
                    self.log(f"[sensor] MISSING cache {m} {key}: {d}"); continue
                a = np.load(f"{d}/rsu.npy", mmap_mode="r"); f = np.load(fr_f)
                self.arr[(m, key)] = a; self.row[(m, key)] = {int(x): j for j, x in enumerate(f)}; self.shape[m] = tuple(a.shape[1:])
        self.row_bytes = {m: int(np.prod(self.shape[m])) * 2 for m in self.mods if m in self.shape}
        self.frame_cache = frame_cache; self._lru = OrderedDict(); self.stats = dict(frame_hit=0, frame_miss=0, disk_bytes=0, missing_rows=0)

    def frame(self, key, fnum):
        """(town,scen), frame → dict mod → CPU fp16 tensor(행 복사). 캐시 없으면 0 텐서 + missing_rows 카운트."""
        k = (key, int(fnum))
        if k in self._lru:
            self._lru.move_to_end(k); self.stats["frame_hit"] += 1; return self._lru[k]
        out = {}
        for m in self.mods:
            a = self.arr.get((m, key)); r = self.row.get((m, key), {}).get(int(fnum))
            if a is None or r is None:
                self.stats["missing_rows"] += 1; out[m] = torch.zeros(self.shape.get(m, {"cam": CAM_SHAPE, "lidar": LIDAR_SHAPE, "radar": RADAR_SHAPE}[m]), dtype=torch.float16); continue
            v = np.array(a[r]); self.stats["disk_bytes"] += v.nbytes
            if m == "cam": v = v.reshape(CAM_SHAPE)
            out[m] = torch.from_numpy(v)
        self.stats["frame_miss"] += 1; self._lru[k] = out
        if self.frame_cache and len(self._lru) > self.frame_cache: self._lru.popitem(last=False)
        return out

    def coverage(self, idx):
        ft = np.asarray(idx["frame_traj"]); fn = np.asarray(idx["frame_num"]); trajs = idx["trajs"]; rep = {}
        for m in self.mods:
            miss, ex = 0, []
            for i in range(len(ft)):
                t = trajs[ft[i]]; key = (t["town"], t["scen"])
                if int(fn[i]) not in self.row.get((m, key), {}):
                    miss += 1
                    if len(ex) < 5: ex.append(f"{key[1]}/{t['cav']}/{int(fn[i]):06d}")
            rep[m] = dict(channel_frames=int(len(ft)), missing=miss, examples=ex, rsu_frames_cached=sum(len(v) for (mm, _), v in self.row.items() if mm == m), row_bytes=self.row_bytes.get(m))
        return rep


class PosStore:
    def __init__(self, cache_root, source="predicted", log=print):
        f = f"{cache_root}/pos/pos_features.npz"; assert os.path.exists(f), f"pos 캐시 없음: {f} (cp_sensor_data.py --build_pos)"
        z = np.load(f, allow_pickle=False); assert source in POS_SOURCES, source
        self.feat = torch.from_numpy(np.nan_to_num(z[f"feat_{source}"], nan=0.0)); self.nan_rows = int(np.isnan(z[f"feat_{source}"][:, 0]).sum()); self.source = source
        self.frames = z["frames"]; self.missing = [str(x) for x in z["missing"]]; self.rsu_info = json.loads(str(z["rsu_info"])); self.phi_true = z["phi_true"]; self.raw = {k: z[f"raw_{k}"] for k in ("true", "pred", "gps", "vel")}

    def window(self, s, K): return self.feat[s:s + K]  # [K,10] float32 (전역 프레임 순서 = index 순서)

    def coverage(self): return dict(source=self.source, nan_rows=self.nan_rows, missing_yaml=len(self.missing), examples=self.missing[:5], n=int(self.feat.shape[0]))


class MMWindowSet(WindowSet):
    """WindowSet + 센서. batch() 가 (X, Y, scale, meta, sensors) 를 돌려준다. sensors = {cam [B,K,196,768] fp16, lidar [B,K,64,384] fp16, radar [B,K,2,64,64] fp16, pos [B,K,10] fp32}."""
    def __init__(self, frames_fp16, windows, K_hist, H_pred, n_rx, idx, sensors, pos_source="predicted", cache_root=None, frame_cache=0, window_cache=256, share=None, log=print):
        super().__init__(frames_fp16, windows, K_hist, H_pred, n_rx)
        self.sensors = [s for s in (sensors.split(",") if isinstance(sensors, str) else sensors) if s]; bad = [s for s in self.sensors if s not in ALL_SENSORS]; assert not bad, bad
        self.idx = idx; self.frame_num = np.asarray(idx["frame_num"]); self.key_of_traj = [(t["town"], t["scen"]) for t in idx["trajs"]]; self.log = log
        cache_root = cache_root or DEFAULT_CACHE_ROOT
        if share is not None: self.store, self.pos = share.store, share.pos
        else:
            self.store = RSUSensorStore(idx, self.sensors, cache_root, frame_cache=frame_cache, log=log) if any(m in self.sensors for m in ("cam", "lidar", "radar")) else None
            self.pos = PosStore(cache_root, pos_source, log=log) if "pos" in self.sensors else None
        self.window_cache = window_cache; self._wlru = OrderedDict(); self.stats = dict(window_hit=0, window_miss=0, sensor_sec=0.0, batches=0)

    def window_sensors(self, w):
        s, ti = self.windows[w]; k = (s, ti)
        if k in self._wlru: self._wlru.move_to_end(k); self.stats["window_hit"] += 1; return self._wlru[k]
        out = {}
        if self.store is not None:
            key = self.key_of_traj[ti]; fr = [self.store.frame(key, f) for f in self.frame_num[s:s + self.K]]   # 이력 K 프레임만(미래 프레임 미사용)
            for m in self.store.mods: out[m] = torch.stack([f[m] for f in fr])
        if self.pos is not None: out["pos"] = self.pos.window(s, self.K)
        self.stats["window_miss"] += 1; self._wlru[k] = out
        if self.window_cache and len(self._wlru) > self.window_cache: self._wlru.popitem(last=False)
        return out

    def batch(self, item_ids, device):
        X, Y, scale, meta = super().batch(item_ids, device); t0 = time.time(); per = {}
        for i in item_ids:
            w, _ = self.items[i]
            for m, t in self.window_sensors(w).items(): per.setdefault(m, []).append(t)
        sens = {m: torch.stack(v).to(device, non_blocking=True) for m, v in per.items()}
        self.stats["sensor_sec"] += time.time() - t0; self.stats["batches"] += 1
        return X, Y, scale, meta, sens

    def window_bytes(self):
        b = {m: int(np.prod(self.store.shape[m])) * 2 * self.K for m in self.store.mods} if self.store is not None else {}
        if self.pos is not None: b["pos"] = POS_DIM * 4 * self.K
        return b

    def coverage(self):
        rep = {}
        if self.store is not None: rep.update(self.store.coverage(self.idx))
        if self.pos is not None: rep["pos"] = self.pos.coverage()
        return rep

    def stat_line(self):
        st = dict(self.stats); st.update({f"store_{k}": v for k, v in (self.store.stats.items() if self.store is not None else [])}); st["frames_in_ram"] = len(self.store._lru) if self.store is not None else 0; st["windows_in_ram"] = len(self._wlru)
        return st


def coverage_report(idx, cache_root, sensors=ALL_SENSORS, log=print):
    """커버리지 표 + 위치 변환 자체 점검(true_ego_pos → φ vs derived/aod phi_t)."""
    ws = MMWindowSet(torch.zeros(1, 1, 1, 1, 2, dtype=torch.float16), [], 16, 4, 16, idx, sensors, cache_root=cache_root, log=log)
    rep = ws.coverage(); log("[coverage] " + json.dumps(rep))
    chk = {}
    if ws.pos is not None:
        ft = np.asarray(idx["frame_traj"]); fn = np.asarray(idx["frame_num"]); errs = []; per = {}
        for ti, t in enumerate(idx["trajs"]):
            f = f"{AOD_DIR}/{t['town']}/{t['scen']}/{t['cav']}.npz"
            if not os.path.exists(f): continue
            z = np.load(f); pos = {int(x): j for j, x in enumerate(z["frames"])}; ii = np.where(ft == ti)[0]
            a = np.array([z["aod"][pos[int(fn[i])]] if int(fn[i]) in pos else np.nan for i in ii]); p = ws.pos.phi_true[ii]
            e = np.degrees(np.angle(np.exp(1j * (p - a)))); e = e[~np.isnan(e)]; errs.append(e); per[f"{t['scen']}/{t['cav']}"] = dict(median_abs_deg=float(np.median(np.abs(e))), p90_abs_deg=float(np.percentile(np.abs(e), 90)), n=int(len(e)))
        e = np.concatenate(errs) if errs else np.array([np.nan])
        chk = dict(what="atan2(y',x') of true_ego_pos in RSU-local vs derived/aod phi_t (dominant path AoD)", n=int(len(e)), median_abs_deg=float(np.median(np.abs(e))), p90_abs_deg=float(np.percentile(np.abs(e), 90)),
                   max_abs_deg=float(np.abs(e).max()), frac_within_2deg=float((np.abs(e) < 2).mean()), per_traj=per)
        log(f"[pos-check] φ(true pos, RSU-local) vs AoD: n {chk['n']} median |Δ| {chk['median_abs_deg']:.2f}° p90 {chk['p90_abs_deg']:.2f}° max {chk['max_abs_deg']:.1f}° within 2°: {chk['frac_within_2deg']:.3f}")
    return rep, chk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build_lidar_pool", action="store_true"); ap.add_argument("--build_radar", action="store_true"); ap.add_argument("--build_pos", action="store_true"); ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--cache_root", default=DEFAULT_CACHE_ROOT); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--log", default=None)
    a = ap.parse_args(); os.makedirs(a.cache_root, exist_ok=True)
    lf = open(a.log, "a") if a.log else None
    def log(*s):
        t = " ".join(str(x) for x in s); print(t, flush=True)
        if lf: lf.write(t + "\n"); lf.flush()
    idx, _ = load_index()
    if a.build_lidar_pool: build_lidar_pool(idx, a.cache_root, device=a.device, batch=a.batch, log=log)
    if a.build_radar: build_radar(idx, a.cache_root, log=log)
    if a.build_pos: build_pos(idx, a.cache_root, log=log)
    if a.coverage:
        rep, chk = coverage_report(idx, a.cache_root, log=log)
        json.dump(dict(coverage=rep, pos_check=chk, cache_root=a.cache_root, du_bytes={d: sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(f"{a.cache_root}/{d}") for f in fs) for d in ("lidar_pool8", "radar_raster64", "pos") if os.path.isdir(f"{a.cache_root}/{d}")}),
                  open(f"{a.cache_root}/coverage.json", "w"), indent=1)
        log(f"saved {a.cache_root}/coverage.json")
