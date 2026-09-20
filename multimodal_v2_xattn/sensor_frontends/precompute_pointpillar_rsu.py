#!/usr/bin/env python3
"""RSU LiDAR → 동결 PointPillars(OpenCOOD) BEV 특징맵 X_L 캐시 — 논문 [B] III-A2 그대로 (2026-08-30).

논문: PointPillars 앞 두 단계(frozen, [17] OpenCOOD 가중치) → X_L ∈ R^{C_L×H_L×W_L}. 학습 중 변하지 않으므로 프레임별로 미리 계산.
가중치 출처: OpenCOOD(OPV2V 공식 코드, github.com/DerrickXuNu/OpenCOOD 커밋 31ba160) 모델 zoo — 기본 "Naive Late"(UCLA Box file
  1621128604521, 로컬 pretrained/opencood/pointpillar_naive_late/.../net_epoch30.pth md5 eed40b69…), 대안 "Cooper"(file 1621122534978).
  상세 mmw_repro/pointpillars_opencood.py 상단 출처 블록. 사용한 체크포인트 경로·md5는 meta.json에 기록됨.
RSU는 시나리오당 1대·정적 → 시나리오 단위 1회(모든 cav 공유).
시간 정렬(논문 III-B1, 식 20): LiDAR는 채널(100 Hz)보다 느린 j=10(10 Hz) → `--every 10` 프레임만 계산하고,
  학습 시 dataset_faithful.py가 "가장 최근 측정을 뒤로 복제(backward replication)"로 40스텝을 채움. every=1이면 원해상도.
출력: derived/feat_pp_rsu/<town>/<scen>/rsu.npy  [Nf, 384, H_L, W_L] fp16 memmap
      derived/feat_pp_rsu/<town>/<scen>/rsu_frames.npy, meta.json(격자·범위·체크포인트 md5·옵션)
사용: hoyun_312 python precompute_pointpillar_rsu.py --gpu 0 --shard 0 --nshard 4
"""
import sys, os, glob, re, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch, open3d as o3d
from mmw_repro.pointpillars_opencood import PointPillarsFrozen, rsu_points_to_pp_input, CKPT_NAIVE_LATE, CKPT_COOPER

ROOT = '/mnt/ssd_7t_2/carla-wireless-dataset/mmw_reproduction'
CH = f'{ROOT}/sunny/channel_data/v2i/Nt_1_64_Nr_1_16_fc_28GHz'
SE = f'{ROOT}/sunny/sensor_data'

def frame_num(p): return int(re.findall(r'\d+', os.path.basename(p))[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='naive_late', help="naive_late | cooper | <체크포인트 디렉토리 경로>")
    ap.add_argument('--out', default=f'{ROOT}/derived/feat_pp_rsu')
    ap.add_argument('--every', type=int, default=1, help='캐시 프레임 간격. 논문 식(20) 윈도우 앵커 정렬(dataset align=window)은 임의 프레임이 구간 끝이 되므로 1(전 프레임, 16시나리오 ~238 GB) 필요. 10 = 고정 그리드 근사(24 GB)')
    ap.add_argument('--z_shift', type=float, default=-2.1, help='RSU 4 m 높이 보정 (0 = OpenCOOD z창 그대로)')
    ap.add_argument('--intensity', default='carla', choices=['carla', 'const'])
    ap.add_argument('--intensity_const', type=float, default=1.0)
    ap.add_argument('--atten', type=float, default=0.004, help='CARLA LiDAR atmosphere_attenuation_rate 기본값')
    ap.add_argument('--max_voxels', type=int, default=None, help='기본 = config max_voxel_test')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0); ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--batch', type=int, default=8)
    args = ap.parse_args()
    ckpt = {'naive_late': CKPT_NAIVE_LATE, 'cooper': CKPT_COOPER}.get(args.ckpt, args.ckpt)
    dev = torch.device(f'cuda:{args.gpu}')
    pp = PointPillarsFrozen(ckpt, max_voxels=args.max_voxels).to(dev)
    C, (H, W) = pp.out_channels, pp.out_hw
    meta_common = dict(pointpillars=pp.describe(), every=args.every, z_shift=args.z_shift, intensity=args.intensity,
                       intensity_const=args.intensity_const, atten=args.atten, channels=C, out_hw=[H, W],
                       frame_axes='row=y(array frame, CARLA y 반전), col=x', created=time.strftime('%Y-%m-%d %H:%M'))
    print(json.dumps(meta_common['pointpillars'], indent=1), flush=True)

    for sd in sorted(glob.glob(f'{CH}/Town*/*'))[args.shard::args.nshard]:
        scen = os.path.basename(sd); town = os.path.basename(os.path.dirname(sd))
        od = f'{args.out}/{town}/{scen}'; outp = f'{od}/rsu.npy'
        if os.path.exists(outp): print('skip', outp, flush=True); continue
        sscen = [os.path.basename(p) for p in glob.glob(f'{SE}/{town}/{scen}*') if os.path.isdir(p)][0]
        rdir = f'{SE}/{town}/{sscen}/rsu_1'
        frames = sorted(frame_num(p) for p in glob.glob(f'{rdir}/[0-9]*.pcd'))[::args.every]
        os.makedirs(od, exist_ok=True)
        arr = np.lib.format.open_memmap(outp + '.tmp', mode='w+', dtype=np.float16, shape=(len(frames), C, H, W))
        t0 = time.time(); npts = []
        for b0 in range(0, len(frames), args.batch):
            frs = frames[b0:b0 + args.batch]; pts = []
            for fr in frs:
                xyz = np.asarray(o3d.io.read_point_cloud(f'{rdir}/{fr:06d}.pcd').points, np.float32)
                p4 = rsu_points_to_pp_input(xyz, args.z_shift, args.intensity, args.atten, args.intensity_const)
                pts.append(p4); npts.append(len(xyz))
            with torch.no_grad():
                xl = pp(pts, dev)                                   # [b, C, H, W]
            arr[b0:b0 + len(frs)] = xl.half().cpu().numpy()
        arr.flush(); del arr; os.rename(outp + '.tmp', outp)
        np.save(f'{od}/rsu_frames.npy', np.array(frames, np.int32))
        json.dump(dict(meta_common, town=town, scen=scen, sensor_scen=sscen, n_frames=len(frames),
                       mean_points=float(np.mean(npts))), open(f'{od}/meta.json', 'w'), indent=1)
        print(f'done {town}/{scen} ({len(frames)}f, {time.time()-t0:.0f}s, {os.path.getsize(outp)/1e9:.2f} GB)', flush=True)
    print('=== shard 완료 ===', flush=True)

if __name__ == '__main__':
    main()
