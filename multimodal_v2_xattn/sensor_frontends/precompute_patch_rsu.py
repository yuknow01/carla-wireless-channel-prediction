#!/usr/bin/env python3
"""RSU 카메라 ViT-B/16 patch 토큰(196) 캐시 — 충실 재현(GPT-2)용 (2026-07-30).

논문: "extract the sequence of transformed patch tokens to preserve spatial information".
RSU는 시나리오당 카메라 1대 → [Nf,1,196,768] fp16, 시나리오 단위 저장(모든 cav 공유).
출력: derived/feat_patch_rsu/<town>/<scen>/rsu.npy + rsu_frames.npy (memmap 호환 비압축)
사용: hoyun_312 python precompute_patch_rsu.py --gpu 0 --shard 0 --nshard 4
"""
import sys, os, glob, re, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from PIL import Image
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import transforms as T

ROOT = '/mnt/ssd_7t_2/carla-wireless-dataset/mmw_reproduction'
CH = f'{ROOT}/sunny/channel_data/v2i/Nt_1_64_Nr_1_16_fc_28GHz'
SE = f'{ROOT}/sunny/sensor_data'
OUT = f'{ROOT}/derived/feat_patch_rsu'        # (구) torchvision 기본 전처리: 짧은 변 256 → 224 crop — 논문과 다름
OUT224 = f'{ROOT}/derived/feat_patch_rsu224'  # 논문 III-A3: 짧은 변 224 리사이즈 → 224 center crop (2026-08-30)

def frame_num(p): return int(re.findall(r'\d+', os.path.basename(p))[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--resize', type=int, default=224, choices=[224, 256],
                    help='224 = 논문 III-A3 "짧은 변 224로 등방 리사이즈 후 224 center crop" (출력 feat_patch_rsu224) / 256 = (구) torchvision 기본')
    args = ap.parse_args()
    dev = torch.device(f'cuda:{args.gpu}')
    w = ViT_B_16_Weights.IMAGENET1K_V1
    vit = vit_b_16(weights=w).to(dev).eval()
    for p in vit.parameters(): p.requires_grad_(False)
    if args.resize == 224:
        # 논문 III-A3 원문대로: 짧은 변 224 등방 리사이즈 → 224×224 center crop → ImageNet 정규화 (torchvision 기본은 256→224라 FOV 손실)
        tf = T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        out_root = OUT224
    else:
        tf = w.transforms(); out_root = OUT

    for sd in sorted(glob.glob(f'{CH}/Town*/*'))[args.shard::args.nshard]:
        scen = os.path.basename(sd); town = os.path.basename(os.path.dirname(sd))
        outp = f'{out_root}/{town}/{scen}/rsu.npy'
        if os.path.exists(outp): print('skip', outp, flush=True); continue
        sscen = [os.path.basename(p) for p in glob.glob(f'{SE}/{town}/{scen}*') if os.path.isdir(p)][0]
        rdir = f'{SE}/{town}/{sscen}/rsu_1'
        frames = sorted(frame_num(p) for p in glob.glob(f'{rdir}/[0-9]*.pcd'))
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        arr = np.lib.format.open_memmap(outp + '.tmp', mode='w+', dtype=np.float16,
                                        shape=(len(frames), 1, 196, 768))
        for b0 in range(0, len(frames), args.batch):
            frs = frames[b0:b0+args.batch]
            ims = torch.stack([tf(Image.open(f'{rdir}/{fr:06d}_camera0.png').convert('RGB'))
                               for fr in frs]).to(dev)
            with torch.no_grad():
                feats = vit._process_input(ims)
                cls = vit.class_token.expand(ims.shape[0], -1, -1)
                z = vit.encoder(torch.cat([cls, feats], 1))[:, 1:]       # [b,196,768] patch
            arr[b0:b0+len(frs), 0] = z.cpu().numpy().astype(np.float16)
        arr.flush(); del arr
        os.rename(outp + '.tmp', outp)
        np.save(f'{out_root}/{town}/{scen}/rsu_frames.npy', np.array(frames, np.int32))
        print(f'done {town}/{scen} ({len(frames)}f)', flush=True)
    print('=== shard 완료 ===', flush=True)

if __name__ == '__main__':
    main()
