#!/usr/bin/env python3
"""OpenCOOD(OPV2V 공식 구현, [B]의 [17]) PointPillars **앞 두 단계**를 동결 상태로 재현 — 논문 [B] III-A2.

논문 원문: "the point cloud is first processed by a frozen pre-processor that adopts the first two stages of the
PointPillars architecture, i.e., a feature encoder and a 2D convolutional backbone ... the weights for these two
stages are initialized from a public implementation [17] and remain frozen during training."

구성 (OpenCOOD `opencood/models/point_pillar.py` 조립 순서 그대로, 검출 head `cls_head`/`reg_head`만 제거):
  points[N,4](x,y,z,intensity) → voxelize(spconv VoxelGenerator 규약 복제) → PillarVFE(PointNet 64ch, frozen)
  → PointPillarScatter(pseudo-image 64×ny×nx) → BaseBEVBackbone(3블록 [3,5,8]층 + upsample → 384ch, frozen)
  → X_L ∈ R^{384 × ny/2 × nx/2}   (논문 X_L ∈ R^{C_L×H_L×W_L}; 행=y, 열=x)

모듈 코드는 OpenCOOD `sub_modules/{pillar_vfe,point_pillar_scatter,base_bev_backbone}.py`를 그대로 옮김
(원 출처 OpenPCDet; OpenCOOD 라이선스 UCLA Academic Software License — 학술 내부 사용, 재배포 금지).
체크포인트·config·출처: `pretrained/opencood/README.md`, `REPRO_V2_PREREQUISITES.md` §3.15.
"""
from __future__ import annotations
import glob, hashlib, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml

ROOT = '/mnt/ssd_7t_2/carla-wireless-dataset/mmw_reproduction'

# ───────────────────────── 사전학습 가중치 출처 (2026-08-30 확정) ─────────────────────────
# 논문 [B] arXiv:2603.15093 III-A2: "the weights for these two stages are initialized from a public implementation [17]
#   and remain frozen"  — [17] = Xu et al., "OPV2V: An Open Benchmark Dataset and Fusion Pipeline for Perception with
#   Vehicle-to-Vehicle Communication," ICRA 2022. 논문은 체크포인트명·URL을 적지 않음(8종 중 어느 것인지 미특정).
# 공개 구현 = OpenCOOD (OPV2V 공식 코드) https://github.com/DerrickXuNu/OpenCOOD
#   main 브랜치 커밋 31ba16025da27ffe4e336f011290dfbc66f9a1f1 (2024-08-17), README "Benchmark and model zoo" → OPV2V LiDAR-track 표
#   다운로드: UCLA Box 공유 폴더 https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V
#     · Naive Late (PointPillar, fusion 없음) : file/1621128604521 → pointpillar_naive_late.zip
#     · Cooper (PointPillar, Early fusion)     : file/1621122534978 → pointpillar_early_cooper.zip
#   라이선스: repo LICENSE = UCLA Academic Software License (© 2021 UCLA Mobility Lab) — 학술·비영리 연구 사용 허용, 재배포 금지.
#   (README 배지는 "MIT"이나 LICENSE 파일이 우선. 가중치 파일을 외부 repo/PPT에 올리지 말 것)
# 로컬 사본 (2026-08-30 다운로드, zip SHA256은 pretrained/opencood/SHA256SUMS):
#   CKPT_NAIVE_LATE/net_epoch30.pth  26,419,373 B  md5 eed40b69d9c3c6e3c5ce5787bba0a034  (config name point_pillar_late_fusion_low_res)
#   CKPT_COOPER/latest.pth           26,419,373 B  md5 83851f7bf3fe3471cb053e34d5534f92  (config name point_pillar_early_fusion_low_res)
#   동봉 config.yaml: voxel_size [0.4,0.4,4] m, max_points_per_voxel 32, pillar_vfe 64ch, backbone [3,5,8]/[64,128,256] → 384ch;
#     lidar_range Naive Late [-70.4,-40,-3,70.4,40,1] (max_voxel 16000/40000) · Cooper [-140.8,-40,-3,140.8,40,1] (32000/70000)
# 사용 부분: pillar_vfe.* + backbone.* (138 키, 6,578,176 파라미터) = 논문 "feature encoder + 2D convolutional backbone".
#   검출 head cls_head.*/reg_head.* 4 키는 로드 시 버림. 전부 동결(requires_grad=False, BN eval 고정).
# 상세: pretrained/opencood/README.md, REPRO_V2_PREREQUISITES.md §3.15
CKPT_NAIVE_LATE = f'{ROOT}/pretrained/opencood/pointpillar_naive_late/pointpillar_late_fusion'   # 권장: 단일 차량 PointPillar (OpenCOOD "Naive Late", Box file 1621128604521)
CKPT_COOPER = f'{ROOT}/pretrained/opencood/pointpillar_early_cooper/pointpillar_early_fusion'   # 대안: early fusion (OpenCOOD "Cooper", Box file 1621122534978)


# ───────────────────────── config (체크포인트 동봉 config.yaml) ─────────────────────────
class _IgnorePyTags(yaml.SafeLoader):
    """config.yaml 안의 numpy python/object 태그(grid_size)는 무시하고 직접 재계산."""
_IgnorePyTags.add_multi_constructor('tag:yaml.org,2002:python/', lambda loader, suffix, node: None)


def load_config(ckpt_dir: str) -> dict:
    cfg = yaml.load(open(os.path.join(ckpt_dir, 'config.yaml')), Loader=_IgnorePyTags)
    m, pre = cfg['model']['args'], cfg['preprocess']['args']
    rng = [float(v) for v in m['lidar_range']]; vs = [float(v) for v in m['voxel_size']]
    grid = np.round((np.array(rng[3:]) - np.array(rng[:3])) / np.array(vs)).astype(int)   # [nx, ny, nz]
    return dict(name=cfg.get('name'), lidar_range=rng, voxel_size=vs, grid_size=[int(g) for g in grid],
                max_points_per_voxel=int(pre['max_points_per_voxel']),
                max_voxel_train=int(pre['max_voxel_train']), max_voxel_test=int(pre['max_voxel_test']),
                pillar_vfe=m['pillar_vfe'],
                point_pillar_scatter=dict(num_features=int(m['point_pillar_scatter'].get('num_features', 64)),
                                          grid_size=[int(g) for g in grid]),
                base_bev_backbone=m['base_bev_backbone'])


# ───────────────────────── voxelization (spconv VoxelGenerator 동작 복제, numpy) ─────────────────────────
def voxelize(points: np.ndarray, voxel_size, pc_range, max_points: int, max_voxels: int):
    """points[N,F] (앞 3열 = x,y,z) → voxels[M,max_points,F], coords[M,3]=(z,y,x) int32, num_points[M] int32.
    spconv 규약: 범위 밖 점 제거, voxel은 점 **등장 순서**대로 생성, voxel당 처음 max_points개 점만, 처음 max_voxels개 voxel만."""
    r = np.asarray(pc_range, np.float32); vs = np.asarray(voxel_size, np.float32)
    grid = np.round((r[3:] - r[:3]) / vs).astype(np.int64)
    idx = np.floor((points[:, :3] - r[:3]) / vs).astype(np.int64)
    ok = np.all((idx >= 0) & (idx < grid), axis=1)
    pts, idx = points[ok].astype(np.float32), idx[ok]
    Fdim = points.shape[1]
    if len(pts) == 0:
        return np.zeros((0, max_points, Fdim), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.int32)
    vid = (idx[:, 2] * grid[1] + idx[:, 1]) * grid[0] + idx[:, 0]
    uniq, first, inv = np.unique(vid, return_index=True, return_inverse=True)
    order = np.argsort(first, kind='stable')                  # voxel 등장 순서
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    vrank = rank[inv]                                          # 점별 voxel 순위
    o = np.argsort(vrank, kind='stable'); vr_sorted = vrank[o]
    pos_sorted = np.arange(len(o)) - np.searchsorted(vr_sorted, vr_sorted)   # voxel 내 점 순위(등장 순)
    pos = np.empty_like(pos_sorted); pos[o] = pos_sorted
    keep = (vrank < max_voxels) & (pos < max_points)
    M = int(min(len(uniq), max_voxels))
    voxels = np.zeros((M, max_points, Fdim), np.float32)
    voxels[vrank[keep], pos[keep]] = pts[keep]
    num = np.bincount(vrank[keep], minlength=M).astype(np.int32)
    vidx = idx[first[order[:M]]]                                # voxel별 (ix,iy,iz)
    coords = np.stack([vidx[:, 2], vidx[:, 1], vidx[:, 0]], 1).astype(np.int32)   # (z,y,x)
    return voxels, coords, num


# ───────────────────────── OpenCOOD sub_modules (원문 그대로) ─────────────────────────
class PFNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, last_layer=False):
        super().__init__()
        self.last_vfe = last_layer; self.use_norm = use_norm
        if not self.last_vfe: out_channels = out_channels // 2
        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            num_parts = inputs.shape[0] // self.part
            x = torch.cat([self.linear(inputs[n * self.part:(n + 1) * self.part]) for n in range(num_parts + 1)], 0)
        else:
            x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        x_max = torch.max(x, dim=1, keepdim=True)[0]
        if self.last_vfe: return x_max
        return torch.cat([x, x_max.repeat(1, inputs.shape[1], 1)], dim=2)


class PillarVFE(nn.Module):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__()
        self.use_norm = model_cfg['use_norm']; self.with_distance = model_cfg['with_distance']
        self.use_absolute_xyz = model_cfg['use_absolute_xyz']
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance: num_point_features += 1
        self.num_filters = model_cfg['num_filters']
        num_filters = [num_point_features] + list(self.num_filters)
        self.pfn_layers = nn.ModuleList([PFNLayer(num_filters[i], num_filters[i + 1], self.use_norm,
                                                  last_layer=(i >= len(num_filters) - 2)) for i in range(len(num_filters) - 1)])
        self.voxel_x, self.voxel_y, self.voxel_z = voxel_size
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    @staticmethod
    def get_paddings_indicator(actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape); max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        return actual_num.int() > max_num

    def forward(self, batch_dict):
        voxel_features, voxel_num_points, coords = batch_dict['voxel_features'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        f_cluster = voxel_features[:, :, :3] - points_mean
        f_center = torch.zeros_like(voxel_features[:, :, :3])
        f_center[:, :, 0] = voxel_features[:, :, 0] - (coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
        features = [voxel_features, f_cluster, f_center] if self.use_absolute_xyz else [voxel_features[..., 3:], f_cluster, f_center]
        if self.with_distance: features.append(torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True))
        features = torch.cat(features, dim=-1)
        mask = self.get_paddings_indicator(voxel_num_points, features.shape[1], axis=0)
        features *= torch.unsqueeze(mask, -1).type_as(voxel_features)
        for pfn in self.pfn_layers: features = pfn(features)
        batch_dict['pillar_features'] = features.squeeze(1)          # 원문 .squeeze() (M=1 안전을 위해 dim 지정)
        return batch_dict


class PointPillarScatter(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.num_bev_features = model_cfg['num_features']
        self.nx, self.ny, self.nz = model_cfg['grid_size']; assert self.nz == 1

    def forward(self, batch_dict):
        pillar_features, coords = batch_dict['pillar_features'], batch_dict['voxel_coords']
        batch_size = coords[:, 0].max().int().item() + 1
        out = []
        for b in range(batch_size):
            sf = torch.zeros(self.num_bev_features, self.nz * self.nx * self.ny, dtype=pillar_features.dtype, device=pillar_features.device)
            m = coords[:, 0] == b; c = coords[m, :]
            indices = (c[:, 1] + c[:, 2] * self.nx + c[:, 3]).type(torch.long)
            sf[:, indices] = pillar_features[m, :].t(); out.append(sf)
        batch_dict['spatial_features'] = torch.stack(out, 0).view(batch_size, self.num_bev_features * self.nz, self.ny, self.nx)
        return batch_dict


class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        layer_nums, layer_strides, num_filters = model_cfg['layer_nums'], model_cfg['layer_strides'], model_cfg['num_filters']
        num_upsample_filters, upsample_strides = model_cfg['num_upsample_filter'], model_cfg['upsample_strides']
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks, self.deblocks = nn.ModuleList(), nn.ModuleList()
        for idx in range(len(layer_nums)):
            cur = [nn.ZeroPad2d(1), nn.Conv2d(c_in_list[idx], num_filters[idx], 3, stride=layer_strides[idx], padding=0, bias=False),
                   nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01), nn.ReLU()]
            for _ in range(layer_nums[idx]):
                cur += [nn.Conv2d(num_filters[idx], num_filters[idx], 3, padding=1, bias=False),
                        nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01), nn.ReLU()]
            self.blocks.append(nn.Sequential(*cur))
            stride = upsample_strides[idx]
            if stride >= 1:
                self.deblocks.append(nn.Sequential(nn.ConvTranspose2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False),
                                                   nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01), nn.ReLU()))
            else:
                stride = int(np.round(1 / stride))
                self.deblocks.append(nn.Sequential(nn.Conv2d(num_filters[idx], num_upsample_filters[idx], stride, stride=stride, bias=False),
                                                   nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01), nn.ReLU()))
        self.num_bev_features = sum(num_upsample_filters)

    def forward(self, data_dict):
        x = data_dict['spatial_features']; ups = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x); ups.append(self.deblocks[i](x))
        data_dict['spatial_features_2d'] = torch.cat(ups, dim=1) if len(ups) > 1 else ups[0]
        return data_dict


# ───────────────────────── frozen wrapper ─────────────────────────
def _md5(path, n=1 << 20):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(n), b''): h.update(chunk)
    return h.hexdigest()


class PointPillarsFrozen(nn.Module):
    """OpenCOOD PointPillar에서 pillar_vfe + scatter + backbone만 로드(strict), 검출 head 제거, 영구 eval + no-grad.
    forward(list of points[N,4]) → X_L [B, 384, ny/2, nx/2] (float32)."""
    def __init__(self, ckpt_dir: str = CKPT_NAIVE_LATE, max_voxels: int | None = None):
        super().__init__()
        self.cfg = c = load_config(ckpt_dir)
        self.max_voxels = int(max_voxels or c['max_voxel_test'])
        self.pillar_vfe = PillarVFE(c['pillar_vfe'], 4, c['voxel_size'], c['lidar_range'])
        self.scatter = PointPillarScatter(c['point_pillar_scatter'])
        self.backbone = BaseBEVBackbone(c['base_bev_backbone'], c['point_pillar_scatter']['num_features'])
        # 가중치 로드: OpenCOOD 모델 zoo 체크포인트(위 출처 블록 참조) — 저자가 OPV2V(CARLA LiDAR)에서 검출용으로 학습한 것을
        # 우리가 추가 학습 없이 그대로 사용. md5를 기록해 어느 파일을 썼는지 result.json/meta.json에 남김.
        pths = sorted(glob.glob(os.path.join(ckpt_dir, '*.pth'))); assert len(pths) == 1, pths
        self.ckpt_path = pths[0]; self.ckpt_md5 = _md5(self.ckpt_path)
        sd = torch.load(self.ckpt_path, map_location='cpu', weights_only=False)
        sd = sd.get('model_state_dict', sd) if isinstance(sd, dict) and 'model_state_dict' in sd else sd
        self.dropped_keys = sorted(k for k in sd if k.startswith(('cls_head', 'reg_head')))
        sd = {k: v for k, v in sd.items() if k not in self.dropped_keys}
        self.load_state_dict(sd, strict=True)                      # 키 불일치 시 즉시 실패(무수정 로드 보장)
        for p in self.parameters(): p.requires_grad_(False)
        self.eval()
        self.out_channels = self.backbone.num_bev_features          # 384
        nx, ny, _ = c['grid_size']; self.out_hw = (ny // 2, nx // 2)   # (H_L, W_L) = (100, 176) for Naive Late
        self.n_params = sum(p.numel() for p in self.parameters())

    def train(self, mode: bool = True):                            # BN 통계 고정: 어떤 경우에도 eval 유지
        return super().train(False)

    @torch.no_grad()
    def forward(self, point_list, device=None):
        device = device or next(self.parameters()).device
        vf, vc, vn = [], [], []
        for i, p in enumerate(point_list):
            v, c, n = voxelize(p, self.cfg['voxel_size'], self.cfg['lidar_range'], self.cfg['max_points_per_voxel'], self.max_voxels)
            if len(v) == 0:                                         # 빈 프레임 보호: 0-voxel 1개
                v = np.zeros((1, self.cfg['max_points_per_voxel'], p.shape[1]), np.float32); c = np.zeros((1, 3), np.int32); n = np.ones(1, np.int32)
            vf.append(v); vn.append(n); vc.append(np.pad(c, ((0, 0), (1, 0)), constant_values=i))
        bd = {'voxel_features': torch.from_numpy(np.concatenate(vf)).to(device),
              'voxel_coords': torch.from_numpy(np.concatenate(vc)).to(device),
              'voxel_num_points': torch.from_numpy(np.concatenate(vn)).to(device)}
        bd = self.backbone(self.scatter(self.pillar_vfe(bd)))
        return bd['spatial_features_2d']

    def describe(self) -> dict:
        c = self.cfg
        return dict(ckpt=self.ckpt_path, ckpt_md5=self.ckpt_md5, config_name=c['name'], lidar_range=c['lidar_range'],
                    voxel_size=c['voxel_size'], grid_size=c['grid_size'], max_points_per_voxel=c['max_points_per_voxel'],
                    max_voxels=self.max_voxels, out_channels=self.out_channels, out_hw=list(self.out_hw),
                    n_params=self.n_params, dropped_keys=self.dropped_keys)


# ───────────────────────── RSU 점구름 → PointPillars 입력 ─────────────────────────
def rsu_points_to_pp_input(xyz: np.ndarray, z_shift: float = -2.1, intensity: str = 'carla', atten: float = 0.004,
                           intensity_const: float = 1.0) -> np.ndarray:
    """RSU LiDAR 센서 프레임 xyz[N,3] → PointPillars 입력 [N,4] (배열 좌표계, OPV2V z 창, intensity 복원).
    - y 부호반전: CARLA(왼손) → Sionna 배열 좌표계(오른손). 셀 방위 atan2(y,x)가 빔각과 같은 프레임 (precompute_features_rsu.py와 동일)
    - z_shift: OPV2V 창 [−3,1]은 지붕 LiDAR(≈1.9 m) 기준. RSU 4 m라 지면이 z≈−4 → z − z_shift 로 옮겨 같은 지면-상대 창 적용
      (기본 −2.1 → 센서 프레임 [−5.1, −1.1]). 0이면 config 창 그대로(점 19~46%만 생존, 차량 몸체 절단).
    - intensity: [A] pcd에는 없음(FIELDS x y z rgb, rgb≈0). 'carla' = CARLA LiDAR 기본 모델 exp(−atten·d), 'const' = 상수."""
    x, y, z = xyz[:, 0].astype(np.float32), -xyz[:, 1].astype(np.float32), xyz[:, 2].astype(np.float32) - np.float32(z_shift)
    if intensity == 'carla':
        i = np.exp(-atten * np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)).astype(np.float32)
    elif intensity == 'const':
        i = np.full(len(x), intensity_const, np.float32)
    else:
        raise ValueError(intensity)
    return np.stack([x, y, z, i], 1)


def cell_centers(lidar_range, out_hw):
    """backbone 출력 격자(H_L,W_L)의 셀 중심 좌표(배열 프레임, m). 행=y, 열=x. BGAM Θ[i,j] 계산용."""
    H, W = out_hw; x0, y0, _, x1, y1, _ = lidar_range
    cw, ch = (x1 - x0) / W, (y1 - y0) / H
    xs = x0 + (np.arange(W) + 0.5) * cw; ys = y0 + (np.arange(H) + 0.5) * ch
    return np.meshgrid(xs, ys)   # X[H,W], Y[H,W]
