"""저장소 기존 채널 백본 래퍼 (multimodal_code_index/models). 인터페이스 통일: forward(X [B,K,Nt,Ksc,2]) -> [B,H,Nt,Ksc,2].
우리 데이터에서는 '안테나 축' = RSU 송신 안테나 Nt=64 (RX 행 표본), 부반송파 Ksc=64.
"""
import os, sys, torch, torch.nn as nn
REPO = os.environ.get("CP_REPO", "/mnt/ssd_7t_2/carla-wireless-dataset/multimodal_code_index")  # 로컬 실행: CP_REPO 로 덮어씀
if REPO not in sys.path: sys.path.insert(0, REPO)


class RepoWrap(nn.Module):
    def __init__(self, m, H, Nt, Ksc):
        super().__init__(); self.m = m; self.H, self.Nt, self.Ksc = H, Nt, Ksc
    def forward(self, X, Y=None):
        out = self.m(X)
        if isinstance(out, dict):
            for k in ("prediction", "pred", "channel", "output"):
                if k in out: out = out[k]; break
            else: out = next(v for v in out.values() if torch.is_tensor(v))
        return out.reshape(X.shape[0], self.H, self.Nt, self.Ksc, 2)


def build_repo_model(name, K, H, Nt, Ksc, **kw):
    D = kw.get("D", 256); L = kw.get("L", 6); dropout = kw.get("dropout", 0.1)
    if name == "chiron":
        from models.chiron_channel import ChironChannelPredictor
        m = ChironChannelPredictor(num_bs_antennas=Nt, num_subcarriers=Ksc, embed_dim=D, depth=L, num_heads=4, history_len=K, prediction_horizon=H, patch_h=4, patch_w=32, dropout=dropout)
    elif name == "chiron_v2":
        from models.chiron_v2 import ChironV2
        m = ChironV2(num_bs_antennas=Nt, num_subcarriers=Ksc, embed_dim=D, depth=L, num_heads=4, history_len=K, prediction_horizon=H, dropout=dropout)
    elif name == "nova":
        from models.nova_channel import NOVAChannelPredictor
        m = NOVAChannelPredictor(num_bs_antennas=Nt, num_subcarriers=Ksc, embed_dim=D, depth=L, num_heads=4, history_len=K, prediction_horizon=H, patch_h=4, patch_w=8, dropout=dropout)
    elif name == "dtcn":
        from models.delay_tcn import DelayTCN
        m = DelayTCN(num_bs_antennas=Nt, num_subcarriers=Ksc, embed_dim=D, history_len=K, prediction_horizon=H, dropout=dropout)
    elif name == "mamba":
        from models.mamba_channel import MambaChannelPredictor
        m = MambaChannelPredictor(num_bs_antennas=Nt, num_subcarriers=Ksc, d_model=kw.get("D", 128), depth=kw.get("L", 4), history_len=K, prediction_horizon=H, dropout=dropout)
    elif name == "lwm_temporal":
        from models.lwm_temporal_multimodal import LWMTemporalMultiModalPredictor
        m = LWMTemporalMultiModalPredictor(mode="channel_only", num_bs_antennas=Nt, num_subcarriers=Ksc, history_len=K, prediction_horizon=H,
                                           patch_h=kw.get("patch_h", 4), patch_w=kw.get("patch_w", 16), embed_dim=D, depth=L, num_heads=8, use_image=False, use_lidar=False, pretrained_image=False)
    elif name == "lwm":
        from models.lwm_multimodal import LWMMultiModalPredictor
        m = LWMMultiModalPredictor(mode="channel_only", num_bs_antennas=Nt, num_subcarriers=Ksc, history_len=K, prediction_horizon=H, d_model=kw.get("D", 128), n_layers=kw.get("L", 12), n_heads=8, d_ff=4 * kw.get("D", 128),
                                   use_image=False, use_lidar=False, pretrained_image=False, delta_t=0.01)
    else:
        raise ValueError(name)
    return RepoWrap(m, H, Nt, Ksc)
