"""채널 예측 모델 레지스트리. 인터페이스: forward(X [B,K,Nt,Ksc,2]) -> [B,H,Nt,Ksc,2].
- transformer : 프레임 평탄화 -> 선형 임베딩 D + 학습 위치임베딩 -> pre-norm TransformerEncoder(L, 8헤드, FFN 4D) -> 마지막 토큰 -> 지평별 선형 헤드
- lstm        : 프레임 평탄화 -> 선형 임베딩 -> 2층 LSTM -> 마지막 은닉 -> 지평별 선형 헤드
- convlstm_ae : Conv2D 공간 인코더 -> 2층 ConvLSTM 인코더/디코더(자기회귀 H스텝, 학습 시 teacher forcing) -> TransConv 디코더 (P04 구조)
- lwm11       : LWM v1.1(wi-lab, 사전학습 가중치) 프레임 인코더 + 패치별 시간 Transformer (cp_lwm11.py)
- repo:<name> : 저장소 기존 백본(chiron/nova/dtcn/mamba/lwm_temporal) 래퍼 (cp_repo_models.py)
"""
import math, torch, torch.nn as nn


class FrameTokenizer(nn.Module):
    def __init__(self, Nt, Ksc, D):
        super().__init__(); self.Nt, self.Ksc = Nt, Ksc; self.proj = nn.Linear(Nt * Ksc * 2, D)
    def forward(self, X):  # [B,K,Nt,Ksc,2] -> [B,K,D]
        B, K = X.shape[:2]; return self.proj(X.reshape(B, K, -1))


class MultiHorizonHead(nn.Module):
    def __init__(self, D, H, Nt, Ksc):
        super().__init__(); self.H, self.Nt, self.Ksc = H, Nt, Ksc; self.heads = nn.ModuleList([nn.Linear(D, Nt * Ksc * 2) for _ in range(H)])
    def forward(self, z):  # [B,D] -> [B,H,Nt,Ksc,2]
        return torch.stack([h(z) for h in self.heads], dim=1).reshape(z.shape[0], self.H, self.Nt, self.Ksc, 2)


class TransformerPredictor(nn.Module):
    def __init__(self, K, H, Nt, Ksc, D=512, L=6, heads=8, dropout=0.1, pos_emb=True, ffn_mult=4, residual=False):
        super().__init__(); self.residual = residual; self.tok = FrameTokenizer(Nt, Ksc, D); self.pos = nn.Parameter(torch.zeros(1, K, D)) if pos_emb else None
        layer = nn.TransformerEncoderLayer(D, heads, ffn_mult * D, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, L); self.norm = nn.LayerNorm(D); self.head = MultiHorizonHead(D, H, Nt, Ksc)
        if self.pos is not None: nn.init.normal_(self.pos, std=0.02)
    def forward(self, X):
        u = self.tok(X); u = u + self.pos if self.pos is not None else u
        z = self.norm(self.enc(u)[:, -1]); out = self.head(z)
        return out + X[:, -1:] if self.residual else out


class LSTMPredictor(nn.Module):
    def __init__(self, K, H, Nt, Ksc, D=512, layers=2, dropout=0.1, residual=False):
        super().__init__(); self.residual = residual; self.tok = FrameTokenizer(Nt, Ksc, D); self.lstm = nn.LSTM(D, D, num_layers=layers, batch_first=True, dropout=dropout if layers > 1 else 0.0)
        self.head = MultiHorizonHead(D, H, Nt, Ksc)
    def forward(self, X):
        out, _ = self.lstm(self.tok(X)); y = self.head(out[:, -1])
        return y + X[:, -1:] if self.residual else y


class ConvLSTMCell(nn.Module):
    def __init__(self, cin, chid, k=3):
        super().__init__(); self.chid = chid; self.conv = nn.Conv2d(cin + chid, 4 * chid, k, padding=k // 2)
    def forward(self, x, state):
        h, c = state; i, f, o, g = torch.chunk(self.conv(torch.cat([x, h], 1)), 4, 1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g; h = o * torch.tanh(c); return h, c
    def init_state(self, B, HW, device):
        z = torch.zeros(B, self.chid, *HW, device=device); return z, z.clone()


class ConvLSTMAE(nn.Module):
    """P04 ConvLSTM-AE: spatial encoder (Conv2D+BN+ReLU) -> 2-layer ConvLSTM encoder -> decoder ConvLSTM (init from encoder) autoregressive H steps -> TransConv decoder."""
    def __init__(self, K, H, Nt, Ksc, c_enc=32, c_hid=128, layers=2, k=3, teacher_forcing=True):
        super().__init__(); self.H, self.K = H, K; self.tf = teacher_forcing
        self.enc_sp = nn.Sequential(nn.Conv2d(2, c_enc, 3, padding=1), nn.BatchNorm2d(c_enc), nn.ReLU(), nn.Conv2d(c_enc, c_enc, 3, stride=2, padding=1), nn.BatchNorm2d(c_enc), nn.ReLU())
        self.enc_t = nn.ModuleList([ConvLSTMCell(c_enc if i == 0 else c_hid, c_hid, k) for i in range(layers)])
        self.dec_t = nn.ModuleList([ConvLSTMCell(c_enc if i == 0 else c_hid, c_hid, k) for i in range(layers)])
        self.dec_sp = nn.Sequential(nn.ConvTranspose2d(c_hid, c_enc, 4, stride=2, padding=1), nn.BatchNorm2d(c_enc), nn.ReLU(), nn.Conv2d(c_enc, 2, 3, padding=1))
    def _sp(self, x):  # [B,Nt,Ksc,2] -> [B,c,Nt/2,Ksc/2]
        return self.enc_sp(x.permute(0, 3, 1, 2))
    def forward(self, X, Y=None):
        B, K, Nt, Ksc, _ = X.shape; dev = X.device; HW = (Nt // 2, Ksc // 2)
        st = [c.init_state(B, HW, dev) for c in self.enc_t]
        for t in range(K):
            x = self._sp(X[:, t])
            for li, cell in enumerate(self.enc_t): st[li] = cell(x, st[li]); x = st[li][0]
        dst = st; inp = self._sp(X[:, -1]); outs = []
        for h in range(self.H):
            x = inp
            for li, cell in enumerate(self.dec_t): dst[li] = cell(x, dst[li]); x = dst[li][0]
            y = self.dec_sp(x).permute(0, 2, 3, 1)  # [B,Nt,Ksc,2]
            outs.append(y)
            nxt = Y[:, h] if (self.training and self.tf and Y is not None) else y.detach()
            inp = self._sp(nxt)
        return torch.stack(outs, 1)


def build_model(name, K, H, Nt, Ksc, **kw):
    if name == "transformer": return TransformerPredictor(K, H, Nt, Ksc, D=kw.get("D", 512), L=kw.get("L", 6), heads=kw.get("heads", 8), dropout=kw.get("dropout", 0.1), pos_emb=kw.get("pos_emb", True), residual=kw.get("residual", False))
    if name == "lstm": return LSTMPredictor(K, H, Nt, Ksc, D=kw.get("D", 512), layers=kw.get("L", 2), dropout=kw.get("dropout", 0.1), residual=kw.get("residual", False))
    if name == "convlstm_ae": return ConvLSTMAE(K, H, Nt, Ksc, c_enc=kw.get("c_enc", 32), c_hid=kw.get("c_hid", 128), layers=kw.get("L", 2), k=kw.get("k", 3))
    if name == "lwm11":
        from cp_lwm11 import LWM11Predictor; return LWM11Predictor(K, H, Nt, Ksc, init=kw.get("lwm11_init", "pretrained"), freeze=kw.get("lwm11_freeze", "none"), in_scale=kw.get("in_scale", 1.0), t_layers=kw.get("t_layers", 2), dropout=kw.get("dropout", 0.1), residual=kw.get("residual", False))
    if name.startswith("repo:"):
        from cp_repo_models import build_repo_model; return build_repo_model(name.split(":", 1)[1], K, H, Nt, Ksc, **kw)
    if name.startswith("mm:"):  # 멀티모달(채널 + RSU 센서) 실험 모델 (cp_multimodal.py, 2026-09-07). forward(X, sensors)
        from cp_multimodal import build_mm_model; return build_mm_model(name.split(":", 1)[1], K, H, Nt, Ksc, **kw)
    if name.startswith("mm2:"):  # 멀티모달 v2: 논문 [B] 식(20)(21) cross-modality attention 융합 (cp_multimodal_v2.py, 2026-09-20). forward(X, sensors)
        from cp_multimodal_v2 import build_mm2_model; return build_mm2_model(name.split(":", 1)[1], K, H, Nt, Ksc, **kw)
    raise ValueError(name)


def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
