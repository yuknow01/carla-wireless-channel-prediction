"""실험용 멀티모달 채널 예측 모델 `mm:chiron` (2026-09-07, 신규 파일). 설계 = MULTIMODAL_ARCHITECTURES.md §5.1 안 (a) + §5.2 안 (b)-(i).
인터페이스: forward(X [B,K,Nt,Ksc,2], sensors: dict|None) -> [B,H,Nt,Ksc,2].  sensors 키(선택, 없는 센서는 건너뜀):
  cam   [B,K,196,768] fp16(캐시 그대로; 입구에서 float)   lidar [B,K,64,384] fp16   radar [B,K,2,64,64] fp16   pos [B,K,10] fp32
구조:
  백본  = models/chiron_channel.py::ChironChannelPredictor 를 그대로 인스턴스화(D 256, L 6, heads 4, patch 4×32; S1 s1_chiron_lr3e-4 와 동일 하이퍼파라미터).
          encode_tokens() 대신 그 서브모듈(patch_embed·temporal_pos·spatial_pos·blocks·final_norm)을 같은 순서로 호출해 **헤드 직전 토큰 [B,512,256]** 을 꺼내고
          (pos_mode=broadcast 면 위치 인코딩을 프레임별 32 패치에 가산), 융합 후 같은 backbone.head(ChannelPredictionHead)에 넣는다. 클래스 복사·수정 없음.
  센서 인코더(프레임별) = cam: Linear 768→256 + 학습 질의 1개 MHA(196 패치 → 1 토큰) · lidar: Linear 384→256 + 학습 질의 16개(64 셀 → 16 토큰)
          · radar: 소형 CNN(2→32→64→64, stride 2 ×3, GAP) → Linear 64→256 → 1 토큰 · pos: MLP 10→256→256 → 1 토큰.  + 모달 임베딩 + 프레임 임베딩(16) → [B, 16·n_tok, 256].
  융합  = GatedXAttnBlock × fuse_layers(기본 3): pre-norm cross-attention(q = 채널 토큰, kv = 센서 토큰), x ← x + g ⊙ attn, g = σ(Linear([x; attn])) 게이트 Linear **0 초기화**(g = 0.5),
          attn.out_proj·FFN(w3) 도 0 초기화 → 학습 시작 시 블록 = 항등 = 채널 전용 chiron 과 같은 함수(13 프로브 max|diff| 로 확인). FFN = chiron GatedFFN(SwiGLU) 재사용.
  causal_mask = 채널 프레임 k(패치 32개) 가 센서 프레임 ≤ k 만 보도록 attn_mask [512, 16·n_tok].
옵션: --backbone_init <ckpt>(S1 chiron best.pt, RepoWrap 'm.' 접두 제거 후 strict 로드), --pos_mode token|broadcast, --sensors, --fuse_layers, --causal_mask.
진단: gate_means()(층별 게이트 평균), attention_mass()(층별·모달별 어텐션 질량; diag=True 로 forward 했을 때), param_groups()(파라미터 분해).
"""
import os, sys, math, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get("CP_REPO", "/mnt/ssd_7t_2/carla-wireless-dataset/multimodal_code_index")
if REPO not in sys.path: sys.path.insert(0, REPO)
from models.chiron_channel import ChironChannelPredictor, GatedFFN   # 무수정 원본

MODALS = ("cam", "lidar", "radar", "pos")
CAM_DIM, LIDAR_DIM, RADAR_CH, POS_DIM = 768, 384, 2, 10
POS_SCALE = torch.tensor([100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0])   # m → /100, m/s → /10 (입력 스케일 정규화, 고정 상수)


def _trunc(p): nn.init.trunc_normal_(p, std=0.02)


class QueryPool(nn.Module):
    """Linear(in→D) + LayerNorm + 학습 질의 n_q 개 MHA: [B,K,N,in] → [B,K,n_q,D]"""
    def __init__(self, in_dim, D, n_q, heads=4, dropout=0.1):
        super().__init__(); self.proj = nn.Linear(in_dim, D); self.norm = nn.LayerNorm(D); self.query = nn.Parameter(torch.zeros(1, n_q, D)); _trunc(self.query)
        self.attn = nn.MultiheadAttention(D, heads, dropout=dropout, batch_first=True); self.n_q = n_q
    def forward(self, x):
        B, K, N, C = x.shape; h = self.norm(self.proj(x.float().reshape(B * K, N, C)))
        out, _ = self.attn(self.query.expand(B * K, -1, -1), h, h)
        return out.view(B, K, self.n_q, -1)


class RadarCNN(nn.Module):
    """[B,K,2,64,64] (ch0 count → log1p, ch1 mean velocity m/s) → [B,K,1,D]"""
    def __init__(self, D, ch=(32, 64, 64)):
        super().__init__()
        layers, c_in = [], RADAR_CH
        for c in ch: layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.GELU()]; c_in = c
        self.cnn = nn.Sequential(*layers); self.out = nn.Linear(c_in, D)
    def forward(self, x):
        B, K = x.shape[:2]; x = x.float().reshape(B * K, RADAR_CH, x.shape[-2], x.shape[-1]).clone(); x[:, 0] = torch.log1p(x[:, 0])
        return self.out(self.cnn(x).mean(dim=(2, 3))).view(B, K, 1, -1)


class PosMLP(nn.Module):
    """[B,K,10] → [B,K,1,D]; broadcast 모드에서 항등 시작을 위해 마지막 Linear 0 초기화(zero_last)."""
    def __init__(self, D, zero_last=False):
        super().__init__(); self.register_buffer("scale", POS_SCALE.clone()); self.mlp = nn.Sequential(nn.Linear(POS_DIM, D), nn.GELU(), nn.Linear(D, D))
        if zero_last: nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)
    def forward(self, x): return self.mlp(x.float() / self.scale).unsqueeze(2)


class GatedXAttnBlock(nn.Module):
    """pre-norm 게이트 cross-attention + SwiGLU FFN. 0 초기화(gate Linear, attn.out_proj, ffn.w3) → 초기 = 항등."""
    def __init__(self, D, heads=4, dropout=0.1):
        super().__init__(); self.q_norm = nn.LayerNorm(D); self.kv_norm = nn.LayerNorm(D)
        self.attn = nn.MultiheadAttention(D, heads, dropout=dropout, batch_first=True); self.attn_drop = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * D, D); self.ffn = GatedFFN(embed_dim=D, mlp_ratio=4.0, dropout=dropout)
        for p in (self.gate.weight, self.gate.bias, self.attn.out_proj.weight, self.attn.out_proj.bias, self.ffn.w3.weight, self.ffn.w3.bias): nn.init.zeros_(p)
        self.last_gate_mean = None; self.last_attn = None
    def forward(self, x, kv, attn_mask=None, diag=False):
        q = self.q_norm(x); k = self.kv_norm(kv)
        a, w = self.attn(q, k, k, attn_mask=attn_mask, need_weights=diag, average_attn_weights=True)
        a = self.attn_drop(a); g = torch.sigmoid(self.gate(torch.cat([x, a], -1)))
        self.last_gate_mean = g.detach().mean(); self.last_attn = w.detach() if (diag and w is not None) else None
        return self.ffn(x + g * a)


class MMChiron(nn.Module):
    def __init__(self, K, H, Nt, Ksc, sensors=("cam", "lidar", "radar", "pos"), D=256, L=6, heads=4, dropout=0.1, fuse_layers=3, fuse_heads=4,
                 causal_mask=False, pos_mode="token", backbone_init="", lidar_tokens=16):
        super().__init__(); assert pos_mode in ("token", "broadcast"), pos_mode
        self.K, self.H, self.Nt, self.Ksc, self.D, self.pos_mode, self.causal = K, H, Nt, Ksc, D, pos_mode, causal_mask
        self.sensors = [s for s in (sensors.split(",") if isinstance(sensors, str) else sensors) if s]; bad = [s for s in self.sensors if s not in MODALS]; assert not bad, bad
        self.backbone = ChironChannelPredictor(num_bs_antennas=Nt, num_subcarriers=Ksc, embed_dim=D, depth=L, num_heads=heads, history_len=K, prediction_horizon=H, patch_h=4, patch_w=32, dropout=dropout)
        self.S = self.backbone._num_spatial
        if backbone_init: self.load_backbone(backbone_init)
        self.enc = nn.ModuleDict(); self.n_tok = {}
        for m in self.sensors:
            if m == "cam": self.enc[m] = QueryPool(CAM_DIM, D, 1, fuse_heads, dropout); self.n_tok[m] = 1
            elif m == "lidar": self.enc[m] = QueryPool(LIDAR_DIM, D, lidar_tokens, fuse_heads, dropout); self.n_tok[m] = lidar_tokens
            elif m == "radar": self.enc[m] = RadarCNN(D); self.n_tok[m] = 1
            elif m == "pos": self.enc[m] = PosMLP(D, zero_last=(pos_mode == "broadcast")); self.n_tok[m] = 0 if pos_mode == "broadcast" else 1
        self.token_mods = [m for m in self.sensors if self.n_tok[m] > 0]
        self.modal_emb = nn.Parameter(torch.zeros(len(MODALS), D)); self.frame_emb = nn.Parameter(torch.zeros(K, D)); _trunc(self.modal_emb); _trunc(self.frame_emb)
        self.fusion = nn.ModuleList([GatedXAttnBlock(D, fuse_heads, dropout) for _ in range(fuse_layers)]) if self.token_mods else nn.ModuleList()
        self._mask_cache = {}

    # ---- 백본 가중치 ----
    def load_backbone(self, ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
        sd = {(k[2:] if k.startswith("m.") else k): v for k, v in sd.items()}          # RepoWrap 접두 'm.' 제거
        res = self.backbone.load_state_dict(sd, strict=True); self._backbone_init = ckpt; return res

    # ---- 채널 인코딩: ChironChannelPredictor.encode_tokens 와 같은 순서로 서브모듈 호출(+ 선택적 프레임별 가산) ----
    def encode_channel(self, X, add=None):
        bb = self.backbone; B, K, Na, Nsc, _ = X.shape; S = self.S
        t = bb.patch_embed(X.reshape(B * K, Na, Nsc, 2)).view(B, K, S, self.D)
        t = t + bb.temporal_pos[:, :K] + bb.spatial_pos
        if add is not None: t = t + add[:, :, None, :]                                    # broadcast: [B,K,D] → 32 패치 전부에 가산
        t = t.reshape(B, K * S, self.D)
        for blk in bb.blocks: t = blk(t, K, S)
        return bb.final_norm(t)                                                            # [B, K·S, D] = 헤드 직전 토큰

    # ---- 센서 토큰 ----
    def encode_sensors(self, sensors):
        toks, mods = [], []
        for m in self.token_mods:
            if m not in sensors or sensors[m] is None: continue
            t = self.enc[m](sensors[m]) + self.modal_emb[MODALS.index(m)]                   # [B,K,n_m,D]
            toks.append(t); mods.append(m)
        if not toks: return None, []
        t = torch.cat(toks, 2); B, K, n_tot, D = t.shape
        t = t + self.frame_emb[:K, None, :]
        return t.reshape(B, K * n_tot, D), mods                                            # 프레임-major 순서: 토큰 j → 프레임 j // n_tot

    def _mask(self, K, n_tot, device):
        key = (K, n_tot, str(device))
        if key not in self._mask_cache:
            fc = torch.arange(K * self.S, device=device) // self.S; fs = torch.arange(K * n_tot, device=device) // n_tot
            self._mask_cache[key] = fs[None, :] > fc[:, None]                             # True = 차단(센서 프레임이 채널 프레임보다 미래)
        return self._mask_cache[key]

    def forward(self, X, sensors=None, diag=False):
        sensors = sensors or {}; B, K = X.shape[:2]
        add = self.enc["pos"](sensors["pos"]).squeeze(2) if (self.pos_mode == "broadcast" and "pos" in self.enc and sensors.get("pos") is not None) else None
        tok = self.encode_channel(X, add)
        st, mods = self.encode_sensors(sensors); self._last_mods = mods
        if st is not None:
            n_tot = st.shape[1] // K; mask = self._mask(K, n_tot, X.device) if self.causal else None
            for blk in self.fusion: tok = blk(tok, st, attn_mask=mask, diag=diag)
        out = self.backbone.head(tok)
        return out.reshape(B, self.H, self.Nt, self.Ksc, 2)

    # ---- 진단 ----
    def gate_means(self): return [float(b.last_gate_mean) if b.last_gate_mean is not None else None for b in self.fusion]

    def attention_mass(self):
        """직전 forward(diag=True) 의 층별 어텐션 [B,512,K·n_tot] 을 모달별로 합산(평균 over B·query). 반환 [{mod: mass}] per layer."""
        mods = getattr(self, "_last_mods", []); out = []
        if not mods: return out
        n = [self.n_tok[m] for m in mods]; n_tot = sum(n)
        for b in self.fusion:
            if b.last_attn is None: out.append(None); continue
            w = b.last_attn.mean(dim=(0, 1))                                               # [K·n_tot]
            w = w.view(self.K, n_tot); off = 0; d = {}
            for m, nm in zip(mods, n): d[m] = float(w[:, off:off + nm].sum()); off += nm
            out.append(d)
        return out

    def param_groups(self):
        c = lambda mod: sum(p.numel() for p in mod.parameters() if p.requires_grad)
        g = dict(backbone_no_head=c(self.backbone) - c(self.backbone.head), head=c(self.backbone.head), fusion=c(self.fusion), embeddings=int(self.modal_emb.numel() + self.frame_emb.numel()))
        for m in self.sensors: g[f"enc_{m}"] = c(self.enc[m])
        g["sensor_encoders"] = sum(g[f"enc_{m}"] for m in self.sensors); g["total"] = c(self); return g


def build_mm_model(name, K, H, Nt, Ksc, **kw):
    if name != "chiron": raise ValueError(f"mm:{name} (지원: mm:chiron)")
    # 백본 하이퍼파라미터는 S1 s1_chiron_lr3e-4(D 256, L 6, heads 4, patch 4×32)로 고정 — train_cp.py 의 --D(기본 512)/--L/--heads 는 mm:chiron 에서 무시된다(--backbone_init 호환 목적)
    return MMChiron(K, H, Nt, Ksc, sensors=kw.get("sensors", "cam,lidar,radar,pos"), D=256, L=6, heads=4, dropout=kw.get("dropout", 0.1),
                    fuse_layers=kw.get("fuse_layers", 3), causal_mask=kw.get("causal_mask", False), pos_mode=kw.get("pos_mode", "token"), backbone_init=kw.get("backbone_init", ""))
