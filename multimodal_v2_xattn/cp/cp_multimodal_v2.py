"""멀티모달 채널 예측 v2 `mm2:chiron` — 논문 [B] 식(20)(21) 형태의 Cross-Modality Attention 융합 (2026-09-20, 신규 파일).

v1(`mm:chiron`, `cp_multimodal.py`) 과의 차이는 **융합 방식 하나** 다. 백본(ChironChannelPredictor 원본)·센서 인코더(QueryPool/RadarCNN/PosMLP, v1 파일에서 import)·헤드·데이터 로더는 v1 그대로.

  v1: 센서 토큰을 프레임 안에서 concat → 채널 512 토큰 = Q, 센서 토큰 = K/V 인 게이트 cross-attention ×3 (센서끼리 상호작용 없음, 9/18 미팅 지적)
  v2: 프레임 k 마다 모달리티 토큰 U_k = [c_k ; s_k^cam ; s_k^lidar ; s_k^radar ; s_k^pos] (M 개, 각 1×D) 를 쌓고,
      프레임마다 다른 학습 질의 R_k 가 U_k 를 attend → 융합 토큰 f_k = MHA(R_k, U_k, U_k) (1×D)      ← [B] 식(20)(21) 그대로
      f_k 를 그 프레임의 패치 토큰 32개에 가산(broadcast) → chiron 블록 (fuse_where=input, 기본) 또는 헤드 직전 (fuse_where=output)

논문 [B] arXiv:2603.15093 §III-B: 식(20) U = Concat(U'_B, U'_L, U'_C) ∈ R^{P×M×d_m}, 식(21) B = CrossAttention(R, U, U), R ∈ R^{P×1×d_m}(스텝별 학습 질의).
재현 코드 `mmw_repro/models_faithful.py` (`paper_concat=True, paper_fuse_query=True`): mods [B·P, M, d_m], fuse_q [1,P,1,d_m], nn.MultiheadAttention 8 heads, 잔차·LN 없음.
이 파일의 `CrossModalityFusion` 은 그 재현 코드와 같은 연산이다(모달 수 M 과 스텝 수 P=K 만 다름). 소프트맥스 가중치가 곧 "프레임 k 에서 어느 모달을 얼마나 믿는가" 다.

[AI 결정] — 논문 [B] 에 없는, 채널 예측(chiron)에 맞추기 위한 선택. 근거·대안은 README §4.
  (1) 채널 토큰 c_k = 프레임 k 의 패치 토큰 32개 평균(mean pooling). [B] 의 빔 토큰(1D conv 출력)에 해당하는 "프레임당 벡터 1개" 가 필요해서.
  (2) 융합 토큰 f_k 주입 = 프레임 k 의 패치 토큰 32개 전부에 가산(broadcast). chiron 은 프레임을 32 토큰으로 표현하므로 [B] 처럼 토큰을 "치환" 하면 공간 구조가 사라진다.
  (3) 주입 위치 fuse_where: `input`(기본) = 패치 임베딩 직후·chiron 블록 앞 = [B] 처럼 시퀀스 모델 **앞**에서 융합 / `output` = 백본 뒤·헤드 앞 = v1 의 융합 위치.
  (4) LiDAR 토큰 1개(v1 은 16개). [B] 식(14)~(17) 은 LiDAR 를 학습 질의 1개로 벡터 하나 u_L 로 요약한다. (생성자 lidar_tokens 는 프로브 대조용, CLI 노출 없음)
  (5) fuse_zero_init(기본 True): 융합 MHA 의 out_proj 를 0 초기화 → 학습 시작 시 f_k = 0 → v2 = 채널 전용 chiron 과 같은 함수(v1 프로토콜과 같은 출발점, 센서 증분을 같은 시작점에서 잼).
      [B] 에는 없는 초기화이며 `--no_fuse_zero_init` 로 끈다.
  (6) 모달리티 임베딩 없음·프레임별 질의 R_k: 재현 코드의 paper_concat / paper_fuse_query 와 동일.
  (7) 없는 모달은 U 에서 빠짐(M 가변, [B] "존재 모달만"). 센서가 하나도 없으면 융합을 건너뛴다(= 채널 전용 chiron).
  (8) pos 는 token 모드만(U 의 한 모달). v1 의 broadcast 모드는 v2 에 없음.

인터페이스: forward(X [B,K,Nt,Ksc,2], sensors: dict|None, diag=False) -> [B,H,Nt,Ksc,2]. sensors 키·shape 는 v1 과 같다:
  cam [B,K,196,768] fp16   lidar [B,K,64,384] fp16   radar [B,K,2,64,64] fp16   pos [B,K,10] fp32
진단: attention_over_modalities() = 직전 forward 의 식(21) 소프트맥스 가중치(모달별 평균·프레임별 [K,M]), param_groups(). v1 호환용 gate_means()=[] / attention_mass().
"""
import os, sys, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path: sys.path.insert(0, _p)
from cp_multimodal import QueryPool, RadarCNN, PosMLP, MODALS, CAM_DIM, LIDAR_DIM      # v1 센서 인코더 재사용(무수정)
from models.chiron_channel import ChironChannelPredictor                                # 백본 원본(이 폴더 models/ 사본 = multimodal_code_index 와 동일 md5)


class CrossModalityFusion(nn.Module):
    """[B] 식(21): 프레임별 학습 질의 R_k ∈ R^{1×D} 가 모달리티 토큰 U_k ∈ R^{M×D} 를 attend → f_k ∈ R^{1×D}. 잔차·LN·FFN 없음(논문·재현 코드와 동일).
    입력 U [B,K,M,D] → 출력 f [B,K,D]. 소프트맥스 가중치(head 평균) [B,K,M] 을 last_attn 에 저장."""
    def __init__(self, K, D, heads=8, dropout=0.0, zero_init=True):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, K, 1, D) * 0.02)                       # R ∈ R^{K×1×D}: 재현 코드 fuse_q 와 같은 초기화(randn·0.02)
        self.attn = nn.MultiheadAttention(D, heads, dropout=dropout, batch_first=True)   # 재현 코드 fuse_attn 과 같은 모듈(8 heads)
        if zero_init:                                                                   # [AI 결정 5] out_proj = 0 → 초기 f_k = 0 → 채널 전용과 같은 출발점
            nn.init.zeros_(self.attn.out_proj.weight); nn.init.zeros_(self.attn.out_proj.bias)
        self.last_attn = None

    def forward(self, U):
        B, K, M, D = U.shape
        q = self.query[:, :K].expand(B, -1, -1, -1).reshape(B * K, 1, D)               # 프레임 k 마다 R_k
        kv = U.reshape(B * K, M, D)
        f, w = self.attn(q, kv, kv, need_weights=True, average_attn_weights=True)       # f [B·K,1,D], w [B·K,1,M]
        self.last_attn = w.detach().reshape(B, K, M)
        return f.reshape(B, K, D)


class MMChironX(nn.Module):
    def __init__(self, K, H, Nt, Ksc, sensors=("cam", "lidar", "radar", "pos"), D=256, L=6, heads=4, dropout=0.1,
                 fuse_where="input", fuse_heads=8, fuse_zero_init=True, include_channel_token=True, backbone_init="", lidar_tokens=1):
        super().__init__(); assert fuse_where in ("input", "output"), fuse_where
        self.K, self.H, self.Nt, self.Ksc, self.D = K, H, Nt, Ksc, D
        self.fuse_where, self.include_channel_token = fuse_where, include_channel_token
        self.sensors = [s for s in (sensors.split(",") if isinstance(sensors, str) else sensors) if s]; bad = [s for s in self.sensors if s not in MODALS]; assert not bad, bad
        # 백본 = v1·S1 s1_chiron_lr3e-4 와 같은 하이퍼파라미터(D 256, L 6, heads 4, patch 4×32). 클래스 원본 무수정.
        self.backbone = ChironChannelPredictor(num_bs_antennas=Nt, num_subcarriers=Ksc, embed_dim=D, depth=L, num_heads=heads, history_len=K, prediction_horizon=H, patch_h=4, patch_w=32, dropout=dropout)
        self.S = self.backbone._num_spatial
        if backbone_init: self.load_backbone(backbone_init)
        # 센서 인코더 = v1 과 동일 클래스·하이퍼파라미터(QueryPool heads 4). LiDAR 질의 수만 기본 1 ([AI 결정 4]).
        self.enc = nn.ModuleDict(); self.n_tok = {}
        for m in self.sensors:
            if m == "cam": self.enc[m] = QueryPool(CAM_DIM, D, 1, 4, dropout); self.n_tok[m] = 1
            elif m == "lidar": self.enc[m] = QueryPool(LIDAR_DIM, D, lidar_tokens, 4, dropout); self.n_tok[m] = lidar_tokens
            elif m == "radar": self.enc[m] = RadarCNN(D); self.n_tok[m] = 1
            elif m == "pos": self.enc[m] = PosMLP(D); self.n_tok[m] = 1
        self.token_mods = list(self.sensors)
        # 융합 = [B] 식(21) 1층. 모달 임베딩·프레임 임베딩 없음([AI 결정 6]; 프레임 정보는 질의 R_k 가 프레임마다 달라 담긴다).
        self.fusion = nn.ModuleList([CrossModalityFusion(K, D, fuse_heads, 0.0, fuse_zero_init)]) if self.sensors else nn.ModuleList()
        self._last_mods = []

    # ---- 백본 가중치(v1 과 동일) ----
    def load_backbone(self, ckpt):
        sd = torch.load(ckpt, map_location="cpu", weights_only=False); sd = sd.get("model", sd)
        sd = {(k[2:] if k.startswith("m.") else k): v for k, v in sd.items()}          # RepoWrap 접두 'm.' 제거
        res = self.backbone.load_state_dict(sd, strict=True); self._backbone_init = ckpt; return res

    # ---- 센서 토큰: 프레임별 [B,K,n_s,D] (모달 임베딩 없음 = [B] 식 20) ----
    def _sensor_tokens(self, sensors):
        toks, mods = [], []
        for m in self.token_mods:
            if sensors.get(m) is None: continue
            toks.append(self.enc[m](sensors[m])); mods.append(m)
        return (torch.cat(toks, 2) if toks else None), mods

    # ---- 융합: 패치 토큰 t [B,K,S,D] → t + f_k (broadcast) ----
    def _fuse(self, t, sensors):
        st, mods = self._sensor_tokens(sensors); self._last_mods = mods
        if st is None or not len(self.fusion): return t                                 # 센서 없음 → 채널 전용 경로
        parts = ([t.mean(2, keepdim=True)] if self.include_channel_token else []) + [st]  # [AI 결정 1] c_k = 32 패치 평균
        U = torch.cat(parts, 2)                                                          # [B,K,M,D], M = (1) + Σ n_tok   ← 식(20)
        f = self.fusion[0](U)                                                            # [B,K,D]                        ← 식(21)
        return t + f[:, :, None, :]                                                      # [AI 결정 2] 프레임 k 의 32 패치에 가산

    def forward(self, X, sensors=None, diag=False):
        sensors = sensors or {}; bb = self.backbone; B, K, Na, Nsc, _ = X.shape; S = self.S
        t = bb.patch_embed(X.reshape(B * K, Na, Nsc, 2)).view(B, K, S, self.D) + bb.temporal_pos[:, :K] + bb.spatial_pos   # encode_tokens 와 같은 순서
        if self.fuse_where == "input": t = self._fuse(t, sensors)                        # [AI 결정 3] 시퀀스 모델 앞([B] 위치)
        t = t.reshape(B, K * S, self.D)
        for blk in bb.blocks: t = blk(t, K, S)
        t = bb.final_norm(t)                                                             # 헤드 직전 토큰 [B,K·S,D]
        if self.fuse_where == "output": t = self._fuse(t.view(B, K, S, self.D), sensors).reshape(B, K * S, self.D)   # v1 위치
        out = bb.head(t)
        return out.reshape(B, self.H, self.Nt, self.Ksc, 2)

    # ---- 진단 ----
    def attention_over_modalities(self):
        """직전 forward 의 식(21) 소프트맥스 가중치. per_modal = 모달별 평균(B·K 평균, 합 1), per_frame = [K, M] (열 이름 columns)."""
        blk = self.fusion[0] if len(self.fusion) else None
        if blk is None or blk.last_attn is None or not self._last_mods: return None
        w = blk.last_attn.float().mean(0)                                                # [K, M]
        cols = (["channel"] if self.include_channel_token else []) + [m for m in self._last_mods for _ in range(self.n_tok[m])]
        d = {}
        for j, n in enumerate(cols): d[n] = d.get(n, 0.0) + float(w[:, j].mean())
        return dict(per_modal=d, per_frame=w.cpu().tolist(), columns=cols)

    def gate_means(self): return []                                                      # v1 호환: 게이트 없음
    def attention_mass(self):                                                            # v1 호환: 층 1개, 모달별 가중치
        a = self.attention_over_modalities(); return [a["per_modal"]] if a else []

    def param_groups(self):
        c = lambda mod: sum(p.numel() for p in mod.parameters() if p.requires_grad)
        g = dict(backbone_no_head=c(self.backbone) - c(self.backbone.head), head=c(self.backbone.head), fusion=c(self.fusion), embeddings=0)
        for m in self.sensors: g[f"enc_{m}"] = c(self.enc[m])
        g["sensor_encoders"] = sum(g[f"enc_{m}"] for m in self.sensors); g["total"] = c(self); return g


def build_mm2_model(name, K, H, Nt, Ksc, **kw):
    if name != "chiron": raise ValueError(f"mm2:{name} (지원: mm2:chiron)")
    if kw.get("pos_mode", "token") != "token": raise ValueError("mm2:chiron 은 pos_mode=token 만 지원(pos 는 U 의 한 모달)")
    # 백본 하이퍼파라미터는 v1·S1 과 같게 고정(D 256, L 6, heads 4) — train_cp.py 의 --D/--L/--heads 는 무시된다(--backbone_init 호환)
    return MMChironX(K, H, Nt, Ksc, sensors=kw.get("sensors", "cam,lidar,radar,pos"), D=256, L=6, heads=4, dropout=kw.get("dropout", 0.1),
                     fuse_where=kw.get("fuse_where", "input"), fuse_heads=kw.get("fuse_heads", 8), fuse_zero_init=kw.get("fuse_zero_init", True),
                     include_channel_token=not kw.get("fuse_no_channel_token", False), backbone_init=kw.get("backbone_init", ""), lidar_tokens=kw.get("lidar_tokens", 1))
