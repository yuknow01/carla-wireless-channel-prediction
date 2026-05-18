from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

# -----------------------------------------------------------------------------
# Tokenization
# -----------------------------------------------------------------------------
class ComplexPatchTokenizer:
    def __init__(self, phase_mode: str = "real_imag") -> None:
        if phase_mode not in {"real_imag", "mag_phase"}:
            raise ValueError("phase_mode must be 'real_imag' or 'mag_phase'")
        self.phase_mode = phase_mode

    def _split_channels(self, tensor: Tensor) -> Tensor:
        if self.phase_mode == "real_imag":
            real = tensor.real.unsqueeze(-1)
            imag = tensor.imag.unsqueeze(-1)
            return torch.cat([real, imag], dim=-1)
        magnitude = tensor.abs().unsqueeze(-1)
        phase = torch.angle(tensor).unsqueeze(-1)
        return torch.cat([magnitude, phase], dim=-1)

    def __call__(self, seq: Tensor, patch_size: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
        if not torch.is_complex(seq):
            raise TypeError("expected complex tensor shaped (B, T, N, M)")
        ph, pw = patch_size
        if seq.size(2) % ph != 0 or seq.size(3) % pw != 0:
            raise ValueError("patch_size must evenly divide channel dimensions")
        channels = self._split_channels(seq)
        b, t, n, m, c = channels.shape
        h = n // ph
        w = m // pw
        channels = channels.view(b, t, h, ph, w, pw, c)
        channels = channels.permute(0, 1, 2, 4, 3, 5, 6).contiguous()
        tokens = channels.view(b, t * h * w, ph * pw * c)
        mask = torch.zeros((b, tokens.size(1)), dtype=torch.bool, device=tokens.device)
        return tokens, mask


# -----------------------------------------------------------------------------
# Sparse spatio-temporal attention
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class AttentionCacheKey:
    temporal: int
    height: int
    width: int
    same_frame_window: int
    same_frame_window_h: Optional[int]
    same_frame_window_w: Optional[int]
    same_frame_dilation_h: int
    same_frame_dilation_w: int
    temporal_offsets: Tuple[int, ...]
    temporal_spatial_window: int
    temporal_spatial_window_h: Optional[int]
    temporal_spatial_window_w: Optional[int]
    temporal_spatial_dilation_h: int
    temporal_spatial_dilation_w: int
    temporal_drift_h: int
    temporal_drift_w: int
    include_cls: bool


class NeighborIndexer:
    def __init__(self) -> None:
        self._cache: Dict[Tuple[int, int, int, AttentionCacheKey], Tensor] = {}

    def get(self, T: int, H: int, W: int, include_cls: bool, config: "LWMConfig", device: torch.device) -> Tensor:
        key = (
            T,
            H,
            W,
            AttentionCacheKey(
                temporal=T,
                height=H,
                width=W,
                same_frame_window=config.same_frame_window,
                same_frame_window_h=config.same_frame_window_h,
                same_frame_window_w=config.same_frame_window_w,
                same_frame_dilation_h=config.same_frame_dilation_h,
                same_frame_dilation_w=config.same_frame_dilation_w,
                temporal_offsets=config.temporal_offsets,
                temporal_spatial_window=config.temporal_spatial_window,
                temporal_spatial_window_h=config.temporal_spatial_window_h,
                temporal_spatial_window_w=config.temporal_spatial_window_w,
                temporal_spatial_dilation_h=config.temporal_spatial_dilation_h,
                temporal_spatial_dilation_w=config.temporal_spatial_dilation_w,
                temporal_drift_h=config.temporal_drift_h,
                temporal_drift_w=config.temporal_drift_w,
                include_cls=include_cls,
            ),
        )
        if key in self._cache:
            tensor = self._cache[key]
            return tensor if tensor.device == device else tensor.to(device)
        indices = self._build_indices(T, H, W, include_cls, config)
        if indices:
            max_len = max(len(neighbors) for neighbors in indices)
            if any(len(neighbors) != max_len for neighbors in indices):
                padded = []
                for neighbors in indices:
                    if len(neighbors) < max_len:
                        neighbors = neighbors + [-1] * (max_len - len(neighbors))
                    padded.append(neighbors)
                indices = padded
        tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        self._cache[key] = tensor
        return tensor

    def _build_indices(self, T: int, H: int, W: int, include_cls: bool, config: "LWMConfig") -> List[List[int]]:
        neighbors: List[List[int]] = []
        same_h = config.same_frame_window if config.same_frame_window_h is None else config.same_frame_window_h
        same_w = config.same_frame_window if config.same_frame_window_w is None else config.same_frame_window_w

        def frame_base(frame: int) -> int:
            return frame * H * W

        for t_idx in range(T):
            base = frame_base(t_idx)
            for h_idx in range(H):
                for w_idx in range(W):
                    current = base + h_idx * W + w_idx
                    local: List[int] = []
                    if config.same_frame_window < 0:
                        local.extend(range(base, base + H * W))
                    else:
                        for dh in range(-same_h, same_h + 1, config.same_frame_dilation_h):
                            for dw in range(-same_w, same_w + 1, config.same_frame_dilation_w):
                                nh = h_idx + dh
                                nw = w_idx + dw
                                if 0 <= nh < H and 0 <= nw < W:
                                    local.append(base + nh * W + nw)
                    if not config.spatial_only:
                        for dt in config.temporal_offsets:
                            other_t = t_idx + dt
                            if other_t < 0 or other_t >= T:
                                continue
                            other_base = frame_base(other_t)
                            drift_h = config.temporal_spatial_window if config.temporal_drift_h == 0 else min(config.temporal_spatial_window, abs(dt) * config.temporal_drift_h)
                            drift_w = config.temporal_spatial_window if config.temporal_drift_w == 0 else min(config.temporal_spatial_window, abs(dt) * config.temporal_drift_w)
                            window_h = config.temporal_spatial_window if config.temporal_spatial_window_h is None else config.temporal_spatial_window_h
                            window_w = config.temporal_spatial_window if config.temporal_spatial_window_w is None else config.temporal_spatial_window_w
                            for dh in range(-min(window_h, drift_h), min(window_h, drift_h) + 1, config.temporal_spatial_dilation_h):
                                for dw in range(-min(window_w, drift_w), min(window_w, drift_w) + 1, config.temporal_spatial_dilation_w):
                                    nh = max(0, min(H - 1, h_idx + dh))
                                    nw = max(0, min(W - 1, w_idx + dw))
                                    local.append(other_base + nh * W + nw)
                    if include_cls:
                        local.append(T * H * W)
                    if not local:
                        local.append(current)
                    neighbors.append(sorted(set(local)))
        if include_cls:
            neighbors.append(list(range(T * H * W)))
        return neighbors


class SparseSpatioTemporalAttention(nn.Module):
    def __init__(self, config: "LWMConfig", embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.indexer = NeighborIndexer()

    def _apply_rope(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated_first = x1 * cos - x2 * sin
        rotated_second = x1 * sin + x2 * cos
        return torch.stack([rotated_first, rotated_second], dim=-1).flatten(-2)

    def _rope_factors(self, S: int, device: torch.device) -> Tuple[Tensor, Tensor]:
        half = self.head_dim // 2
        inv_freq = 1.0 / (self.config.rope_base ** (torch.arange(0, half, dtype=torch.float32, device=device) / max(1, half)))
        positions = torch.arange(S, dtype=torch.float32, device=device)
        angles = positions[:, None] * inv_freq[None, :]
        return torch.cos(angles)[None, None, :, :], torch.sin(angles)[None, None, :, :]

    def forward(self, hidden_states: Tensor, T: int, H: int, W: int, include_cls: bool) -> Tensor:
        bsz, seq_len, _ = hidden_states.shape
        neighbors = self.indexer.get(T, H, W, include_cls, self.config, hidden_states.device)
        qkv = self.qkv(hidden_states)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.config.posenc == "rope_sincos":
            cos, sin = self._rope_factors(seq_len, hidden_states.device)
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)

        gather_idx = neighbors.clamp_min(0)
        valid_mask = neighbors >= 0
        k = k[:, :, gather_idx, :]
        v = v[:, :, gather_idx, :]
        scores = torch.einsum("bhqd,bhqkd->bhqk", q, k) * self.scale
        scores = scores.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if self.config.routing_topk_enable:
            K = scores.size(-1)
            keep = min(self.config.routing_topk_max, max(self.config.routing_topk_min, int(self.config.routing_topk_fraction * K)))
            if self.config.routing_topk_per_head:
                _, idx = torch.topk(scores, keep, dim=-1)
                topk_mask = torch.zeros_like(scores, dtype=torch.bool)
                topk_mask.scatter_(-1, idx, True)
            else:
                avg_scores = scores.mean(dim=1, keepdim=True)
                _, idx = torch.topk(avg_scores, keep, dim=-1)
                topk_mask = torch.zeros_like(scores, dtype=torch.bool)
                topk_mask.scatter_(-1, idx.expand_as(scores), True)
            scores = scores.masked_fill(~topk_mask, float("-inf"))
        elif self.config.topk_neighbors is not None:
            keep = min(self.config.topk_neighbors, scores.size(-1))
            if self.config.topk_per_head:
                _, idx = torch.topk(scores, keep, dim=-1)
                topk_mask = torch.zeros_like(scores, dtype=torch.bool)
                topk_mask.scatter_(-1, idx, True)
            else:
                avg_scores = scores.mean(dim=1, keepdim=True)
                _, idx = torch.topk(avg_scores, keep, dim=-1)
                topk_mask = torch.zeros_like(scores, dtype=torch.bool)
                topk_mask.scatter_(-1, idx.expand_as(scores), True)
            scores = scores.masked_fill(~topk_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = attn.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(0), 0.0)
        context = torch.einsum("bhqk,bhqkd->bhqd", attn, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.proj(context)


class LWMEncoderLayer(nn.Module):
    def __init__(self, config: "LWMConfig") -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = SparseSpatioTemporalAttention(config, config.embed_dim, config.num_heads)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, config.embed_dim),
        )

    def forward(self, x: Tensor, T: int, H: int, W: int, include_cls: bool) -> Tensor:
        x = x + self.attn(self.norm1(x), T, H, W, include_cls)
        x = x + self.mlp(self.norm2(x))
        return x


class LWMEncoder(nn.Module):
    def __init__(self, config: "LWMConfig") -> None:
        super().__init__()
        self.layers = nn.ModuleList([LWMEncoderLayer(config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.embed_dim)

    def forward(self, x: Tensor, T: int, H: int, W: int, include_cls: bool) -> Tensor:
        for layer in self.layers:
            x = layer(x, T, H, W, include_cls)
        return self.norm(x)


# -----------------------------------------------------------------------------
# Config and main model
# -----------------------------------------------------------------------------
@dataclass
class LWMConfig:
    patch_size: Tuple[int, int] = (1, 1)
    phase_mode: str = "real_imag"
    embed_dim: int = 32
    depth: int = 12
    num_heads: int = 8
    mlp_ratio: float = 4.0
    same_frame_window: int = 2
    same_frame_window_h: Optional[int] = None
    same_frame_window_w: Optional[int] = None
    same_frame_dilation_h: int = 1
    same_frame_dilation_w: int = 1
    temporal_offsets: Tuple[int, ...] = (-4, -3, -2, -1, 1, 2, 3)
    temporal_spatial_window: int = 2
    temporal_spatial_window_h: Optional[int] = None
    temporal_spatial_window_w: Optional[int] = None
    temporal_spatial_dilation_h: int = 1
    temporal_spatial_dilation_w: int = 1
    temporal_drift_h: int = 1
    temporal_drift_w: int = 1
    spatial_only: bool = False
    routing_topk_enable: bool = True
    routing_topk_fraction: float = 0.2
    routing_topk_min: int = 8
    routing_topk_max: int = 32
    routing_topk_per_head: bool = True
    topk_neighbors: Optional[int] = None
    topk_per_head: bool = True
    global_cls: bool = False
    posenc: str = "learned"
    rope_base: float = 10000.0
    rope_mode: str = "flat"
    rope_base_t: Optional[float] = None
    rope_base_h: Optional[float] = None
    rope_base_w: Optional[float] = None
    max_seq_len: Optional[int] = None

    def __post_init__(self) -> None:
        self.patch_size = (int(self.patch_size[0]), int(self.patch_size[1]))
        self.temporal_offsets = tuple(int(o) for o in self.temporal_offsets)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LWMConfig":
        return cls(**data)


class LWMModel(nn.Module):
    def __init__(self, config: LWMConfig) -> None:
        super().__init__()
        self.config = config
        patch_dim = config.patch_size[0] * config.patch_size[1] * 2
        self.tokenizer = ComplexPatchTokenizer(config.phase_mode)
        self.patch_embed = nn.Linear(patch_dim, config.embed_dim)
        self.global_cls = config.global_cls
        pos_len = (config.max_seq_len or 0) + (1 if self.global_cls else 0)
        if pos_len == 0:
            pos_len = 1
        if config.posenc == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_len, config.embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        else:
            self.register_buffer("pos_embed", torch.zeros(1, pos_len, config.embed_dim), persistent=False)
        if self.global_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.encoder = LWMEncoder(config)
        self.head = nn.Linear(config.embed_dim, patch_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _add_positional(self, tokens: Tensor) -> Tensor:
        if self.config.posenc == "learned":
            return tokens + self.pos_embed[:, : tokens.size(1)]
        return tokens

    def forward_tokens(
        self,
        tokens: Tensor,
        mask: Tensor,
        T: int,
        H: int,
        W: int,
        *,
        return_cls: bool = False,
    ) -> Dict[str, Optional[Tensor]]:
        embeddings = self.patch_embed(tokens)
        include_cls = self.global_cls
        if include_cls:
            cls_tokens = self.cls_token.expand(embeddings.size(0), -1, -1)
            embeddings = torch.cat([embeddings, cls_tokens], dim=1)
            cls_mask = torch.zeros((embeddings.size(0), 1), dtype=torch.bool, device=embeddings.device)
            mask = torch.cat([mask, cls_mask], dim=1)
        embeddings = self._add_positional(embeddings)
        embeddings = embeddings.masked_fill(mask.unsqueeze(-1), 0.0)
        encoded = self.encoder(embeddings, T, H, W, include_cls)
        if include_cls:
            reconstruction = self.head(encoded[:, :-1, :])
            cls = encoded[:, -1, :]
        else:
            reconstruction = self.head(encoded)
            cls = None
        return {"reconstruction": reconstruction, "cls": cls if return_cls else None}

    def forward(self, seq: Tensor, mask: Optional[Tensor] = None, *, return_cls: bool = False) -> Dict[str, Optional[Tensor]]:
        tokens, base_mask = self.tokenizer(seq, self.config.patch_size)
        total_mask = base_mask if mask is None else mask
        ph, pw = self.config.patch_size
        T = seq.size(1)
        H = seq.size(2) // ph
        W = seq.size(3) // pw
        return self.forward_tokens(tokens, total_mask, T, H, W, return_cls=return_cls)

    @torch.no_grad()
    def forward_features(self, seq: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        outputs = self.forward(seq, return_cls=True)
        return outputs["reconstruction"], outputs["cls"]


class LWMBackbone(LWMModel):
    """Minor alias kept for backwards compatibility with legacy scripts."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *model_args: Any,
        config: Optional[LWMConfig] = None,
        map_location: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "LWMBackbone":
        path = Path(pretrained_model_name_or_path)
        state: Dict[str, Tensor]

        if path.is_dir():
            directory = path
            state_path = directory / "pytorch_model.bin"
            if not state_path.exists():
                raise FileNotFoundError(f"Pretrained weights not found at {state_path}")
            raw = torch.load(state_path, map_location=map_location)
            if isinstance(raw, dict) and any(isinstance(v, torch.Tensor) for v in raw.values()):
                state = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
            else:
                raise ValueError(f"Unexpected checkpoint format at {state_path}")
            checkpoint_config_dict = None
            config_path = directory / "config.json"
            if config_path.exists():
                with config_path.open("r") as handle:
                    checkpoint_config_dict = json.load(handle)
                    checkpoint_config = LWMConfig.from_dict(checkpoint_config_dict)
                    if config is None:
                        config = checkpoint_config
                    else:
                        checkpoint_dict = checkpoint_config.to_dict()
                        provided_dict = config.to_dict()
                        merged_dict = {**checkpoint_dict, **provided_dict}
                        config = LWMConfig.from_dict(merged_dict)
        else:
            if not path.exists():
                raise FileNotFoundError(f"Pretrained weights not found at {path}")
            raw = torch.load(path, map_location=map_location)
            if isinstance(raw, dict) and "model_state_dict" in raw:
                state = raw["model_state_dict"]
                checkpoint_config = raw.get("config")
            elif isinstance(raw, dict):
                state = {k: v for k, v in raw.items() if isinstance(v, torch.Tensor)}
            else:
                raise ValueError("Unsupported checkpoint format; expected a state_dict or training checkpoint.")

        if config is None:
            config = LWMConfig()

        if config.max_seq_len is None and "pos_embed" in state:
            pos_len = int(state["pos_embed"].shape[1])
            cls_tokens = 1 if config.global_cls else 0
            inferred = max(0, pos_len - cls_tokens)
            if inferred > 0:
                config.max_seq_len = inferred
        remapped_state = cls._remap_state_dict(state)
        model = cls(config, *model_args, **kwargs)
        model.load_state_dict(remapped_state, strict=False)
        return model

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        config_path = directory / "config.json"
        with config_path.open("w") as handle:
            json.dump(self.config.to_dict(), handle, indent=2)
        state_path = directory / "pytorch_model.bin"
        torch.save(self.state_dict(), state_path)

    @staticmethod
    def _config_from_checkpoint(data: Any) -> Optional[LWMConfig]:
        if not isinstance(data, dict):
            return None
        model_cfg = data.get("model", data)
        if not isinstance(model_cfg, dict):
            return None
        allowed = {field.name for field in fields(LWMConfig)}
        kwargs: Dict[str, Any] = {}
        for key, value in model_cfg.items():
            if key not in allowed:
                continue
            if key == "patch_size" and isinstance(value, (list, tuple)):
                value = tuple(int(v) for v in value)
            if key == "temporal_offsets" and isinstance(value, (list, tuple)):
                value = tuple(int(v) for v in value)
            kwargs[key] = value
        if not kwargs:
            return None
        return LWMConfig(**kwargs)

    @staticmethod
    def _remap_state_dict(state: Dict[str, Tensor]) -> Dict[str, Tensor]:
        remapped: Dict[str, Tensor] = {}
        for key, value in state.items():
            new_key = key
            if key.startswith("embed."):
                new_key = key.replace("embed", "patch_embed", 1)
            elif key.startswith("blocks."):
                new_key = key.replace("blocks", "encoder.layers", 1)
            elif key.startswith("norm."):
                new_key = key.replace("norm", "encoder.norm", 1)
            remapped[new_key] = value
        return remapped


def compute_nmse(pred: Tensor, target: Tensor, mask: Tensor) -> float:
    B = pred.size(0)
    nmse_vals = []
    for b in range(B):
        m = mask[b]
        if m.sum() == 0:
            continue
        se = (pred[b][m] - target[b][m]).pow(2).sum()
        sp = target[b][m].pow(2).sum().clamp_min(1e-12)
        nmse_vals.append((se / sp).item())
    if not nmse_vals:
        return float('nan')
    return sum(nmse_vals) / len(nmse_vals)


def masked_nmse_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    diff = (pred - target).abs() ** 2
    power = target.abs() ** 2
    mask_f = mask.float()
    diff_sum = (diff.sum(-1) * mask_f).sum(-1)
    power_sum = (power.sum(-1) * mask_f).sum(-1).clamp_min(1e-12)
    nmse = diff_sum / power_sum
    valid = mask.sum(-1) > 0
    return nmse[valid].mean() if valid.any() else nmse.mean()


def masked_mse_loss(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    diff = (pred - target).abs() ** 2
    mask_f = mask.float()
    num = (diff.sum(-1) * mask_f).sum()
    denom = mask_f.sum().clamp_min(1.0)
    return num / denom


__all__ = [
    "ComplexPatchTokenizer",
    "LWMConfig",
    "LWMModel",
    "LWMBackbone",
    "compute_nmse",
    "masked_nmse_loss",
    "masked_mse_loss",
]
