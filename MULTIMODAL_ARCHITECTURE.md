# Multimodal Architecture

This document describes the active 16-to-4 channel prediction model family in full
implementation detail. The source of truth is the code under
`multimodal_code_index/models/` and the runner
`multimodal_code_index/train_multimodal4.py`.

The goal of this document is to leave **no ambiguity** about how each model
encodes the channel and image inputs, how the two modalities are fused, and —
most importantly — **where exactly the prediction is produced inside the
network**. The latter is the single most common point of confusion when reading
the four `*_multimodal.py` files for the first time.

## Table of Contents

1. [Active task](#1-active-task)
2. [Representation vs Prediction split](#2-representation-vs-prediction-split)
3. [Where exactly does prediction happen?](#3-where-exactly-does-prediction-happen)
4. [Shared image and fusion blocks](#4-shared-image-and-fusion-blocks)
5. [Model 1: LSTM Multimodal (deep dive)](#5-model-1-lstm-multimodal-deep-dive)
6. [Model 2: LWM Multimodal](#6-model-2-lwm-multimodal)
7. [Model 3: LWM-Temporal Multimodal](#7-model-3-lwm-temporal-multimodal)
8. [Model 4: Chiron Multimodal](#8-model-4-chiron-multimodal)
9. [Direct vs iterated multi-step](#9-direct-vs-iterated-multi-step)
10. [Runner defaults](#10-runner-defaults)
11. [Side-by-side comparison](#11-side-by-side-comparison)
12. [Common misconceptions](#12-common-misconceptions)
13. [Reporting notes](#13-reporting-notes)

---

## 1. Active task

```text
channel_history : (B, K=16, Na=16, Nsc, 2)
image_seq       : (B, T_img=8, 3, 224, 224)   # multimodal mode only
image_valid_mask: (B, T_img) bool              # left-pad indicator
target          : (B, P=4, Na=16, Nsc, 2)
prediction      : (B, P=4, Na=16, Nsc, 2)
```

- Channels are stored as complex `H ∈ ℂ^{16×Nsc}` decomposed into `[real, imag]`
  along the last dimension.
- Images come from `latest_past` policy — for each channel sample at time `t`,
  the dataset loader returns the 8 most recent past image frames (with
  left-padding for warm-up samples).
- Default runner uses `--no-pretrained-image` (ResNet18 trained from scratch).

The active runner supports:

| Mode | Inputs used |
|---|---|
| `channel_only` | channel history |
| `multimodal` | channel history + RGB image sequence |

LiDAR encoders exist in some model classes, but `train_multimodal4.py` builds
models with `use_lidar=False`. Radar is not used by the active runner.

Sources: `multimodal_code_index/train_multimodal4.py`,
`dataset_loader.py::ChannelPredictionDataset`.

---

## 2. Representation vs Prediction split

The four models share the same high-level skeleton:

```
            ┌─────────────────────────────┐
            │     Representation stage    │   (no actual prediction here)
            │                             │
   channel ─►   channel_encoder           │── ch_tokens (B, S_c, D=256)
   image ───►   image_encoder             │── img_tokens (B, N_s, D=256)
            │                             │
            │   Fusion(ch_tokens, img)    │── fused (B, S_c, D=256)
            └──────────────┬──────────────┘
                           │
            ┌──────────────▼──────────────┐
            │      Prediction stage       │   ★ real channel values emitted
            │       (head Linear)         │
            └─────────────────────────────┘
                           │
                           ▼
              pred (B, P, Na, Nsc, 2)
```

Key idea:

- **Representation stage** produces abstract D=256 vectors. These are *not*
  channel values.
- **Prediction stage** is the last `Linear` layer inside the head. It is the
  first place in the network where the physical channel quantity (real/imag of
  H) appears.
- The MSE loss compares the head output against ground-truth channels, and the
  gradient flows back through fusion and the two encoders so that every
  representation learns to be useful for prediction.

This split makes it precise to talk about "what each module does":

| Module | Role |
|---|---|
| LSTM / LWM / LWM-Temporal / Chiron channel backbone | builds channel representation |
| ResNet18 (`ImageTokenEncoder`) | builds image representation |
| `GatedCrossModalFusion` / CLS injection | mixes the two representations |
| Last `Linear` in head | **the prediction** |

---

## 3. Where exactly does prediction happen?

The "prediction" — i.e. the mapping from a 256-d abstract vector to actual
channel real/imag values — is produced by exactly one module per model:

| Model | Module emitting prediction | Last `Linear` shape | Output meaning |
|---|---|---|---|
| LSTM Multimodal | `self.head[-1]` | `Linear(256, P·Na·2 = 128)` | flat (P, Na, real/imag) per subcarrier token |
| LWM Multimodal | `self.head[-1]` | `Linear(512, P·Na·2 = 128)` | same |
| LWM-Temporal Multimodal | `lwm.head` (a single `Linear`) | `Linear(D=256, ph·pw·2 = 128)` | per-patch real/imag (one patch = `4×16` antenna×SC) |
| Chiron Multimodal | `ChannelPredictionHead.mlp[-1]` | `Linear(1024, Na·Nsc·2 = 2048)` | per future-step `(Na, Nsc, 2)` |

→ **No model uses LSTM / Transformer / Chiron blocks / ResNet18 to directly
emit channel values.** Those modules produce intermediate representations.
The last `Linear` weight matrix is the only thing learning the
*abstract → physical channel* mapping.

This single fact resolves most "how does fusion work?" questions: fusion does
not predict; it just shapes the channel-token representation that the head
will subsequently decode.

---

## 4. Shared image and fusion blocks

### 4.1 `ImageTokenEncoder`

Source: `multimodal_code_index/models/image_encoders.py`

```text
input: (B, 3, 224, 224)
  └ torchvision ResNet18.stem (conv1 + bn1 + relu + maxpool)
  └ ResNet18 stages (layer1..layer4) → (B, 512, 7, 7)
  └ AdaptiveAvgPool2d(7, 7) (identity if already 7×7)
  └ Conv2d(512 → D=256, kernel=1)
  └ flatten 7·7 = 49 → (B, 49, 256)
  └ + learnable spatial_pos (1, 49, 256)
  └ LayerNorm
output: (B, 49, D=256)        # "49 spatial tokens per frame"
```

For an image sequence `(B, T_img, 3, 224, 224)`, the encoder is applied to
`B·T_img` flattened frames and reassembled into `(B, T_img·49, 256)`. Frames
that are marked invalid by `image_valid_mask` (left-pad warm-up frames) are
replaced by a learned `no_image_token (1, 1, 256)` so that downstream attention
never sees padding noise.

`ImageTokenEncoder` does **not** predict anything — it only produces tokens.

### 4.2 `GatedCrossModalFusion`

Source: `multimodal_code_index/models/fusion_blocks.py`

```python
# Q = channel tokens, K = V = sensor tokens (image, optionally lidar)
q  = LayerNorm(channel_tokens)                          # (B, S_c, D)
kv = LayerNorm(sensor_tokens)                           # (B, N_s, D)
attn_out, _ = MultiheadAttention(q, kv, kv,
                                 num_heads=4, dropout=0.1)
gate = sigmoid(Linear(concat([channel_tokens, attn_out])))   # (B, S_c, D)
channel_tokens = channel_tokens + gate * attn_out       # gated residual
output         = GatedFFN(channel_tokens)               # SwiGLU FFN + residual
```

- **Direction matters**: Q = channel, K = V = sensor. Channel tokens *pull* the
  information they need from image tokens; image tokens are not updated.
- **Gate**: per-token, per-dim sigmoid value in `(0, 1)` learns how much of the
  cross-attention output to mix back. A gate near 0 effectively bypasses
  the image modality for that subcarrier; near 1 fully incorporates it.
- Token count `S_c` is preserved — the prediction head only ever decodes
  channel tokens.

### 4.3 Chiron internal blocks (used only by Chiron)

`models/chiron_channel.py` defines `PatchEmbed2D`, `TemporalBlock`,
`SpatialBlock`, `GatedFFN`, `ChironBlock`, and `ChannelPredictionHead`. These
are detailed in §8. LSTM and LWM models do not use them.

---

## 5. Model 1: LSTM Multimodal (deep dive)

Source: `multimodal_code_index/models/lstm_multimodal.py`

LSTM Multimodal is the simplest of the four. We document it first in full and
later models highlight only their differences.

### 5.1 Why per-subcarrier LSTM?

Feeding the raw `(B, K, Na, Nsc, 2)` tensor into an LSTM would mean a sequence
of K=16 timesteps with feature width `Na·Nsc·2 = 2048` — overly wide and forces
the LSTM to model strong subcarrier correlations all at once. Instead:

```python
# (B, K, Na, Nsc, 2) → (B*Nsc, K, Na*2 = 32)
x = channel_history.permute(0, 3, 1, 2, 4).reshape(B * Nsc, K, Na * 2)
```

This produces **Nsc=64 independent timeseries**, each with K=16 steps and
feature width 32. All subcarriers share the same LSTM weights (parameter
sharing); the batch axis simply grows by 64×.

### 5.2 What "last hidden only" means precisely

```python
output, _ = self.lstm(x)          # (B*Nsc, K=16, hidden=256)
last      = output[:, -1, :]      # (B*Nsc, 256)
ch_tokens = last.reshape(B, Nsc, 256)
```

`nn.LSTM` emits a hidden state for **each of the 16 timesteps**:

```text
t=0      t=1      t=2     ...    t=14    t=15
  ◇        ◇        ◇                ◇       ◇       ← input (32-d)
  │        │        │                │       │
  ▼        ▼        ▼                ▼       ▼
[h₀] →  [h₁] →  [h₂] →  ...  →  [h₁₄] → [h₁₅]       ← hidden (256-d each)
                                              │
                                              ▼  ★ only this is kept
                                         (256-d) → channel_token[sc]
```

`output[:, -1, :]` slices the **last index along the time axis** — the hidden
state after the LSTM has seen all 16 history frames. By the recurrent structure
of LSTMs, this hidden contains a compressed summary of the entire input
sequence and is therefore the most information-dense single vector.

Why not use all 16 hidden states?

- Token count would grow 16× (64 → 1024), inflating fusion attention cost
  by 16× too.
- The LSTM already performs temporal compression; preserving every step is
  redundant.
- Empirically the last-hidden design is what the file ships.

Using all hiddens would *be valid* (treat them as `(B, Nsc·K, 256)`), but is
not what this implementation does.

### 5.3 Full data flow

```text
channel_history : (B, K=16, Na=16, Nsc, 2)
    │
    │  permute(0,3,1,2,4) → reshape          # per-SC unfold
    ▼
(B·Nsc, K, Na·2 = 32)
    │
    ▼  nn.LSTM(in=32, hidden=256, layers=3, dropout=0.1, batch_first=True)
    │      output: (B·Nsc, 16, 256)
    │      output[:, -1, :] → (B·Nsc, 256)
    ▼
channel_tokens : (B, Nsc, D=256)             ◀── one token per subcarrier

──────── channel_only mode branches to head here ────────

image_seq : (B, T_img=8, 3, 224, 224)
    │  reshape (B·T_img, 3, 224, 224)
    ▼  ImageTokenEncoder
    │      ResNet18 stem + layer1..4 → (B·T_img, 512, 7, 7)
    │      Conv1×1(512→256) → flatten 49 → +spatial_pos → LN
    ▼
(B·T_img, 49, 256) → reshape (B, T_img·49 = 392, 256)
    │
    │  image_valid_mask: pad frames replaced by learned no_image_token
    ▼
sensor_tokens : (B, 392, 256)                ◀── K/V for fusion

    ┌────────────────────────────────────────────┐
    │  GatedCrossModalFusion × 3 (heads = 4)     │
    │    Q   = channel_tokens (B, Nsc, 256)      │
    │    K=V = sensor_tokens  (B, 392, 256)      │
    │    fused = ch + sigmoid(L([ch, attn])) * attn
    │    out   = GatedFFN(fused)   (SwiGLU)      │
    └────────────────────────────────────────────┘
    │
    ▼
fused_channel_tokens : (B, Nsc, 256)         ◀── representation complete

    ▼  MLP head:
    │     LayerNorm(256)
    │     Linear(256 → 256)
    │     GELU
    │     ★ Linear(256 → P·Na·2 = 128) ★    ◀══════ prediction happens here ★
    ▼
(B, Nsc, 128) → reshape (B, Nsc, P, Na, 2)
              → permute → pred (B, P, Na, Nsc, 2)
```

### 5.4 ASCII diagram

```text
┌─────────────────────────────────────────────────────────────────┐
│                  LSTM Multimodal Predictor                      │
│                                                                 │
│ channel_history (B,K,Na,Nsc,2)      image_seq (B,T,3,224,224)   │
│        │ per-SC reshape                    │ flatten B·T        │
│        ▼                                   ▼                    │
│ ┌─────────────────┐               ┌──────────────────────┐      │
│ │ 3-layer LSTM    │               │ ResNet18 (scratch)   │      │
│ │ in=32 hid=256   │               │ → 7×7×512            │      │
│ │ dropout=0.1     │               │ → Conv1×1 → D=256    │      │
│ └────────┬────────┘               │ + spatial_pos + LN   │      │
│          │ last hidden            └──────────┬───────────┘      │
│          ▼                                   ▼                  │
│  channel_tokens                       image_tokens              │
│  (B, Nsc, 256)                        (B, T·49=392, 256)        │
│          │                                   │                  │
│          │ Q                              K, V                  │
│          ▼                                   │                  │
│       ┌─────────────────────────────────────────┐               │
│       │ GatedCrossModalFusion × 3               │               │
│       │  MHA(Q=ch, K=V=img, heads=4)            │               │
│       │  sigmoid gate × attn  (gated residual)  │               │
│       │  GatedFFN  (SwiGLU, mlp_ratio=4)        │               │
│       └─────────────────────────────────────────┘               │
│                       │                                         │
│                       ▼ fused (B, Nsc, 256)                     │
│               ┌──────────────────┐                              │
│               │ MLP head         │                              │
│               │ 256→256→GELU     │                              │
│               │ ★ →128 (P·Na·2) ★│ ◀── prediction Linear         │
│               └──────────────────┘                              │
│                       │                                         │
│                       ▼                                         │
│            prediction (B, P=4, Na=16, Nsc, 2)                   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.5 Default backbone settings

| Parameter | Value |
|---|---:|
| LSTM hidden | `256` |
| LSTM layers | `3` |
| Dropout | `0.1` |
| Fusion layers | `3` |
| Fusion heads | `4` |

### 5.6 Parameter breakdown (multimodal, Nsc=64)

| Module | Params | Notes |
|---|---:|---|
| LSTM (3-layer, 32→256) | ~1.45M | shared across all 64 subcarriers |
| ResNet18 (scratch) | 11.69M | image backbone |
| Image 1×1 conv 512→256 + spatial_pos + LN | ~0.14M | |
| `GatedCrossModalFusion` × 3 | ~2.5M | ~0.83M each (MHA + gate + GatedFFN) |
| `no_image_token` | <0.01M | |
| MLP head (LayerNorm + 2× Linear) | ~0.1M | **last Linear(256, 128) = 32,896 + 128 bias** |
| **Total** | **16,324,288** | matches checkpoint summary |

### 5.7 Gradient path — how images influence prediction

```text
loss = MSE(pred, target)
        │ backward
        ▼
prediction ← Linear(256, 128)  (the "prediction weights")
        │
        ▼
fused_channel_tokens ← GatedCrossModalFusion × 3
        │       │
        │       └─→ image_tokens ← ResNet18      (image weights updated)
channel_tokens ← LSTM (last hidden only)
        │
        ▼
LSTM weights ← gradient
```

Images influence the prediction **only** through the fusion step that mutates
the 256-d representation of each channel token. There is no separate
image-to-prediction path.

---

## 6. Model 2: LWM Multimodal

Source: `multimodal_code_index/models/lwm_multimodal.py`

LWM Multimodal replaces the LSTM temporal encoder with a per-subcarrier
**12-layer Transformer (d_model=64, heads=8, d_ff=256)** and inserts a
**Projection Adapter** to lift `d_model=64` → `D=256` before fusion.

### 6.1 Data flow (delta from LSTM)

```text
channel_history (B, K, Na, Nsc, 2)
    │  permute & reshape (identical to LSTM)
    ▼
(B·Nsc, K=16, Na·2 = 32)
    ▼  _Embedding: Linear(32 → 64) + learned PosEmbed(64) + LayerNorm
    ▼  ★ 12 × Transformer Encoder ★
    │     - MHA(d=64, heads=8)
    │     - FFN(64 → 256 → 64, ReLU)
    ▼  x[:, -1, :]                                   # last token, d=64
    ▼  reshape → (B, Nsc, 64)
    ▼  Projection Adapter:
    │     Linear(64 → 256) + LayerNorm                ◀── LSTM does not have this
channel_tokens : (B, Nsc, D=256)
    ▼  ── identical fusion + head pipeline from here on
```

Notes:

- The adapter runs even in `channel_only` mode so the head weights stay
  compatible across modes.
- "Last token slicing" is the same idea as LSTM's "last hidden only" — the
  transformer outputs K=16 contextual vectors per subcarrier, and only the last
  one (most temporally complete) is kept.
- Despite 12 layers the transformer is *thin* (d=64); the parameter count is
  smaller than the LSTM backbone in `channel_only` mode (0.82M vs 1.45M).

### 6.2 Head

```python
self.head = nn.Sequential(
    nn.LayerNorm(256),
    nn.Linear(256, 512),       # LSTM head is 256→256
    nn.GELU(),
    nn.Linear(512, P*Na*2),    # ★ prediction Linear(512, 128)
)
```

Mechanism is identical to LSTM's; only the intermediate width differs (512 vs
256).

### 6.3 Default backbone settings

| Parameter | Value |
|---|---:|
| Transformer `d_model` | `64` |
| Layers | `12` |
| Heads | `8` |
| FFN dim | `256` |
| Adapter dim | `256` |

---

## 7. Model 3: LWM-Temporal Multimodal

Sources:
- `multimodal_code_index/models/lwm_temporal.py`
- `multimodal_code_index/models/lwm_temporal_multimodal.py`

LWM-Temporal is the structurally most different model of the four. Two
distinctive choices:

1. It processes **time, antenna and subcarrier axes jointly** with a sparse
   spatio-temporal attention transformer (no per-SC factorization).
2. Multimodal fusion is **not token-level cross-attention**. Instead, the
   entire image sequence is compressed to a single `scene_ctx` token which is
   added to the LWM CLS token before encoding ("CLS Injection").

### 7.1 Channel pipeline

```text
channel_history (B, K=16, Na=16, Nsc, 2)
        ▼  real/imag → complex H ∈ ℂ^(B, K, Na, Nsc)
        ▼  + P=4 blank future frames (zeros) appended
(B, T = K+P = 20, Na=16, Nsc) complex
        ▼  ComplexPatchTokenizer (patch_size = ph=4, pw=16)
        │     patches/frame: H = 16/4 = 4, W = Nsc/16 → e.g. 4 for Nsc=64
        ▼
tokens : (B, T·H·W, ph·pw·2 = 128)
        ▼  mask = True only on future patches (P·H·W) — future is what we predict
        │
        ▼  _LWMModelCLSInject.forward_tokens:
        │     1) PatchEmbed Linear(128 → D=256)
        │     2) cls_token (1,1,256) appended; if scene_ctx supplied: cls += scene_ctx
        │     3) learned positional encoding
        │     4) ★ SparseSpatioTemporalAttention × 6 ★
        │          same_frame_window = -1           # full within-frame attention
        │          temporal_offsets = (-4,-3,-2,-1, 1, 2, 3)
        │          top-K routing 30% active
        │          CLS is always a neighbor → global access
        ▼
encoded : (B, T·H·W + 1, 256)
        ▼  ★ head(encoded[:, :-1, :])  excludes CLS ★
        │      head = single Linear(D=256 → ph·pw·2 = 128)   ← prediction
recon : (B, T·H·W, 128)
        ▼  slice last P·H·W tokens (future)
        ▼  view → permute → reshape
prediction : (B, P=4, Na=16, Nsc, 2)
```

### 7.2 Multimodal fusion via CLS injection

```text
image_seq → ImageTokenEncoder → sensor_tokens (B, T_img·49, 256)
                                              │
                                              │ Q = learned scene_query (1,1,256)
                                              ▼
                              MultiheadAttention(Q=query, K=V=sensor)
                                              │
                                              ▼  LayerNorm
                                    scene_ctx (B, 1, 256)
                                              │   ★ entire image sequence → 1 token
                                              ▼
                                    cls_token = cls_token + scene_ctx
                                              │
                                              ▼ sparse attention propagates
                                              │   scene context to all channel patches
                                              │   via CLS-as-neighbor wiring
```

Why CLS injection?

- The sparse `NeighborIndexer` caches a (T, H, W)-shaped index map. Putting
  sensor tokens beside channel patches would force a re-indexing.
- Adding `scene_ctx` into the CLS token leaves the indexer untouched while
  still making the global image summary visible to every channel patch
  (because CLS attends to every patch and every patch attends to CLS).

Trade-off: scene_ctx collapses the entire image sequence into a single
256-d vector, so fine-grained spatial cues from individual frames are lost.

### 7.3 Where is the prediction?

`lwm.head` is a single `Linear(D=256 → ph·pw·2 = 128)` applied to every patch
token. The values that matter for inference are the last `P·H·W` patches
(future frames). The head weight is the only place where abstract 256-d
representations are mapped to real/imag channel values.

### 7.4 Default backbone settings

| Parameter | Value |
|---|---:|
| Patch size | `(4, 16)` |
| Embed dim | `256` |
| Depth | `6` |
| Heads | `8` |
| Future prediction | direct masked reconstruction of `P=4` frames |
| Sparse temporal offsets | `(-4, -3, -2, -1, 1, 2, 3)` |

### 7.5 ASCII diagram

```text
                       ┌──────────────────────────────┐
                       │ image_seq → ResNet18 tokens  │
                       │  (B, T·49, 256)              │
                       └──────────────┬───────────────┘
                                      │ learned scene_query
                                      ▼
                                MHA(Q=query, K=V=img)
                                      │
                                      ▼
                            scene_ctx (B, 1, 256)
                                      │   ★ single-token compression
                                      ▼
channel_history      ┌────────────────┴───────────────┐
   ↓ complex+future  │  cls_token += scene_ctx        │
patch (4×16)         │  + patch tokens (B, T·H·W, 256)│
   ↓                 │  + future-mask (P·H·W)         │
                     │                                │
                     │  Sparse Spatio-Temporal        │
                     │  Transformer × 6               │
                     │  (top-K 30% routing)           │
                     └────────────────┬───────────────┘
                                      │
                                      ▼
                       ★ head: Linear(256 → 128) ★
                                      │ (per-patch real/imag)
                                      ▼
                      extract last P frames → reshape
                                      │
                                      ▼
                       (B, P=4, Na=16, Nsc, 2)
```

---

## 8. Model 4: Chiron Multimodal

Sources:
- `multimodal_code_index/models/chiron_channel.py`
- `multimodal_code_index/models/chiron_multimodal.py`

Chiron is the heaviest of the four (29M params in multimodal mode). It tokenizes
the channel into **2D `(antenna × subcarrier)` patches** and processes time and
space alternately via a stack of factorized `ChironBlock`s. Multimodal fusion
runs the image sequence through a dedicated `ImageSequenceEncoder` first, then
applies token-level cross-attention.

### 8.1 Channel pipeline

```text
channel_history (B, K=16, Na=16, Nsc, 2)
        ▼  reshape (B·K, Na, Nsc, 2)
        ▼  PatchEmbed2D (patch_h=4, patch_w=32)
        │     S = (Na/4) × (Nsc/32)            e.g. 4 × 2 = 8 patches/frame for Nsc=64
        ▼  per-patch Linear(ph·pw·2 → 256) + LN + GELU
tokens : (B, K·S, D=256)
        ▼  + temporal_pos(1, K, 1, 256) + spatial_pos(1, 1, S, 256)
        ▼  ★ ChironBlock × 6 ★
        │     TemporalBlock:
        │        gated symmetric Conv1d(kernel=7, groups=D) on time axis (B·S, K, D)
        │        + bidirectional MultiheadAttention(heads=4)
        │     SpatialBlock:
        │        MultiheadAttention(heads=4) on (B·K, S, D)
        │     GatedFFN (SwiGLU, ratio=4)
        ▼
channel_tokens : (B, K·S, 256)
        ▼  LayerNorm
```

### 8.2 Image pipeline (Chiron has its own image temporal encoder)

```text
image_seq (B, T_img, 3, 224, 224)
        ▼  ImageTokenEncoder → (B·T_img, 49, 256)
        ▼  ImageSequenceEncoder:
        │     + frame_pos (1, T_img, 256)
        │     frame_summary = mean over 49 tokens per frame  → (B, T_img, 256)
        │     MHA(heads=4) across T_img frame summaries     ← temporal context
        │     + broadcast back into per-token representations
        │     LayerNorm
image_tokens : (B, T_img·49 = 392, 256)  (+ token_padding_mask)
```

LWM-Temporal collapses image into 1 token; Chiron keeps all 392 with a learned
temporal mixer applied first. This is the heaviest fusion path of the four.

### 8.3 Fusion + prediction head

```text
        ▼  GatedCrossModalFusion × 3 (Q = channel, K=V = image)
fused_channel_tokens : (B, K·S, 256)
        │
        ▼  ★ ChannelPredictionHead ★
        │     - P=4 learnable queries cross-attend to (K·S) tokens
        │     - pooled (B, 4, 256)
        │     - shared MLP(256 → 1024 → 1024 → Na·Nsc·2 = 2048)
        │                              ★ prediction Linear(1024, 2048) ★
        ▼
prediction : (B, P=4, Na=16, Nsc, 2)
```

Chiron's head differs from LSTM/LWM in that **each future step has its own
learnable query** that performs cross-attention over the entire channel-token
sequence. The MLP that follows is shared across steps. This is still
**direct multi-step** (no autoregression), but with explicit per-step
query slots.

### 8.4 Default backbone settings

| Parameter | Value |
|---|---:|
| Patch size | `(4, 32)` |
| Embed dim | `256` |
| Depth | `6` |
| Heads | `4` |
| Temporal conv kernel | `7` |
| FFN ratio | `4.0` |

### 8.5 ASCII diagram

```text
channel_history (B,16,16,64,2)             image_seq (B,8,3,224,224)
        │                                          │
        ▼                                          ▼
PatchEmbed2D(4×32) → 8 patches/frame         ResNet18 + 1×1 conv
+ temporal_pos + spatial_pos                   → (B·8, 49, 256)
        │                                          │
        ▼                                          ▼
(B, K·S = 128, 256)                          ImageSequenceEncoder:
        │                                     + frame_pos
        │                                     frame_summary = mean(49 tokens)
        │                                     MHA across 8 frames
        │                                     inject back into per-token reps
        │                                          │
        │                                          ▼
        │                                    image_tokens (B, 392, 256)
        │                                          │
        ▼                                          │
ChironBlock × 6                                    │
 ┌──────────────────────────┐                      │
 │ TemporalBlock            │                      │
 │   GatedConv1d kernel=7   │                      │
 │   + bidirectional MHA    │                      │
 │ SpatialBlock             │                      │
 │   MHA across 8 patches   │                      │
 │ GatedFFN (SwiGLU x4)     │                      │
 └──────────────────────────┘                      │
        │                                          │
        ▼ channel_tokens (B,128,256)               │
        │                                          │
        └──────►  GatedCrossModalFusion × 3 ◄──────┘
                  Q = channel, K=V = image
                            │
                            ▼
                  fused (B, 128, 256)
                            │
                            ▼
              ChannelPredictionHead:
                4 learnable query → cross-attn → (B, 4, 256)
                MLP(256 → 1024 → 1024 → Na·Nsc·2 = 2048)
                                       ★ prediction ★
                            │
                            ▼
              prediction (B, P=4, Na=16, Nsc, 2)
```

---

## 9. Direct vs iterated multi-step

Channel prediction for `P` future steps can use either paradigm:

| Paradigm | Behaviour | Used in this repo? |
|---|---|---|
| **Iterated / Autoregressive** | predict `t+1`, append, predict `t+2`, … `t+P` | No |
| **Direct multi-step** | output all `P` steps at once | Yes — all four models |

How each model implements direct multi-step:

| Model | Decoder shape |
|---|---|
| LSTM Multimodal | flat MLP head, `Linear(256, P·Na·2 = 128)` |
| LWM Multimodal | flat MLP head, `Linear(512, P·Na·2 = 128)` |
| LWM-Temporal Multimodal | masked future-frame reconstruction — future frames are zero-padded into the input and the transformer fills them in |
| Chiron Multimodal | `P` learnable queries cross-attend, then a shared MLP decodes each pooled vector |

LWM-Temporal is the most "sequence-decoder-like" — every future frame still
goes through the same transformer that processed history, just at masked
positions. LSTM/LWM are encoder-only sequence models with a flat regression
head.

### 9.1 Trade-offs (paper discussion seed)

| | Direct (this repo) | Iterated |
|---|---|---|
| Training | fast (P steps learned at once) | slow (unrolled P times) |
| Inference | one forward pass | P forward passes |
| Error accumulation | none — each step is independent | yes |
| Step-to-step consistency | weaker — `t+1` and `t+4` are learned separately | stronger |
| Long horizon | weaker (head grows with P) | stronger (per-step decoder) |

For the active task (`P=4`, 2 ms horizon at 2 kHz), direct is the natural
choice. For `P ≫ 16` an autoregressive or masked-reconstruction approach
would be more attractive.

---

## 10. Runner defaults

Sources: `multimodal_code_index/train_multimodal4.py`,
`multimodal_code_index/run_multimodal_16to4/run_16to4.py`

| Setting | Default |
|---|---:|
| `K` history length | `16` |
| `P` prediction horizon | `4` |
| BS antennas | `16` |
| Selected subcarriers | `64` |
| Raw channel subcarriers | `512` |
| Image frames | `8` |
| Embed dim | `256` |
| Fusion layers | `3` |
| Fusion heads | `4` |
| Batch size | `4` |
| Optimizer | `AdamW` |
| LR | `1e-3` (wrapper passes `1e-4` for current sc04 runs) |
| Scheduler | cosine with warmup (3 epochs) |
| Loss | `MSELoss` |
| Metrics | NMSE dB, cosine similarity |

---

## 11. Side-by-side comparison

| Item | LSTM Multimodal | LWM Multimodal | LWM-Temporal Multimodal | Chiron Multimodal |
|---|---|---|---|---|
| Channel token unit | per-SC, 1 token | per-SC, 1 token | (antenna × SC) patch (4×16) | (antenna × SC) patch (4×32) |
| Tokens per sample | `Nsc` (e.g. 64) | `Nsc` (e.g. 64) | `T·H·W` (e.g. 320 + CLS) | `K·S` (e.g. 128) |
| Temporal modeling | per-SC LSTM 3L hid=256 | per-SC Transformer 12L d=64 | Sparse spatio-temporal 6L (offsets ±1..±4) | Temporal Conv1d k=7 + Bi-MHA |
| Cross-SC modeling | none until head | none until head | sparse cross-patch attention | SpatialBlock MHA |
| Future-step decoding | flat MLP head | flat MLP head | masked future-patch reconstruction | P learnable query cross-attn |
| **Module emitting prediction** | `head[-1]` = `Linear(256, 128)` | `head[-1]` = `Linear(512, 128)` | `lwm.head` = `Linear(256, 128)` (per-patch) | `ChannelPredictionHead.mlp[-1]` = `Linear(1024, 2048)` |
| Image-fusion site | right after LSTM last hidden | right after LWM projection adapter | scene_ctx added into CLS token | channel tokens vs ImageSequenceEncoder tokens cross-attn |
| Image-fusion mechanism | `GatedCrossModalFusion × 3` | `GatedCrossModalFusion × 3` | scene_query → 1 token → CLS += scene_ctx | `ImageSequenceEncoder` + `GatedCrossModalFusion × 3` |
| Channel-only params | 1.45M | 0.82M | 3.30M | 13.91M |
| Multimodal params | 16.32M | 15.70M | (not run yet) | 29.05M |

---

## 12. Common misconceptions

The single most common question about these models is:

> "In the multimodal models, how do LSTM/Transformer and ResNet18 fuse? Does
> the LSTM produce a channel prediction that is then combined with the image,
> or is the LSTM's hidden state combined?"

The answer for all four models is the same: **hidden-state-level late fusion**.

| Misconception | Reality |
|---|---|
| LSTM produces a channel prediction that gets fused with the image | LSTM only produces a hidden-state representation. Prediction is emitted by the head's last `Linear`, **after** fusion. |
| ResNet18 (or `ImageTokenEncoder`) produces its own channel prediction that is ensembled | The image encoder has no decoder. It only produces 49 spatial tokens per frame. |
| All 16 LSTM hidden states are used | Only the last hidden state (t=K-1) is taken via `output[:, -1, :]` and turned into the channel token. |
| The model is autoregressive (predict t+1, append, predict t+2, …) | All four models are **direct multi-step**: a single forward pass emits all `P` future steps. |
| Fusion output equals the channel prediction | Fusion outputs a 256-d abstract vector per channel token. The mapping to real channel values is done exclusively by the head's last `Linear`. |
| The "image branch" draws the channel | The image branch contributes only by reshaping the 256-d channel-token representation inside fusion. There is no image-to-channel path that bypasses fusion. |

### One-sentence summary

> Across all four models, channel prediction is emitted by a single
> module — the last `Linear` inside the prediction head. The
> LSTM / Transformer / Chiron / ResNet18 modules only build
> representations. `GatedCrossModalFusion` (or CLS injection in LWM-Temporal)
> mixes the two modalities' representations, and the last `Linear`
> weight matrix is the only place in the network that maps abstract
> 256-d vectors to channel real/imag values.

During training, MSE loss gradient starts at that last `Linear` and flows
backward through fusion into both encoders, so all representations learn to
be useful for prediction. Image modality influences prediction **only**
through the fusion step that mutates the channel-token representation.

---

## 13. Reporting notes

When describing results, include:

- dataset root and radio profile
- scenario list
- mode: `channel_only` or `multimodal`
- selected `Nsc`
- `K`, `P`, image frame count
- model name and key backbone settings
- best validation NMSE and checkpoint path

This avoids mixing legacy `dataset_final/` results with current
`wireless-dataset/` results.

---

## Code references

- `multimodal_code_index/models/lstm_multimodal.py` — LSTM Multimodal
- `multimodal_code_index/models/lwm_multimodal.py` — LWM Multimodal
- `multimodal_code_index/models/lwm_temporal_multimodal.py` — LWM-Temporal Multimodal
- `multimodal_code_index/models/lwm_temporal.py` — LWM-Temporal backbone (`LWMConfig`, `LWMModel`, sparse attention)
- `multimodal_code_index/models/chiron_channel.py` — Chiron channel backbone (`PatchEmbed2D`, `ChironBlock`, `ChannelPredictionHead`)
- `multimodal_code_index/models/chiron_multimodal.py` — Chiron Multimodal (`ImageSequenceEncoder`)
- `multimodal_code_index/models/image_encoders.py` — `ImageTokenEncoder`
- `multimodal_code_index/models/fusion_blocks.py` — `GatedCrossModalFusion`
- `multimodal_code_index/train_multimodal4.py` — Training loop
- `dataset_loader.py` — `ChannelPredictionDataset` (chronological 75/25 split)
- `EXPERIMENTS.md` — How to run
- `DATASET_SPEC.md` — Dataset format
