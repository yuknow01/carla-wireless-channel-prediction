# Multimodal Architecture

This document describes the active 16-to-4 model family. The source of truth is
the code under `multimodal_code_index/models/` and the runner
`multimodal_code_index/train_multimodal4.py`.

## Active Task

```text
channel_history: (B, K=16, Na=16, Nsc, 2)
image_seq:       (B, T_img=8, 3, 224, 224)   # multimodal mode
target:          (B, P=4, Na=16, Nsc, 2)
prediction:      (B, P=4, Na=16, Nsc, 2)
```

The active runner supports:

| Mode | Inputs used |
|---|---|
| `channel_only` | channel history |
| `multimodal` | channel history + RGB image sequence |

LiDAR encoders exist in some model classes, but `train_multimodal4.py` builds
models with `use_lidar=False`. Radar is not used by the active runner.

## Shared Image And Fusion Blocks

### Image Encoder

Source: `multimodal_code_index/models/image_encoders.py`

`ImageTokenEncoder` uses a ResNet18 backbone:

```text
image: (B, 3, H, W)
  -> ResNet18 stem and layer1..layer4
  -> AdaptiveAvgPool2d(7 x 7)
  -> 1x1 Conv projection to D=256
  -> spatial positional embedding
  -> LayerNorm
  -> image tokens: (B, 49, 256)
```

For image sequences, model classes flatten frames into `T_img * 49` image
tokens. Chiron additionally uses an image sequence encoder before fusion.

### Gated Cross-Modal Fusion

Source: `multimodal_code_index/models/fusion_blocks.py`

`GatedCrossModalFusion` lets channel tokens attend to sensor tokens:

```text
Q = channel tokens
K,V = image or sensor tokens
attn_out = MultiheadAttention(Q, K, V)
gate = sigmoid(Linear([channel, attn_out]))
fused = channel + gate * attn_out
output = GatedFFN(fused)
```

The direction is important: image tokens condition the channel representation;
the prediction head still predicts channel frames.

## Model 1: LSTM

Source: `multimodal_code_index/models/lstm_multimodal.py`

Purpose: per-subcarrier temporal baseline.

```text
channel_history: (B, K, Na, Nsc, 2)
  -> permute by subcarrier
  -> (B*Nsc, K, Na*2)
  -> 3-layer LSTM, hidden=256
  -> last hidden per subcarrier
  -> channel tokens: (B, Nsc, 256)

multimodal mode:
  image_seq -> ResNet18 tokens: (B, T_img*49, 256)
  channel tokens cross-attend to image tokens

head:
  per-subcarrier MLP
  -> (B, P, Na, Nsc, 2)
```

Default backbone settings:

| Parameter | Value |
|---|---:|
| LSTM hidden | `256` |
| LSTM layers | `3` |
| Dropout | `0.1` |
| Fusion layers | `3` |
| Fusion heads | `4` |

## Model 2: LWM

Source: `multimodal_code_index/models/lwm_multimodal.py`

Purpose: replace the LSTM temporal encoder with a per-subcarrier Transformer.

```text
channel_history: (B, K, Na, Nsc, 2)
  -> (B*Nsc, K, Na*2)
  -> linear embedding + positional embedding
  -> Transformer encoder stack
  -> last token per subcarrier
  -> projection adapter: 64 -> 256
  -> channel tokens: (B, Nsc, 256)

multimodal mode:
  image_seq -> ResNet18 tokens
  GatedCrossModalFusion

head:
  per-subcarrier MLP
  -> (B, P, Na, Nsc, 2)
```

Default backbone settings:

| Parameter | Value |
|---|---:|
| Transformer `d_model` | `64` |
| Layers | `12` |
| Heads | `8` |
| FFN dim | `256` |
| Adapter dim | `256` |

## Model 3: LWM-Temporal

Sources:

- `multimodal_code_index/models/lwm_temporal.py`
- `multimodal_code_index/models/lwm_temporal_multimodal.py`

Purpose: model time and antenna-frequency patches jointly instead of treating
each subcarrier independently.

```text
channel_history: (B, K, Na, Nsc, 2)
  -> real/imag to complex: (B, K, Na, Nsc)
  -> append P blank future frames
  -> seq: (B, K+P, Na, Nsc)
  -> mask future tokens
  -> patchify channel frames with patch=(4, 16)
  -> sparse spatio-temporal LWM encoder
  -> reconstruct masked future patches
  -> extract last P frames
  -> (B, P, Na, Nsc, 2)
```

Multimodal mode compresses image tokens into a single scene context token:

```text
image_seq -> ResNet18 tokens
scene_query cross-attends to image tokens
scene context: (B, 1, 256)
scene context is injected into the LWM CLS token
```

Default backbone settings:

| Parameter | Value |
|---|---:|
| Patch size | `(4, 16)` |
| Embed dim | `256` |
| Depth | `6` |
| Heads | `8` |
| Future prediction | direct masked reconstruction of `P=4` frames |
| Sparse temporal offsets | `(-4, -3, -2, -1, 1, 2, 3)` |

## Model 4: Chiron

Sources:

- `multimodal_code_index/models/chiron_channel.py`
- `multimodal_code_index/models/chiron_multimodal.py`

Purpose: patch-based channel backbone with explicit temporal and spatial blocks.

```text
channel_history: (B, K, Na, Nsc, 2)
  -> 2D channel patch embedding
  -> tokens per frame S = (Na / patch_h) * (Nsc / patch_w)
  -> add temporal and spatial positional embeddings
  -> repeat ChironBlock depth times:
       TemporalBlock over K for each spatial patch
       SpatialBlock over patches for each time frame
       GatedFFN
  -> channel tokens: (B, K*S, 256)

multimodal mode:
  image_seq -> image sequence tokens
  channel tokens cross-attend to image tokens

head:
  learned future-query prediction head
  -> (B, P, Na, Nsc, 2)
```

Default backbone settings:

| Parameter | Value |
|---|---:|
| Patch size | `(4, 32)` |
| Embed dim | `256` |
| Depth | `6` |
| Heads | `4` |
| Temporal conv kernel | `7` |
| FFN ratio | `4.0` |

## Runner Defaults

Source: `multimodal_code_index/train_multimodal4.py` and
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
| LR | `1e-3` |
| Scheduler | cosine with warmup |
| Loss | `MSELoss` |
| Metrics | NMSE dB, cosine similarity |

## Architecture Comparison

| Model | Channel token unit | Temporal modeling | Spatial/frequency modeling | Multimodal injection |
|---|---|---|---|---|
| LSTM | one token per subcarrier | per-subcarrier LSTM | no cross-subcarrier modeling before fusion | gated channel-to-image cross-attention |
| LWM | one token per subcarrier | per-subcarrier Transformer | no cross-subcarrier modeling before fusion | gated channel-to-image cross-attention |
| LWM-Temporal | antenna-frequency patches across `K+P` frames | sparse spatio-temporal attention with masked future frames | patch-level sparse attention | image scene token injected into CLS |
| Chiron | antenna-frequency patches over history frames | temporal conv/attention block | spatial patch attention block | gated channel-to-image cross-attention |

## Reporting Notes

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
