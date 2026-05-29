# Multimodal Architecture

This document describes the active 16-to-4 multimodal channel prediction
models. The source of truth is the code under `multimodal_code_index/models/`
and `multimodal_code_index/train_multimodal4.py`.

## 1. Active Task

```text
channel_history : (B, K=16, Na=16, Nsc, 2)
image_seq       : (B, T_img=8, 3, 224, 224)   # multimodal mode only
image_time_offsets: (B, T_img, 1)             # seconds since sample time
image_valid_mask: (B, T_img) bool
target          : (B, P=4, Na=16, Nsc, 2)
prediction      : (B, P=4, Na=16, Nsc, 2)
```

`2` is `[real, imag]` for the complex channel. The current runner uses
`channel + RGB` in multimodal mode. LiDAR encoders exist in the model classes,
but `train_multimodal4.py` builds the active experiments with `use_lidar=False`.
Radar is not used by the active runner.

## 2. Current Model Families

| Model | Channel representation | Multimodal fusion | Changed in current update |
|---|---|---|---|
| `lstm` | wideband time tokens `(B, K, 256)` | per-time modality attention over `[channel_t, rgb_t]` | yes |
| `lwm` | wideband time tokens `(B, K, 64) -> (B, K, 256)` | per-time modality attention over `[channel_t, rgb_t]` | yes |
| `lwm_temporal` | antenna-subcarrier patch tokens across time | CLS scene injection | no |
| `chiron` | antenna-subcarrier patch tokens across time | channel tokens cross-attend to image sequence tokens | no |

The important change is that `lstm` and `lwm` are no longer per-subcarrier
fusion models. They now encode each channel history step as a wideband token:

```text
old LSTM/LWM:
  (B, K, Na, Nsc, 2) -> (B, Nsc, D)
  one token per subcarrier

current LSTM/LWM:
  (B, K, Na, Nsc, 2) -> (B, K, D)
  one token per time step, each token sees all antennas and subcarriers
```

## 3. LSTM Channel Path

Source: `multimodal_code_index/models/lstm_multimodal.py`

```text
channel_history: (B, K, Na, Nsc, 2)
  -> reshape(B, K, Na*Nsc*2)
  -> nn.LSTM(input=Na*Nsc*2, hidden=256, layers=3)
  -> all hidden states
  -> channel_tokens: (B, K, 256)
```

The LSTM keeps all `K` hidden states. The model therefore exposes one token per
channel-history time step, not one token per subcarrier.

In channel-only mode:

```text
channel_tokens (B, K, 256)
  -> ChannelPredictionHead
  -> prediction (B, P, Na, Nsc, 2)
```

## 4. LWM Channel Path

Source: `multimodal_code_index/models/lwm_multimodal.py`

```text
channel_history: (B, K, Na, Nsc, 2)
  -> reshape(B, K, Na*Nsc*2)
  -> Linear(Na*Nsc*2 -> d_model=64) + time position embedding
  -> Transformer encoder layers
  -> hidden: (B, K, 64)
  -> projection adapter Linear(64 -> 256) + LayerNorm
  -> channel_tokens: (B, K, 256)
```

The LWM model is also wideband-time now. It uses the same prediction head as
LSTM after the projection adapter.

## 5. RGB Sensor Path for LSTM/LWM

Sources:

- `multimodal_code_index/models/lstm_multimodal.py`
- `multimodal_code_index/models/lwm_multimodal.py`
- `multimodal_code_index/models/fusion_blocks.py`

The RGB branch is paper-inspired but simplified for the current `channel + RGB`
experiment. It does not implement the paper's LiDAR branch, BGAM stack, or LLM
reprogramming. The implemented part is the time-aligned sensor representation
and per-time modality attention.

```text
image_seq: (B, T, 3, 224, 224)
  -> ImageTokenEncoder per frame
  -> frame patch tokens: (B, T, 49, 256)
  -> SensorFrameSummarizer
       learnable query attends to the 49 spatial tokens per frame
       optional image_time_offsets projection is added
  -> frame_tokens: (B, T, 256)
  -> align to channel history K using image_time_offsets and delta_t
  -> rgb_tokens: (B, K, 256)
```

Alignment rule:

```text
channel history index k has offset (K - 1 - k) * delta_t
default delta_t = 0.0005 seconds

For each channel-history time, choose the nearest available RGB frame whose
image_time_offset is not newer than that channel time. If no valid frame exists,
use no_image_token and mark that RGB token invalid.
```

This means fusion happens on the same time grid as the channel history:

```text
t-K+1        t-K+2        ...        t
channel_0    channel_1               channel_K-1
rgb_0        rgb_1                   rgb_K-1
```

## 6. Per-Time Modality Fusion for LSTM/LWM

Source: `multimodal_code_index/models/fusion_blocks.py`

`PerTimeModalityFusion` fuses modalities independently at each channel-history
time step:

```text
channel_tokens: (B, K, D)
rgb_tokens:     (B, K, D)

for each time step k:
  modalities_k = [channel_k, rgb_k]
  learnable query attends to modalities_k
  GatedFFN updates the fused token

output: (B, K, D)
```

With multiple sensor modalities, the same block can consume
`sensor_tokens: (B, K, M, D)`. The current runner only passes RGB.

## 7. Prediction Head

LSTM and LWM now use `ChannelPredictionHead` from
`multimodal_code_index/models/chiron_channel.py`.

```text
fused/channel tokens: (B, K, 256)
  -> P learnable future queries attend to all K history tokens
  -> MLP emits Na*Nsc*2 values per future step
  -> prediction: (B, P, Na, Nsc, 2)
```

The physical channel values are emitted only by the prediction head. The
channel encoder, RGB encoder, and fusion blocks produce abstract features.

## 8. End-to-End Flow for Current LSTM/LWM Multimodal

```text
channel_history (B,K,Na,Nsc,2)
    |
    |  wideband channel encoder
    v
channel_tokens (B,K,256)

image_seq (B,T,3,224,224)
    |
    |  ResNet18 token encoder + frame summarizer
    v
frame_tokens (B,T,256)
    |
    |  image_time_offsets + delta_t alignment
    v
rgb_tokens (B,K,256)

for each fusion layer:
    [channel_t, rgb_t] --learnable-query attention--> fused_t

fused_tokens (B,K,256)
    |
    |  ChannelPredictionHead
    v
prediction (B,P,Na,Nsc,2)
```

## 9. LWM-Temporal and Chiron

`lwm_temporal` and `chiron` were already wideband/patch-time models and were not
rewritten in this update.

### LWM-Temporal

Sources:

- `multimodal_code_index/models/lwm_temporal.py`
- `multimodal_code_index/models/lwm_temporal_multimodal.py`

It appends masked future channel frames and processes time plus
antenna-subcarrier patches jointly. Image information is summarized into a
scene context and injected through the LWM temporal model's CLS path.

### Chiron

Sources:

- `multimodal_code_index/models/chiron_channel.py`
- `multimodal_code_index/models/chiron_multimodal.py`

Chiron converts every channel frame into antenna-subcarrier patch tokens,
processes them with temporal and spatial Chiron blocks, and decodes future
frames through `ChannelPredictionHead`. In multimodal mode, Chiron keeps its
existing image-sequence encoder and gated cross-modal fusion path.

## 10. Experiment Implications

Because the `lstm` and `lwm` channel encoders and heads changed, old checkpoints
for those two models should not be reused.

Minimum experiments to rerun:

```text
channel_only:
  lstm
  lwm

multimodal:
  lstm
  lwm
```

For a fully uniform result table, rerun all four multimodal models under the
same dataset split, seed, batch size, and epoch budget. If the existing
`lwm_temporal` and `chiron` results were produced with the same code and data
settings, they can be kept for a minimum rerun.

## 11. Runner Defaults

| Setting | Value |
|---|---|
| history length `K` | `16` |
| prediction horizon `P` | `4` |
| BS antennas `Na` | `16` |
| selected subcarriers `Nsc` | `64` |
| image frames | `8` latest-past RGB frames |
| channel interval `delta_t` | `0.0005` seconds |
| embed dim | `256` |
| fusion layers / heads | `3` / `4` |
| batch size | `4` |

## 12. Source Files

- `multimodal_code_index/models/lstm_multimodal.py` - wideband-time LSTM
- `multimodal_code_index/models/lwm_multimodal.py` - wideband-time Transformer
- `multimodal_code_index/models/fusion_blocks.py` - sensor summarizer and per-time fusion
- `multimodal_code_index/models/image_encoders.py` - ResNet18 image token encoder
- `multimodal_code_index/models/chiron_channel.py` - shared prediction head
- `multimodal_code_index/train_multimodal4.py` - model builder and batch routing
- `dataset_loader.py` - channel/RGB sample construction and image time offsets
