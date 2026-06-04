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

## 2. High-Level Architecture Image

The original high-level multimodal architecture image is kept here for quick
visual reference:

<img width="1024" height="559" alt="Multimodal architecture overview" src="https://github.com/user-attachments/assets/a01869f2-35e9-43ef-a4a0-05a6a98126ef" />

The current LSTM/LWM implementation differs from the old per-subcarrier version:
it uses wideband time tokens and fuses RGB per channel-history time step.

Current code checkpoints:

```python
# multimodal_code_index/train_multimodal4.py
use_image = args.mode == "multimodal"
common = dict(
    mode=args.mode,
    num_bs_antennas=args.num_bs_antennas,
    num_subcarriers=args.num_subcarriers,
    history_len=args.history_len,
    prediction_horizon=args.prediction_horizon,
    embed_dim=args.embed_dim,
    use_image=use_image,
    use_lidar=False,
)
```

```python
# multimodal_code_index/models/lstm_multimodal.py
B, K, Na, Nsc, _ = channel_history.shape
x = channel_history.reshape(B, K, Na * Nsc * 2)
output, _ = self.lstm(x)
ch_tokens = self.channel_proj(output)  # (B, K, embed_dim)
```

```python
# multimodal_code_index/models/lwm_multimodal.py
B, K, Na, Nsc, _ = channel_history.shape
x = channel_history.reshape(B, K, Na * Nsc * 2)
x = self.embedding(x)
for layer in self.layers:
    x = layer(x)
ch_tokens = self.proj_adapter(x)  # (B, K, embed_dim)
```

The older LSTM image/code block that described "last hidden only" per-subcarrier
processing is not restored in the main path because it no longer matches the
current active LSTM/LWM implementation.

## 3. Current Model Families

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

### LSTM/LWM Channel-Only Embedding View (t x subcarrier)

For the LSTM/LWM channel-only path, the channel history can be drawn as a
time-subcarrier grid. The x-axis is time `t`; the y-axis is subcarrier. Each
cell still contains the full BS antenna vector and complex components `(Na, 2)`.

```text
                         time t ->

subcarrier       t0             t1             ...            tK-1
    sc0       H[t0, sc0]     H[t1, sc0]                    H[tK-1, sc0]
    sc1       H[t0, sc1]     H[t1, sc1]                    H[tK-1, sc1]
    sc2       H[t0, sc2]     H[t1, sc2]                    H[tK-1, sc2]
    ...           ...            ...           ...              ...
 scNsc-1   H[t0, scNsc-1] H[t1, scNsc-1]              H[tK-1, scNsc-1]

For one sample, each cell H[tk, scj] = complex channel over all Na antennas:
(Na, 2)
```

The current LSTM/LWM channel-only embedding is 1D along the time axis. For each
time column, all antenna, subcarrier, real, and imaginary values are flattened
into one wideband vector:

```text
one time column tk:
  H_sample[tk, :, :, :]    -> (Na, Nsc, 2)
  flatten antenna/subcarrier/complex axes
  x_tk                    -> (Na * Nsc * 2)

all history columns:
  [x_t0, x_t1, ..., x_tK-1] -> 1D temporal sequence
```

Then each model turns that 1D sequence into channel tokens:

```text
LSTM:
  x_t sequence -> LSTM over time -> channel_tokens (B, K, 256)

LWM:
  x_t sequence
    -> Linear(Na*Nsc*2 -> 64) + time position embedding
    -> Transformer over time
    -> Linear(64 -> 256) + LayerNorm
    -> channel_tokens (B, K, 256)
```

So, in the current LSTM/LWM channel-only setting, the subcarrier axis is not a
separate token axis. It is folded into the feature dimension of each time token.

### Previous Per-Subcarrier Embedding Reference (Archived)

This subsection preserves only the embedding view from the previous LSTM/LWM
multimodal structure. It is kept for comparison, but it is **not** the active
LSTM/LWM runner path after the wideband-time update.

Previous LSTM/LWM channel embedding used one token per subcarrier:

```text
channel_history: (B, K, Na, Nsc, 2)
  -> permute(0, 3, 1, 2, 4)
  -> (B, Nsc, K, Na, 2)
  -> reshape(B*Nsc, K, Na*2)

default:
  (B, 16, 16, 64, 2)
  -> (B, 64, 16, 16, 2)
  -> (B*64, 16, 32)
```

Embedding view:

```text
For each subcarrier sc_j:

              time ->
          t0       t1       t2      ...     t15
       +-------+--------+--------+-------+--------+
sc_j   | Na*2  | Na*2   | Na*2   | ...   | Na*2   |
       | 32    | 32     | 32     |       | 32     |
       +-------+--------+--------+-------+--------+
           |
           v
       temporal encoder shared by all subcarriers
       LSTM or LWM Transformer over K=16
           |
           v
       one channel token for this subcarrier: (256)

All subcarriers:
  sc0 token, sc1 token, ..., sc63 token
  -> channel_tokens: (B, Nsc, 256) = (B, 64, 256)
```

The previous LSTM kept only the last temporal hidden state for each subcarrier:

<img width="2816" height="1536" alt="Previous per-subcarrier LSTM embedding" src="https://github.com/user-attachments/assets/fb5f62a4-f5df-4bfb-bd91-24340f852c55" />

```text
t=0      t=1      t=2     ...    t=14    t=15
  x        x        x                x       x       input per SC: 32-d
  |        |        |                |       |
  v        v        v                v       v
[h0] ->  [h1] ->  [h2] ->  ...  -> [h14] -> [h15]   hidden: 256-d each
                                             |
                                             v
                                      keep last hidden only
                                      channel_token[sc_j]: 256-d
```

Previous LWM used the same per-subcarrier unfold, but replaced the LSTM with a
Transformer:

```text
(B*Nsc, K, Na*2)
  -> Linear(32 -> 64) + learned position embedding
  -> Transformer over K
  -> last token
  -> reshape(B, Nsc, 64)
  -> Linear(64 -> 256) + LayerNorm
  -> channel_tokens: (B, Nsc, 256)
```

The previous image embedding was the same ResNet18 spatial-token idea:

```text
image_seq: (B, T_img, 3, 224, 224)
  -> flatten frames: (B*T_img, 3, 224, 224)
  -> ImageTokenEncoder
  -> (B*T_img, 49, 256)
  -> reshape
  -> image_tokens: (B, T_img*49, 256)

default T_img=8:
  image_tokens: (B, 392, 256)
```

Previous fusion then used channel tokens as queries and image tokens as
keys/values:

```text
channel_tokens: (B, Nsc=64, 256)      Q
image_tokens:   (B, T_img*49=392, 256) K/V
  -> GatedCrossModalFusion
  -> fused_channel_tokens: (B, Nsc=64, 256)
```

Current LSTM/LWM differs in the channel side:

```text
previous:
  one token per subcarrier -> (B, Nsc, 256)

current:
  one token per time step  -> (B, K, 256)
```

## 4. LSTM Channel Path

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

## 5. LWM Channel Path

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

## 6. RGB Sensor Path for LSTM/LWM

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

### LSTM/LWM Multimodal Shape Embedding View

The diagrams below omit the batch dimension `B` and use the default current
runner sizes: `K=16`, `T_img=8`, `Na=16`, `Nsc=64`, and `D=256`.

Channel input before embedding:

```text
                              time ->
              t0        t1        t2       ...      t15
           +--------+--------+--------+--------+--------+
sc0        | Na x 2 | Na x 2 | Na x 2 |  ...   | Na x 2 |
sc1        | Na x 2 | Na x 2 | Na x 2 |  ...   | Na x 2 |
sc2        | Na x 2 | Na x 2 | Na x 2 |  ...   | Na x 2 |
...        |  ...   |  ...   |  ...   |  ...   |  ...   |
sc63       | Na x 2 | Na x 2 | Na x 2 |  ...   | Na x 2 |
           +--------+--------+--------+--------+--------+

Each cell: (Na, 2) = (16, 2)
Full sample shape: (K, Na, Nsc, 2) = (16, 16, 64, 2)
```

LSTM/LWM fold each time column into one wideband vector:

```text
t0 column                    t1 column                    ...   t15 column

+---------------+            +---------------+                  +---------------+
| sc0:  Na x 2  |            | sc0:  Na x 2  |                  | sc0:  Na x 2  |
| sc1:  Na x 2  |            | sc1:  Na x 2  |                  | sc1:  Na x 2  |
| ...           |            | ...           |                  | ...           |
| sc63: Na x 2  |            | sc63: Na x 2  |                  | sc63: Na x 2  |
+---------------+            +---------------+                  +---------------+
       |                            |                                  |
       v                            v                                  v
   x_t0: 2048                  x_t1: 2048                         x_t15: 2048
```

Channel token sequence:

```text
+--------+--------+--------+--------+--------+
| x_t0   | x_t1   | x_t2   |  ...   | x_t15  |
| 2048   | 2048   | 2048   |        | 2048   |
+--------+--------+--------+--------+--------+

shape: (K, Na*Nsc*2) = (16, 2048)

LSTM/LWM encoder output:

+--------+--------+--------+--------+--------+
| c_t0   | c_t1   | c_t2   |  ...   | c_t15  |
| 256    | 256    | 256    |        | 256    |
+--------+--------+--------+--------+--------+

shape: (K, D) = (16, 256)
```

Image input and image embedding:

```text
image_seq:

+------------+------------+------------+--------+------------+
| img0       | img1       | img2       |  ...   | img7       |
| 3x224x224  | 3x224x224  | 3x224x224  |        | 3x224x224  |
+------------+------------+------------+--------+------------+

shape: (T_img, 3, 224, 224) = (8, 3, 224, 224)

after ResNet18 token encoder, each frame becomes a 7x7 token grid:

+-----+-----+-----+-----+-----+-----+-----+
| 256 | 256 | 256 | 256 | 256 | 256 | 256 |
+-----+-----+-----+-----+-----+-----+-----+
| 256 | 256 | 256 | 256 | 256 | 256 | 256 |
+-----+-----+-----+-----+-----+-----+-----+
| ... | ... | ... | ... | ... | ... | ... |
+-----+-----+-----+-----+-----+-----+-----+

shape per frame: (49, 256)
shape for 8 frames: (T_img, 49, D) = (8, 49, 256)
```

The frame summarizer compresses each image frame to one RGB token, then time
alignment maps `T_img=8` frame tokens onto the `K=16` channel time grid:

```text
img0 49 tokens -> r_img0: 256
img1 49 tokens -> r_img1: 256
...
img7 49 tokens -> r_img7: 256

frame_tokens shape: (T_img, D) = (8, 256)

after image_time_offsets alignment:

+--------+--------+--------+--------+--------+
| r_t0   | r_t1   | r_t2   |  ...   | r_t15  |
| 256    | 256    | 256    |        | 256    |
+--------+--------+--------+--------+--------+

rgb_tokens shape: (K, D) = (16, 256)
```

Fusion sees a two-row modality grid at every channel-history time step:

```text
              t0        t1        t2       ...      t15
          +--------+--------+--------+--------+--------+
channel   | c_t0   | c_t1   | c_t2   |  ...   | c_t15  |
          | 256    | 256    | 256    |        | 256    |
          +--------+--------+--------+--------+--------+
RGB       | r_t0   | r_t1   | r_t2   |  ...   | r_t15  |
          | 256    | 256    | 256    |        | 256    |
          +--------+--------+--------+--------+--------+

shape before fusion: (K, M, D) = (16, 2, 256)
M = 2 modalities: channel and RGB
```

At each time step, `PerTimeModalityFusion` attends over the modality axis:

```text
for t0:

+------------+
| c_t0: 256  |
+------------+    attention over M=2 modalities
| r_t0: 256  |  --------------------------------->  f_t0: 256
+------------+

shape: (M, D) = (2, 256) -> (D) = (256)
```

The fused sequence keeps the same time grid:

```text
+--------+--------+--------+--------+--------+
| f_t0   | f_t1   | f_t2   |  ...   | f_t15  |
| 256    | 256    | 256    |        | 256    |
+--------+--------+--------+--------+--------+

fused_tokens shape: (K, D) = (16, 256)
```

The prediction head uses `P=4` future queries to read the `K=16` fused tokens:

```text
future queries:

+--------+--------+--------+--------+
| q_0    | q_1    | q_2    | q_3    |
| 256    | 256    | 256    | 256    |
+--------+--------+--------+--------+
        |
        | attend to fused_tokens (16, 256)
        v

+------------+------------+------------+------------+
| frame+1    | frame+2    | frame+3    | frame+4    |
| 16x64x2    | 16x64x2    | 16x64x2    | 16x64x2    |
+------------+------------+------------+------------+

prediction shape: (P, Na, Nsc, 2) = (4, 16, 64, 2)
```

Compressed shape summary:

```text
Channel:
  (K, Na, Nsc, 2) -> (K, 2048) -> (K, 256)

Image:
  (T_img, 3, 224, 224) -> (T_img, 49, 256) -> (T_img, 256) -> (K, 256)

Fusion:
  channel (K, 256) + RGB (K, 256) -> (K, 2, 256) -> (K, 256)

Prediction:
  (K, 256) -> (P, Na, Nsc, 2)
```

## 7. Per-Time Modality Fusion for LSTM/LWM

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

## 8. Prediction Head

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

## 9. End-to-End Flow for Current LSTM/LWM Multimodal

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

## 10. LWM-Temporal and Chiron

`lwm_temporal` and `chiron` were already wideband/patch-time models and were not
rewritten in this update.

The diagrams in this section use the current runner defaults: `K=16`, `P=4`,
`Na=16`, `Nsc=64`, `T_img=8`, and `D=256`. The x-axis is time `t`; the y-axis is
subcarrier or subcarrier band. Unlike the current LSTM/LWM path, these models do
not fold a full time column into one token. They split each channel frame into
antenna-subcarrier patches and then build a temporal patch-token sequence.

### LWM-Temporal

Sources:

- `multimodal_code_index/models/lwm_temporal.py`
- `multimodal_code_index/models/lwm_temporal_multimodal.py`

It appends masked future channel frames and processes time plus
antenna-subcarrier patches jointly. Image information is summarized into a
scene context and injected through the LWM temporal model's CLS path.

#### LWM-Temporal Shape Embedding View

LWM-Temporal first appends `P=4` blank future frames. The future frames are
masked, but they still occupy patch-token positions so the reconstruction head
can emit the future channel values.

```text
                                      time t ->

                  observed history frames                         masked future frames
subcarrier      t0        t1        ...       t15          t16       t17       t18       t19
sc00-15       [A0-A3]   [A0-A3]              [A0-A3]      [mask]    [mask]    [mask]    [mask]
sc16-31       [A0-A3]   [A0-A3]              [A0-A3]      [mask]    [mask]    [mask]    [mask]
sc32-47       [A0-A3]   [A0-A3]              [A0-A3]      [mask]    [mask]    [mask]    [mask]
sc48-63       [A0-A3]   [A0-A3]              [A0-A3]      [mask]    [mask]    [mask]    [mask]

A0-A3 means four antenna patch groups:
  A0 = ant00-03, A1 = ant04-07, A2 = ant08-11, A3 = ant12-15

Each patch token covers:
  4 antennas x 16 subcarriers x 2 real/imag values = 128 raw values
```

For one frame, the `16 x 64` channel matrix becomes a `4 x 4` patch grid:

```text
antenna patch groups x subcarrier bands:

             sc00-15       sc16-31       sc32-47       sc48-63
ant00-03   patch 0,0     patch 0,1     patch 0,2     patch 0,3
ant04-07   patch 1,0     patch 1,1     patch 1,2     patch 1,3
ant08-11   patch 2,0     patch 2,1     patch 2,2     patch 2,3
ant12-15   patch 3,0     patch 3,1     patch 3,2     patch 3,3

patches per frame:
  H x W = (16/4) x (64/16) = 4 x 4 = 16
```

The embedding sequence is therefore:

```text
channel_history: (K, Na, Nsc, 2) = (16, 16, 64, 2)
  -> real/imag complex frame sequence
  -> append P blank future frames
  -> (T, Na, Nsc) = (20, 16, 64)

patchify with patch=(4,16):
  per patch: 4 x 16 x 2 = 128
  per frame: 16 patches
  all frames: T x 16 = 20 x 16 = 320 patches

patch tokens before embedding:
  (T*S, patch_dim) = (320, 128)

Linear patch embedding:
  (320, 128) -> (320, 256)

CLS scene injection:
  patch_embeddings (320, 256) + cls_token (1, 256)
  -> sequence (321, 256)
```

Current default shape flow:

```text
channel_history
(B, K=16, Na=16, Nsc=64, 2)
        |
        | real/imag -> complex
        v
complex channel history
(B, 16, 16, 64)
        |
        | append P=4 blank future frames
        v
channel sequence with masked future
(B, T=K+P=20, 16, 64)
        |
        | patchify antenna/subcarrier with patch=(4,16)
        | H = 16/4 = 4, W = 64/16 = 4, S = 16 patches/frame
        v
patch tokens before embedding
(B, T*S=320, patch_dim=4*16*2=128)
        |
        | Linear(128 -> 256) + learned position embedding
        v
patch embeddings
(B, 320, 256)
```

The image path is compressed into one scene token:

```text
image_seq
(B, T_img=8, 3, 224, 224)
        |
        | ImageTokenEncoder per frame
        v
image tokens
(B, T_img*49=392, 256)
        |
        | scene_query cross-attends to image tokens
        v
scene_ctx
(B, 1, 256)
```

The scene context is injected into the CLS token before sparse LWM encoding:

```text
patch embeddings                    scene_ctx
(B, 320, 256)                       (B, 1, 256)
        |                                  |
        |                                  v
        |                         cls_token + scene_ctx
        |                            (B, 1, 256)
        |                                  |
        +----------------+-----------------+
                         v
patch + CLS sequence
(B, 321, 256)
        |
        | Sparse spatio-temporal LWM encoder, depth=6
        | future patch positions are masked
        v
reconstructed patch values
(B, 320, 128)
        |
        | take last P*S = 4*16 = 64 future tokens
        v
future patch tokens
(B, 64, 128)
        |
        | unpatchify
        v
prediction
(B, P=4, Na=16, Nsc=64, 2)
```

Architecture view:

```text
                         image_seq (B,8,3,224,224)
                                   |
                                   v
                         ImageTokenEncoder
                                   |
                                   v
                         image_tokens (B,392,256)
                                   |
                                   v
                         scene_query attention
                                   |
                                   v
                         scene_ctx (B,1,256)
                                   |
                                   v
channel_history             cls_token + scene_ctx
(B,16,16,64,2)                       |
        |                            |
        v                            |
append P blank future                |
(B,20,16,64)                         |
        |                            |
        v                            |
patchify (4,16)                      |
(B,320,128)                          |
        |                            |
        v                            |
Linear 128 -> 256                    |
(B,320,256)                          |
        |                            |
        +-------------+--------------+
                      v
             patch tokens + CLS
                (B,321,256)
                      |
                      v
       Sparse Spatio-Temporal LWM encoder
                      |
                      v
        reconstruction (B,320,128)
                      |
                      v
          future patches (B,64,128)
                      |
                      v
          prediction (B,4,16,64,2)
```

### Chiron

Sources:

- `multimodal_code_index/models/chiron_channel.py`
- `multimodal_code_index/models/chiron_multimodal.py`

Chiron converts every channel frame into antenna-subcarrier patch tokens,
processes them with temporal and spatial Chiron blocks, and decodes future
frames through `ChannelPredictionHead`. In multimodal mode, Chiron keeps its
existing image-sequence encoder and gated cross-modal fusion path.

#### Chiron Shape Embedding View

Chiron does not append blank future frames in the channel encoder. It embeds the
`K=16` observed history frames and lets `P=4` learnable future queries decode the
future frames from those encoded tokens.

```text
                           time t ->

subcarrier       t0             t1             ...            t15
sc00-31       [A0-A3]        [A0-A3]                        [A0-A3]
sc32-63       [A0-A3]        [A0-A3]                        [A0-A3]

A0-A3 means four antenna patch groups:
  A0 = ant00-03, A1 = ant04-07, A2 = ant08-11, A3 = ant12-15

Each patch token covers:
  4 antennas x 32 subcarriers x 2 real/imag values = 256 raw values
```

For one frame, the `16 x 64` channel matrix becomes a `4 x 2` patch grid:

```text
antenna patch groups x subcarrier bands:

             sc00-31       sc32-63
ant00-03   patch 0,0     patch 0,1
ant04-07   patch 1,0     patch 1,1
ant08-11   patch 2,0     patch 2,1
ant12-15   patch 3,0     patch 3,1

patches per frame:
  H x W = (16/4) x (64/32) = 4 x 2 = 8
```

The embedding sequence is:

```text
channel_history: (K, Na, Nsc, 2) = (16, 16, 64, 2)

PatchEmbed2D with patch=(4,32):
  per patch: 4 x 32 x 2 = 256
  per frame: 8 patches
  all history frames: K x 8 = 16 x 8 = 128 patches

raw patch tokens:
  (K*S, patch_dim) = (128, 256)

Linear + LayerNorm + GELU:
  (128, 256) -> (128, 256)

add temporal and spatial position embeddings:
  temporal_pos: (K, 1, 256)
  spatial_pos:  (1, S, 256)
  channel patch tokens: (K*S, 256) = (128, 256)
```

Inside each `ChironBlock`, the same tokens are viewed two ways:

```text
Temporal view:
  for each spatial patch position s:
  [token(t0,s), token(t1,s), ..., token(t15,s)]
  shape per spatial position: (K, D) = (16, 256)

Spatial view:
  for each time step t:
  [token(t,0), token(t,1), ..., token(t,7)]
  shape per time step: (S, D) = (8, 256)

After ChironBlock x 6:
  channel_tokens: (K*S, D) = (128, 256)
```

Current default shape flow:

```text
channel_history
(B, K=16, Na=16, Nsc=64, 2)
        |
        | per-frame patch embedding with patch=(4,32)
        | H = 16/4 = 4, W = 64/32 = 2, S = 8 patches/frame
        v
patch tokens per frame
(B*K, S=8, 256)
        |
        | reshape and add temporal/spatial position embeddings
        v
channel patch sequence
(B, K*S=128, 256)
        |
        | ChironBlock x 6
        |   TemporalBlock over K
        |   SpatialBlock over S
        |   GatedFFN
        v
channel_tokens
(B, 128, 256)
```

The Chiron image path keeps a full image-token sequence:

```text
image_seq
(B, T_img=8, 3, 224, 224)
        |
        | ResNet18 ImageTokenEncoder
        v
frame spatial tokens
(B, 8, 49, 256)
        |
        | ImageSequenceEncoder
        |   add frame position
        |   temporal attention over frame summaries
        |   inject frame context back into spatial tokens
        v
image_tokens
(B, T_img*49=392, 256)
```

Fusion and prediction:

```text
channel_tokens                         image_tokens
(B, 128, 256)                          (B, 392, 256)
        |                                    |
        | Q                                  | K/V
        +----------------+-------------------+
                         v
             GatedCrossModalFusion x 3
                         |
                         v
              fused channel tokens
                 (B, 128, 256)
                         |
                         | P=4 future queries attend to all 128 tokens
                         v
              ChannelPredictionHead
                         |
                         v
              prediction
              (B, 4, 16, 64, 2)
```

Architecture view:

```text
channel_history (B,16,16,64,2)              image_seq (B,8,3,224,224)
        |                                                   |
        v                                                   v
PatchEmbed2D patch=(4,32)                         ResNet18 ImageTokenEncoder
        |                                                   |
        v                                                   v
(B*16,8,256)                                  frame tokens (B,8,49,256)
        |                                                   |
        v                                                   v
reshape + temporal/spatial pos                 ImageSequenceEncoder
        |                                      temporal frame context
        v                                                   |
channel patch tokens                                        v
(B,128,256)                                      image_tokens (B,392,256)
        |                                                   |
        v                                                   |
ChironBlock x 6                                            |
        |                                                   |
        v                                                   |
channel_tokens (B,128,256)                                  |
        |                                                   |
        +------------------ Q attends to K/V ----------------+
                             |
                             v
                   GatedCrossModalFusion x 3
                             |
                             v
                   fused_tokens (B,128,256)
                             |
                             v
                   ChannelPredictionHead
                             |
                             v
                   prediction (B,4,16,64,2)
```

## 11. Experiment Implications

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

## 12. Runner Defaults

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

## 13. Source Files

- `multimodal_code_index/models/lstm_multimodal.py` - wideband-time LSTM
- `multimodal_code_index/models/lwm_multimodal.py` - wideband-time Transformer
- `multimodal_code_index/models/fusion_blocks.py` - sensor summarizer and per-time fusion
- `multimodal_code_index/models/image_encoders.py` - ResNet18 image token encoder
- `multimodal_code_index/models/chiron_channel.py` - shared prediction head
- `multimodal_code_index/train_multimodal4.py` - model builder and batch routing
- `dataset_loader.py` - channel/RGB sample construction and image time offsets
