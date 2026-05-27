# Multimodal 16->4 Experiment

This folder is the experiment entrypoint for the current
`multimodal_code_index` models.

Default task:

- input: `K=16` past channel frames plus RGB image history
- target: `P=4` consecutive future channel frames
- prediction: one forward pass returns `(B, 4, 16, Nsc, 2)`
- default `Nsc=64`, selected from the current 512-subcarrier dataset for Task04 compatibility
- current modalities: `channel + RGB image`
- currently not used by this runner: LiDAR, Radar

## Current Defaults

These are the defaults used when you run `run_16to4.py` without overriding CLI
flags.

| Setting | Value |
|---|---|
| data root | `/mnt/ssd_7t_2/carla-wireless-dataset/wireless-dataset` |
| scenario | `sc01` |
| mode | `multimodal` (`--mode channel_only` disables image use) |
| model | `all` (`lstm`, `lwm`, `lwm_temporal`, `chiron`) |
| history length `K` | `16` |
| prediction horizon `P` | `4` future frames, predicted in one forward pass |
| BS antennas `Na` | `16` |
| subcarriers `Nsc` | `64` selected from the 512-subcarrier files |
| subcarrier start / stride | `0` / `1` |
| image frames | `8` latest-past RGB frames |
| image size | `224 x 224` |
| LiDAR / Radar | not used |
| embed dim | `256` |
| fusion layers / heads | `3` / `4` |
| dropout | `0.1` |
| epochs | `50` |
| batch size | `4` |
| optimizer | `AdamW` |
| learning rate | `1e-3` |
| min learning rate | `1e-6` |
| weight decay | `1e-4` |
| warmup epochs | `3` |
| early stopping | none; runs all requested epochs |
| train / val split | `0.75 / 0.25`, chronological |
| normalization stats | all available channel files unless `--stats-max-samples` is set |
| DataLoader workers | `2` |
| AMP | off unless `--amp` is passed |
| AMP dtype | `fp16` |
| pretrained image weights | off by wrapper default; pass `--pretrained-image` to enable |
| output directory | `multimodal_code_index/run_multimodal_16to4/outputs/` |
| default run prefix | `multimodal4_{mode}_sc01_K16_P4_img8` |

Model-specific defaults:

| Model | Main channel stack |
|---|---|
| `lstm` | per-subcarrier LSTM, `3` layers, hidden dim `256` |
| `lwm` | per-subcarrier Transformer, `12` layers, `8` heads, `d_model=64`, `d_ff=256` |
| `lwm_temporal` | sparse spatio-temporal LWM, depth `6`, `8` heads, patch `(4,16)` |
| `chiron` | CHIRON blocks, depth `6`, `4` heads, patch `(4,32)`, conv kernel `7` |

GPU policy:

```text
CUDA_VISIBLE_DEVICES=1,2
```

This is set by default in `run_16to4.py`, `smoke_16to4.py`, and
`train_multimodal4.py` before torch initializes CUDA. Inside PyTorch, physical
GPU 1 appears as `cuda:0`, and physical GPU 2 appears as `cuda:1`.

The current training implementation is single-process/single-device. Therefore,
with the default `--device cuda`, training uses visible `cuda:0`, i.e. physical
GPU 1. To run on physical GPU 2, use:

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py --device cuda:1 ...
```

Using both GPUs simultaneously would require a separate DataParallel/DDP change.
For the current comparison, the simpler option is to run independent model
processes on different visible CUDA devices.

Example two-GPU split:

```bash
mkdir -p multimodal_code_index/run_multimodal_16to4/outputs/logs

python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --mode multimodal --model chiron --device cuda:0 --batch-size 4 --amp \
  > multimodal_code_index/run_multimodal_16to4/outputs/logs/multimodal_chiron_gpu0.log 2>&1 &

python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --mode multimodal --model lwm_temporal --device cuda:1 --batch-size 4 --amp \
  > multimodal_code_index/run_multimodal_16to4/outputs/logs/multimodal_lwm_temporal_gpu1.log 2>&1 &
```

Because each process trains one model, default checkpoint/history/summary names
include both `mode` and `model`.

The same schedule is available as a script:

```bash
multimodal_code_index/run_multimodal_16to4/launch_2gpu_by_model.sh compare
```

`compare` runs `channel_only` first and `multimodal` second. For each mode:

```text
wave 1: cuda:0 = chiron,       cuda:1 = lwm_temporal
wave 2: cuda:0 = lstm,         cuda:1 = lwm
```

You can run only one mode with:

```bash
multimodal_code_index/run_multimodal_16to4/launch_2gpu_by_model.sh channel_only
multimodal_code_index/run_multimodal_16to4/launch_2gpu_by_model.sh multimodal
```

## Quick Commands

Smoke check:

```bash
cd /mnt/ssd_7t_2/carla-wireless-dataset
conda activate hoyun_312
python multimodal_code_index/run_multimodal_16to4/smoke_16to4.py --model lstm --device cuda
```

Quick training run:

```bash
cd /mnt/ssd_7t_2/carla-wireless-dataset
conda activate hoyun_312
python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --model chiron \
  --epochs 5 \
  --batch-size 4 \
  --max-train-samples 1024 \
  --max-val-samples 256
```

Run all four models:

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --model all \
  --epochs 30 \
  --batch-size 4 \
  --amp
```

Use all 512 subcarriers:

```bash
python multimodal_code_index/run_multimodal_16to4/run_16to4.py \
  --model chiron \
  --num-subcarriers 512
```

## Run vs Smoke

`smoke_16to4.py` is a one-batch shape check. It does not train or save a
checkpoint. It loads one batch, runs one forward pass, and asserts:

```text
channel_history: (B, 16, 16, Nsc, 2)
target:          (B, 4, 16, Nsc, 2)
pred:            (B, 4, 16, Nsc, 2)
```

`run_16to4.py` is the training entrypoint. It wraps
`multimodal_code_index/train_multimodal4.py`, builds train/val datasets,
trains the selected model(s), and writes checkpoints/history under:

```text
multimodal_code_index/run_multimodal_16to4/outputs/
```

Default output names include `mode`, so `channel_only` and `multimodal` runs do
not overwrite each other.

## Execution Flow

If you run a multimodal prediction experiment, the code path is:

```text
multimodal_code_index/run_multimodal_16to4/run_16to4.py
  -> multimodal_code_index/train_multimodal4.py
     -> dataset_loader.py
     -> utils.py
     -> multimodal_code_index/models/*_multimodal.py
        -> image_encoders.py / fusion_blocks.py / backbone files
```

## Main Files

### Experiment entrypoints

- `multimodal_code_index/run_multimodal_16to4/run_16to4.py`
  - Thin wrapper for training.
  - Sets defaults for this 16->4 experiment: `K=16`, `P=4`, `Nsc=64`,
    `num_image_frames=8`, `data-root=wireless-dataset`.
  - Runs `multimodal_code_index/train_multimodal4.py`.

- `multimodal_code_index/run_multimodal_16to4/smoke_16to4.py`
  - One-batch verification script.
  - Confirms that the model predicts all four future frames at once.

### Training runner

- `multimodal_code_index/train_multimodal4.py`
  - Actual training loop.
  - Imports dataset, models, and metrics.
  - Builds `ChannelPredictionDataset` for train/val.
  - Builds one of `lstm`, `lwm`, `lwm_temporal`, `chiron`.
  - Computes `MSELoss` for training.
  - Logs `NMSE(dB)` and cosine similarity as metrics.
  - Saves best checkpoints and JSON histories.

Important sections:

- model imports and utility imports: `train_multimodal4.py:39`
- CLI defaults: `train_multimodal4.py:79`
- model builder: `train_multimodal4.py:178`
- dataset builder: `train_multimodal4.py:241`
- forward, loss, NMSE, cosine: `train_multimodal4.py:303`
- main execution flow: `train_multimodal4.py:500`

### Dataset loader

- `dataset_loader.py`
  - Reads `wireless-dataset/scXX/channels/channel_*.npy`.
  - Reads `wireless-dataset/scXX/images/frame_*.png`.
  - Converts complex channel arrays into real/imag tensors.
  - Creates valid samples where `[t-K+1 ... t]` and `[t+1 ... t+P]` all exist.
  - Applies chronological train/val split.
  - Loads latest-past image frames and normalizes them with ImageNet mean/std.

Returned sample fields:

```text
channel_history: (K, Na, Nsc, 2)
target:          (P, Na, Nsc, 2)
image_seq:       (T_img, 3, 224, 224)
image_valid_mask:(T_img,)
sample_index:    scalar t
```

Important sections:

- image loading and normalization: `dataset_loader.py:43`
- channel/image file discovery: `dataset_loader.py:72`
- valid 16->4 sample construction: `dataset_loader.py:346`
- chronological split: `dataset_loader.py:381`
- actual `__getitem__` sample creation: `dataset_loader.py:473`

### Shared utilities

- `utils.py`
  - `nmse(pred, target)`: NMSE in dB.
  - `cosine_similarity(pred, target)`: flattened cosine similarity.
  - `channel_to_real(...)`: complex channel to `[real, imag]`.
  - `normalize_channel(...)`: min-max normalization.
  - `compute_dataset_stats(...)`: per-element min/max stats over channel files.

NMSE formula:

```text
NMSE(dB) = 10 * log10( mean_b( ||pred_b - target_b||^2 / (||target_b||^2 + eps) ) + eps )
```

Implementation location:

- `utils.py:27`

Note: unless `--no-normalize` is passed, NMSE is computed on min-max-normalized
real/imag tensors. For physical-scale NMSE, denormalize `pred` and `target`
before calling `nmse`.

Current result tables and metric interpretation are tracked in
`../../EXPERIMENT_RESULTS.md`.

## Model Files

The selected `--model` decides which model file is used:

- `--model lstm`
  - `multimodal_code_index/models/lstm_multimodal.py`

- `--model lwm`
  - `multimodal_code_index/models/lwm_multimodal.py`

- `--model lwm_temporal`
  - `multimodal_code_index/models/lwm_temporal_multimodal.py`
  - uses `multimodal_code_index/models/lwm_temporal.py`

- `--model chiron`
  - `multimodal_code_index/models/chiron_multimodal.py`
  - uses `multimodal_code_index/models/chiron_channel.py`

Common helper files:

- `multimodal_code_index/models/image_encoders.py`
  - ResNet18 image token encoder.

- `multimodal_code_index/models/fusion_blocks.py`
  - Gated cross-modal fusion blocks.

- `multimodal_code_index/models/sensor_encoders.py`
  - Re-export module used by several multimodal models.

- `multimodal_code_index/models/lidar_encoders.py`
  - Exists, but this runner currently keeps `use_lidar=False`.

## Data Used

Default data root:

```text
/mnt/ssd_7t_2/carla-wireless-dataset/wireless-dataset
```

Default scenario:

```text
sc01
```

Expected per-scenario structure:

```text
wireless-dataset/scXX/
  channels/
  images/
  positions/
  velocities/
  lidar/
  radar/
  metadata.json
```

This runner only consumes:

```text
channels/
images/
```

## Environment Notes

On this machine the base conda `python` did not have torch. The `hoyun_312`
environment was verified with:

```text
torch 2.6.0+cu124
torchvision 0.21.0+cu124
CUDA available
```

Use:

```bash
conda activate hoyun_312
```

By default `run_16to4.py` adds `--no-pretrained-image` so quick runs do not
need ImageNet weight downloads. Add `--pretrained-image` to the wrapper command
if you want the underlying models to load torchvision ImageNet weights.
