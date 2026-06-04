# Experiment Results

This document records the current 16-to-4 channel prediction results for the
`wireless-dataset`.

Updated: 2026-06-05 KST

## Result Scope

The results below use one consolidated visualization pipeline:

```text
multimodal_code_index/run_multimodal_16to4/outputs/figures/all_experiments_overview/all_experiments_overview.ipynb
```

Source histories:

```text
multimodal_code_index/run_multimodal_16to4/outputs/checkpoints/*_history.json
```

Exported GitHub figures and compact summary:

```text
docs/images/results/wideband_all_*.png
docs/images/results/wideband_all_summary_epoch20.csv
```

## Wideband Filter

The LSTM/LWM rows in this page use only the current wideband-time embedding.
The filter is checked from checkpoint weight shape:

```text
current LSTM/LWM wideband input: Na * Nsc * 2 = 16 * 64 * 2 = 2048
legacy per-subcarrier input:    Na * 2       = 16 * 2      = 32
```

Included runs:

- `lstm` and `lwm`: only checkpoints with input dimension `2048`
- `lwm_temporal` and `chiron`: patch-time models, included normally

Excluded runs:

- `8` legacy LSTM/LWM histories with input dimension `32`

All plots and tables below use this filtered run set with a common
`epoch <= 20` comparison window.

## Metric Notes

Lower NMSE is better. Because NMSE is reported in dB, more negative values are
better.

`train_loss` and `val_loss` are plain MSE values on normalized real/imag channel
tensors:

```text
MSE = mean((pred - target)^2)
```

The logged NMSE is target-power-normalized and then converted to dB:

```text
NMSE(dB) = 10 * log10(mean_b(||pred_b - target_b||^2 / (||target_b||^2 + eps)) + eps)
```

For final model comparison, use validation NMSE as the primary metric.

## Visual Results

### Best Validation NMSE

![Wideband-filtered best validation NMSE](docs/images/results/wideband_all_best_val_nmse.png)

### Validation NMSE by Epoch

![Wideband-filtered validation NMSE by epoch](docs/images/results/wideband_all_val_nmse_by_epoch.png)

### Validation Loss by Epoch

![Wideband-filtered validation loss by epoch](docs/images/results/wideband_all_val_loss_by_epoch.png)

### Train and Validation NMSE by Epoch

![Wideband-filtered train and validation NMSE by epoch](docs/images/results/wideband_all_train_val_nmse_by_epoch.png)

### Train and Validation Loss by Epoch

![Wideband-filtered train and validation loss by epoch](docs/images/results/wideband_all_train_val_loss_by_epoch.png)

### Train-Validation Gap

![Wideband-filtered train-validation NMSE gap](docs/images/results/wideband_all_train_val_gap.png)

## Best Result by Scenario

| Scenario | Best mode | Best model | Best epoch | Best validation NMSE |
|---|---|---|---:|---:|
| `sc01` | `channel_only` | `lwm_temporal` | 20 | `-51.540 dB` |
| `sc03` | `channel_only` | `chiron` | 19 | `-38.598 dB` |
| `sc04` | `channel_only` | `lwm_temporal` | 19 | `-37.048 dB` |
| `sc08` | `channel_only` | `chiron` | 20 | `-45.283 dB` |

## Compact Run Table

Values are computed through epoch 20. `wideband_time` means the LSTM/LWM
checkpoint input dimension is `2048`.

| Scenario | Mode | Model | Embedding | Best epoch | Best val NMSE | Final val NMSE | Val loss @ best | Gap @ best |
|---|---|---|---|---:|---:|---:|---:|---:|
| `sc01` | `channel_only` | `chiron` | `patch_time` | 17 | `-46.018 dB` | `-45.995 dB` | `8.761e-04` | `3.126 dB` |
| `sc01` | `channel_only` | `lstm` | `wideband_time` | 3 | `-34.676 dB` | `-28.968 dB` | `1.298e-03` | `-1.729 dB` |
| `sc01` | `channel_only` | `lwm` | `wideband_time` | 5 | `-29.577 dB` | `-28.968 dB` | `1.701e-03` | `-1.872 dB` |
| `sc01` | `channel_only` | `lwm_temporal` | `patch_time` | 20 | `-51.540 dB` | `-51.540 dB` | `5.251e-04` | `15.621 dB` |
| `sc01` | `multimodal` | `chiron` | `patch_time` | 5 | `-37.423 dB` | `-35.473 dB` | `8.773e-04` | `6.078 dB` |
| `sc01` | `multimodal` | `lstm` | `wideband_time` | 13 | `-31.856 dB` | `-29.663 dB` | `1.805e-03` | `4.288 dB` |
| `sc03` | `channel_only` | `chiron` | `patch_time` | 19 | `-38.598 dB` | `-38.539 dB` | `4.413e-05` | `-1.691 dB` |
| `sc03` | `channel_only` | `lstm` | `wideband_time` | 17 | `-35.359 dB` | `-35.064 dB` | `1.067e-04` | `-0.785 dB` |
| `sc03` | `channel_only` | `lwm` | `wideband_time` | 1 | `-27.792 dB` | `-24.386 dB` | `8.955e-04` | `-3.782 dB` |
| `sc04` | `channel_only` | `chiron` | `patch_time` | 20 | `-34.458 dB` | `-34.458 dB` | `1.054e-02` | `17.733 dB` |
| `sc04` | `channel_only` | `lstm` | `wideband_time` | 1 | `-27.798 dB` | `-23.624 dB` | `1.012e-02` | `2.085 dB` |
| `sc04` | `channel_only` | `lwm` | `wideband_time` | 5 | `-26.509 dB` | `-23.624 dB` | `1.040e-02` | `6.013 dB` |
| `sc04` | `channel_only` | `lwm_temporal` | `patch_time` | 19 | `-37.048 dB` | `-36.608 dB` | `1.076e-02` | `24.971 dB` |
| `sc04` | `multimodal` | `chiron` | `patch_time` | 20 | `-29.118 dB` | `-29.118 dB` | `1.053e-02` | `19.338 dB` |
| `sc08` | `channel_only` | `chiron` | `patch_time` | 20 | `-45.283 dB` | `-45.283 dB` | `2.484e-04` | `-8.551 dB` |
| `sc08` | `channel_only` | `lstm` | `wideband_time` | 20 | `-40.600 dB` | `-40.600 dB` | `5.494e-04` | `-6.330 dB` |
| `sc08` | `channel_only` | `lwm` | `wideband_time` | 20 | `-41.823 dB` | `-41.823 dB` | `3.299e-04` | `-6.269 dB` |

The machine-readable compact table is also available at:

```text
docs/images/results/wideband_all_summary_epoch20.csv
```

## Coverage and Missing Runs

Current filtered coverage:

| Scenario | Mode | Available models |
|---|---|---|
| `sc01` | `channel_only` | `lstm`, `lwm`, `lwm_temporal`, `chiron` |
| `sc01` | `multimodal` | `lstm`, `chiron` |
| `sc03` | `channel_only` | `lstm`, `lwm`, `chiron` |
| `sc04` | `channel_only` | `lstm`, `lwm`, `lwm_temporal`, `chiron` |
| `sc04` | `multimodal` | `chiron` |
| `sc08` | `channel_only` | `lstm`, `lwm`, `chiron` |

Not yet present in the filtered result set:

- `sc03` channel-only `lwm_temporal`
- `sc08` channel-only `lwm_temporal`
- `sc01` multimodal `lwm`
- `sc01` multimodal `lwm_temporal`
- `sc04` multimodal `lstm`/`lwm` wideband histories
- `sc04` multimodal `lwm_temporal`

## Interpretation

- The old per-subcarrier LSTM/LWM results are intentionally excluded here.
- Under the current wideband filter, `lwm_temporal` is strongest on `sc01` and
  `sc04`, while `chiron` is strongest on `sc03` and `sc08`.
- Current multimodal results do not beat the best channel-only result in the
  scenarios where both are available.
