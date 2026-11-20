# Experiments Summary — FD001 RUL Prediction

All numbers below come from `experiments/registry.csv` and
`experiments/results/`. Every row has a run directory with a checkpoint,
configuration, training history, predictions and metrics.

* **Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6 GB), CUDA 12.6,
  torch 2.14.1+cu126, Python 3.13.0.
* **Data:** NASA C-MAPSS FD001 (SHA-256 in `FINAL_AUDIT.md` §7).
* **Protocol:** 80/20 engine-wise split (seed 42), window 30, RUL cap
  125, Adam 1e-3, batch 64, ≤60 epochs, early stopping patience 10.
* **Total runs:** 22 registered (1 smoke, 1 baseline, 9 main, 3 ablation,
  4 hyper-parameter, 4 data-strategy).

---

## 1. Main results — validation (3 seeds)

| Model | Val RMSE mean | Std | Min | Max |
| --- | --- | --- | --- | --- |
| LSTM | **13.71** | 0.60 | 13.12 | 14.32 |
| Transformer | 14.21 | 0.96 | 13.23 | 15.14 |
| CNN-BiLSTM-Attention | 14.98 | 0.65 | 14.47 | 15.72 |
| Random Forest | 18.55 | — | — | — |

## 2. Main results — official test

| Model | RMSE | MAE | Bias | R² | NASA sum | NASA mean |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer | **15.54** | **11.96** | +1.67 | **0.860** | **363.2** | **3.63** |
| LSTM | 16.93 | 12.13 | −3.37 | 0.834 | 454.0 | 4.54 |
| CNN-BiLSTM-Attention | 17.32 | 13.21 | −0.38 | 0.826 | 507.2 | 5.07 |
| Random Forest | 18.35 | 13.79 | +3.07 | 0.805 | 733.9 | 7.34 |

Selection: lowest validation RMSE per model family
(`experiments/results/official_test/model_selection.json`). The test set
was evaluated after selection and never used to choose anything.

## 3. Attention ablation

| Variant | Seeds | Val RMSE mean | Std |
| --- | --- | --- | --- |
| CNN-BiLSTM + attention | 3 | 14.98 | 0.65 |
| CNN-BiLSTM + mean pooling | 3 | 15.47 | 0.17 |

Per-seed differences: −0.51, −1.06, +0.11 (attention − mean pooling).
Conclusion: small positive effect, not conclusive; reported as such.

## 4. Hyper-parameter study (LSTM, seed 42)

| Run | Factor changed | Val RMSE |
| --- | --- | --- |
| `lstm_cap125_seed42` | baseline (LR 1e-3, hidden 64) | **13.68** |
| `lstm_hp_lr0.0005_seed42` | LR → 5e-4 | 14.12 |
| `lstm_hp_lr0.002_seed42` | LR → 2e-3 | 14.69 |
| `lstm_hp_hidden32_seed42` | hidden → 32 | 13.87 |
| `lstm_hp_hidden128_seed42` | hidden → 128 | 14.14 |

## 5. Data strategy study (LSTM, seed 42)

| Run | Factor changed | Val RMSE | Val bias |
| --- | --- | --- | --- |
| `lstm_cap50_seed42` | cap → 50 | 29.96 | −19.36 |
| `lstm_cap100_seed42` | cap → 100 | **11.34** | +0.19 |
| `lstm_cap125_seed42` | baseline cap 125 | 13.68 | +3.41 |
| `lstm_window20_cap125_seed42` | window → 20 | 16.14 | +4.08 |
| `lstm_window50_cap125_seed42` | window → 50 | **11.60** | +3.03 |

The cap-50 result is an artefact of the validation protocol (endpoints at
100/60/30), documented rather than hidden. Cap 100 and window 50 are
single-seed improvements worth re-testing across seeds.

## 6. Cost

| Model | Params | Size (MB) | Train (s) | Device | Latency batch 1 (ms) | Latency batch 100 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | 56,641 | 0.22 | 38.1 | cuda | 2.50 | 2.70 |
| Transformer | 103,105 | 0.39 | 121.6 | cuda | 1.75 | 1.59 |
| CNN-BiLSTM-Attn | 198,209 | 0.76 | 74.3 | cuda | 4.92 | 6.20 |
| Random Forest | 606,408 nodes | 12.0 | 8.9 | cpu | 35.72 | 42.20 |

Latency was measured on one device (CUDA) with identical inputs;
training times differ in device (RF is CPU-only) and that is recorded.

## 7. Artifact map

| What | Where |
| --- | --- |
| Registry | `experiments/registry.csv` |
| Per-run config / history / predictions / metrics / checkpoint | `experiments/runs/<experiment_id>/` |
| Official test predictions and metrics | `experiments/results/official_test/` |
| Error analysis tables | `experiments/results/error_analysis/` |
| Cost tables | `experiments/results/cost/` |
| Markdown tables | `experiments/results/summary_tables/` |
| Figures | `experiments/figures/{eda,official_test,error_analysis,cost}/` |

## 8. Known caveats

* Validation has 60 samples, so validation RMSE is noisy; this is why
  model differences below ~1 cycle are not interpreted.
* Early stopping triggers quickly (best epochs between 2 and 31), so
  runs are 12–41 epochs despite a 60-epoch budget.
* Single seed for the hyper-parameter and data-strategy studies.
* No noise-augmentation study was run.
