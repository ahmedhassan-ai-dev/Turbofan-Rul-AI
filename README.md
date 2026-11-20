# Turbofan Predictive Maintenance — RUL Estimation

Deep learning for **Remaining Useful Life (RUL)** prediction on the NASA
C-MAPSS **FD001** turbofan engine degradation dataset, with a Random
Forest baseline, three deep architectures, an attention ablation, an
official test-set evaluation and a Streamlit demo.

> **Status:** every number in this README is produced by a script in
> this repository and stored under `experiments/`. Nothing is typed by
> hand. See `PROJECT_COMPLETION_CHECKLIST.md` for the item-by-item
> status and `FINAL_AUDIT.md` for what is verified and what is not.

---

## Table of contents

1. [Overview](#1-overview)
2. [Abstract](#2-abstract)
3. [Problem statement](#3-problem-statement)
4. [Motivation](#4-motivation)
5. [Dataset](#5-dataset)
6. [Data preprocessing](#6-data-preprocessing)
7. [RUL definition](#7-rul-definition)
8. [Train / validation / test protocol](#8-train--validation--test-protocol)
9. [Feature selection](#9-feature-selection)
10. [Sliding window](#10-sliding-window)
11. [Models](#11-models)
12. [Training configuration](#12-training-configuration)
13. [Experiments](#13-experiments)
14. [Results](#14-results)
15. [Ablation study](#15-ablation-study)
16. [Error analysis](#16-error-analysis)
17. [Computational cost](#17-computational-cost)
18. [Demo](#18-demo)
19. [Installation](#19-installation)
20. [Running preprocessing](#20-running-preprocessing)
21. [Training commands](#21-training-commands)
22. [Evaluation commands](#22-evaluation-commands)
23. [Running the demo](#23-running-the-demo)
24. [Project structure](#24-project-structure)
25. [Reproducibility](#25-reproducibility)
26. [Limitations](#26-limitations)
27. [Future work](#27-future-work)
28. [References](#28-references)
29. [Team contributions](#29-team-contributions)

---

## 1. Overview

The project predicts how many operating cycles a turbofan engine has
left, using only its recent sensor history. Four models are trained on
the same split and compared:

| Model | Family | Role |
| --- | --- | --- |
| Random Forest | classical ML | lower bound / sanity baseline |
| Random Forest (optimized) | classical ML | same features, hyper-parameters tuned on validation only |
| Stacked LSTM | recurrent | sequence baseline |
| Transformer encoder | attention | global temporal context |
| CNN-BiLSTM + additive attention | hybrid | local patterns + bidirectional context + temporal weighting |

A controlled **attention ablation** (identical CNN + BiLSTM, mean
pooling instead of attention) tests whether attention contributes.

## 2. Abstract

Remaining Useful Life estimation is the core quantity of predictive
maintenance. We frame it as supervised regression on run-to-failure
sensor trajectories from the FD001 sub-set of the NASA C-MAPSS
benchmark. Training and validation are split **by engine**, the
standardizer is fitted **on the training engines only**, and every
input window is **strictly causal** — it contains the current cycle and
the past, never the future. Models are selected on validation RMSE and
then evaluated exactly once on the official truncated test set using
the labels distributed with the dataset. We report RMSE, MAE, bias, R²
and the asymmetric NASA scoring function, together with per-horizon
error analysis, an attention ablation and a computational cost
benchmark. A Streamlit application exposes the trained models for
interactive inference.

## 3. Problem statement

Given a multivariate sensor sequence
`x_{t-T+1..t}` of an engine, predict
`RUL_t = t_failure - t`, the number of cycles remaining before failure.
The task is hard because (i) degradation is slow and noisy, (ii) the
first part of a trajectory carries almost no signal about the exact
remaining life, and (iii) the cost of the two error directions is
asymmetric.

## 4. Motivation

* **Safety:** an engine that fails in service is a safety event.
* **Cost:** replacing a healthy module wastes a serviceable part.
* **Asymmetry:** the benchmark's scoring function penalises *late*
  predictions (over-estimating remaining life) more than early ones, so
  a model that is merely accurate on average is not enough.

## 5. Dataset

* **Source:** NASA Prognostics Center of Excellence (PCoE),
  *Turbofan Engine Degradation Simulation Data Set*
  (A. Saxena & K. Goebel, NASA Ames Prognostics Data Repository).
* **Sub-set used:** **FD001** — one operating condition, one fault mode
  (HPC degradation), 100 training engines, 100 test engines.
* **Files:** `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`.

The data set is **not** redistributed in this repository
(`data/raw/` is git-ignored). Download it from the NASA PCoE archive
and place the three files in `data/raw/`.

Raw shape per file: 26 whitespace-separated columns —

| Columns | Content |
| --- | --- |
| 1 | engine unit number |
| 2 | cycle (time, in operational cycles) |
| 3–5 | three operational settings |
| 6–26 | 21 sensor measurements |

Verified row counts in this repository:

| File | Rows | Engines |
| --- | --- | --- |
| `train_FD001.txt` | 20,631 | 100 |
| `test_FD001.txt` | 13,096 | 100 (truncated) |
| `RUL_FD001.txt` | 100 | one label per test engine |

SHA-256 checksums of the exact files used in this repository are
recorded in `FINAL_AUDIT.md`. `data/processed/fd001_metadata.json`
stores the shapes, the selected feature list and the scaler file name.

## 6. Data preprocessing

Implemented in `src/datasets/preprocessing.py`.

1. Parse whitespace-separated files with explicit column names.
2. Validate: schema, no missing values, positive ids/cycles, unique
   `(engine_id, cycle)`, cycles strictly consecutive and starting at 1.
3. **Feature selection:** drop channels that are constant in the
   *training* split (7 of 24 → 17 features remain).
4. **RUL labels:** `max_cycle - cycle` for training/validation
   trajectories only.
5. **Scaling:** `StandardScaler` fitted on training engines only,
   applied unchanged to validation and test.
6. Persist `train_FD001.csv`, `val_FD001.csv`, `test_FD001.csv`,
   `fd001_test_rul.csv`, the fitted scaler and a metadata JSON.

No row-wise shuffling and no resampling happen anywhere.

## 7. RUL definition

For a complete training trajectory:

```
RUL(engine, cycle) = max_cycle(engine) - cycle
```

For the official test trajectories the data is truncated, so the
remaining life **cannot** be recomputed from the file — the labels in
`RUL_FD001.txt` are used (row *i* ↔ engine *i*). The value is the
remaining life at the **last recorded cycle** of that trajectory.

### RUL capping

Long-horizon RUL is not identifiable from the sensors, so the standard
practice caps the training target at **125 cycles**
(`prepare_training_targets`). Capping is applied to *training targets
only*; validation and test metrics are computed against the **uncapped**
true RUL so that different caps stay comparable. The cap value is
recorded in every run's `config.json` and in the registry.

## 8. Train / validation / test protocol

| Split | Engines | Rows | Labels |
| --- | --- | --- | --- |
| Train | 80 | 16,561 windows | `max_cycle - cycle`, capped at 125 |
| Validation | 20 | 60 endpoints | true RUL ∈ {100, 60, 30} |
| Official test | 100 | 100 windows | `RUL_FD001.txt` |

* The split is **engine-wise** (`GroupShuffleSplit`, seed 42, 20 %
  validation) and frozen in `configs/data/fd001_split.json`. No engine
  appears in both training and validation.
* **Validation endpoints** (`configs/data/fd001_validation_endpoints.csv`)
  are three deterministic cut-offs per validation engine, at exactly
  100, 60 and 30 cycles before its recorded failure. They are used for
  early stopping and model selection only.
* The **official test set** is untouched until the final evaluation. It
  is used once, after the model has been selected on validation.
* Model selection rule: lowest validation RMSE among the `main` runs.

## 9. Feature selection

Seven channels are constant on the training engines and are removed:
`setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`,
`sensor_18`, `sensor_19`.

The 17 retained features:

```
setting_1, setting_2,
sensor_2, sensor_3, sensor_4, sensor_6, sensor_7, sensor_8,
sensor_9, sensor_11, sensor_12, sensor_13, sensor_14, sensor_15,
sensor_17, sensor_20, sensor_21
```

Selection is computed on the **training split only** and then reused
everywhere, so the feature list itself cannot leak validation/test
information.

## 10. Sliding window

Implemented in `src/datasets/windowing.py`.

* Window length **30** cycles.
* One window per `(engine, cycle)` row for training/validation, so the
  target is the RUL at the window's last cycle.
* Windows never cross engine boundaries.
* Short histories are **left-padded** by repeating the earliest
  *observed* reading of that engine; a boolean `mask`
  (`True` = real observation) travels with every window and every model
  ignores padded positions.
* The official test set uses exactly **one window per engine**, ending
  at the last recorded cycle (`create_last_windows`).

Leakage guards (see `tests/test_windowing.py`,
`tests/test_official_test_protocol.py`):

* changing a future sensor reading never changes an earlier window;
* every model is padding-invariant (tests mutate padded positions);
* the last timestep of every window is always a real observation.

## 11. Models

### 11.1 Random Forest baseline (`src/models/baseline.py`)

Four statistics per feature (`last`, `mean`, `std`, `change` over the
valid part of the window) → 68 features → `RandomForestRegressor`
(150 trees, `min_samples_leaf=3`). Same split, same cap, same metrics.

### 11.2 Stacked LSTM (`src/models/lstm.py`)

2 LSTM layers × 64 hidden units, dropout 0.2, `pack_padded_sequence`
after converting left padding into right padding, regression head
`64 → 32 → 1`.

### 11.3 Transformer encoder (`src/models/transformer.py`)

Linear input projection to `d_model=64`, learned positional embedding
**numbered from the first real reading** (padding does not shift time),
2 encoder layers (4 heads, FFN 128, GELU, pre-norm), `src_key_padding_mask`
so padding never receives attention, masked mean pooling, head
`64 → 32 → 1`.

### 11.4 CNN-BiLSTM-Attention (`src/models/cnn_bilstm_attention.py`)

Two `Conv1d` layers (17→64→64, kernel 3) for local degradation
patterns, a bidirectional LSTM (64 per direction, 2 layers), masked
**additive attention** (`src/models/components/additive_attention.py`)
producing one weight per timestep (`softmax` over valid steps only),
context vector → `128 → 64 → 1`.

### 11.5 Ablation variant (`cnn_bilstm_mean`)

Identical CNN, BiLSTM and head; the attention module is replaced by
**masked mean pooling**. Parameter counts and shapes of all other
submodules are identical by construction (asserted in
`tests/test_attention_ablation.py`).

## 12. Training configuration

| Setting | Value |
| --- | --- |
| Optimizer | Adam (constant LR) |
| Loss | MSE on capped RUL |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Max epochs | 60 |
| Early stopping | patience 10 on validation RMSE |
| Gradient clipping | norm 1.0 |
| Weight decay | 0.0 |
| Seeds | 42, 123, 3407 |
| Model selection | lowest validation RMSE |

`src/train.py` defaults to 3 epochs **for smoke testing only**; final
runs are produced from `configs/experiments/*.json` with 60 epochs.
Every run writes `config.json`, `training_history.csv`,
`best_predictions.csv`, `summary.json`, `metrics.json` and
`best_model.pt`.

## 13. Experiments

| Suite | File | Purpose |
| --- | --- | --- |
| Main | `configs/experiments/main.json` | 3 models × 3 seeds |
| Attention ablation | `configs/experiments/ablation.json` | mean-pooling variant × 3 seeds |
| Hyper-parameters | `configs/experiments/hyperparameters.json` | LSTM, one factor per run |
| Data strategy | `configs/experiments/data_strategy.json` | RUL cap and window length |

All completed runs are recorded in `experiments/registry.csv`
(experiment id, model, split, cap, seed, LR, batch size, epochs, best
epoch, optimizer, loss, training time, validation metrics, checkpoint
and config paths).

## 14. Results

### 14.1 Validation (3 seeds per model, selected on validation only)

| Model | Val RMSE mean | Std | Min | Max | Seeds |
| --- | --- | --- | --- | --- | --- |
| LSTM | **13.71** | 0.60 | 13.12 | 14.32 | 3 |
| Transformer | 14.21 | 0.96 | 13.23 | 15.14 | 3 |
| CNN-BiLSTM-Attention | 14.98 | 0.65 | 14.47 | 15.72 | 3 |
| Random Forest | 18.55 | — | — | — | 1 |

### 14.2 Official FD001 test set (100 engines, evaluated once)

| Model | RMSE | MAE | Bias | R² | NASA score (sum) | NASA (mean) |
| --- | --- | --- | --- | --- | --- | --- |
| **Transformer** | **15.54** | **11.96** | +1.67 | **0.860** | **363.2** | **3.63** |
| LSTM | 16.93 | 12.13 | −3.37 | 0.834 | 454.0 | 4.54 |
| CNN-BiLSTM-Attention | 17.32 | 13.21 | −0.38 | 0.826 | 507.2 | 5.07 |
| Random Forest | 18.35 | 13.79 | +3.07 | 0.805 | 733.9 | 7.34 |

Bias is `predicted − true`; positive means the model expects more
remaining life than the engine really had (the direction the NASA score
punishes more).

### 14.3 What the numbers say

* **H1 supported:** all three deep models beat the Random Forest on
  validation RMSE (13.71–14.98 vs 18.55) and on the official test
  (15.54–17.32 vs 18.35).
* **Model ranking changes between validation and test.** Validation
  ranks LSTM first; the test ranks Transformer first. The gap between
  the two (≈1.4 cycles RMSE) is comparable to the seed spread
  (Transformer std 0.96), so the two should be treated as close, not as
  a strict ordering.
* **The hybrid is the heaviest and did not win.** CNN-BiLSTM-Attention
  has 3.5× the parameters of the LSTM and finished behind both the LSTM
  and the Transformer on the test set.
* **Sign of the bias differs by model.** Transformer and Random Forest
  predict late (+1.67, +3.07); LSTM predicts early (−3.37), which is the
  safer direction for maintenance.

### 14.4 Per-horizon RMSE on the official test

| Horizon | CNN-BiLSTM-Attn | LSTM | Random Forest | Transformer |
| --- | --- | --- | --- | --- |
| Short (≤ 50) | 12.14 | **6.53** | 11.85 | 7.01 |
| Medium (51–100) | 20.13 | 20.31 | 21.86 | **19.48** |
| Long (> 100) | 18.55 | 20.01 | 19.68 | **17.08** |

All models are far more accurate near failure than at long horizons,
which is the regime that matters operationally.

### 14.5 Hyper-parameter study (LSTM, seed 42, one factor per run)

| Configuration | Val RMSE |
| --- | --- |
| LR 0.0005 | 14.12 |
| LR 0.001 (baseline) | **13.68** |
| LR 0.002 | 14.69 |
| hidden 32 | 13.87 |
| hidden 64 (baseline) | **13.68** |
| hidden 128 | 14.14 |

The default configuration was not beaten by either LR change or by
changing the hidden size; the differences (0.2–1.0 RMSE) are of the same
order as the seed spread.

### 14.6 Data strategy study (LSTM, seed 42, one factor per run)

| Strategy | Val RMSE | Val bias |
| --- | --- | --- |
| RUL cap 50 | 29.96 | −19.36 |
| RUL cap 100 | **11.34** | +0.19 |
| RUL cap 125 (baseline) | 13.68 | +3.41 |
| Window 20 | 16.14 | +4.08 |
| Window 30 (baseline) | 13.68 | +3.41 |
| Window 50 | **11.60** | +3.03 |

* **Cap 50 is not usable with this validation protocol:** the validation
  endpoints are at 100/60/30 cycles, so a model capped at 50 cannot
  reach 100 and accumulates a −19 bias. The comparison shows exactly why
  capping must match the evaluation horizons.
* **Cap 100 and window 50 both improved validation RMSE** (11.34 and
  11.60 vs 13.68). This is a single seed, so it is reported as a
  promising direction, not as a confirmed improvement.

Full tables: `experiments/results/summary_tables/`.

## 15. Ablation study

| Variant | Seeds | Val RMSE mean | Std | Val MAE mean | Mean NASA |
| --- | --- | --- | --- | --- | --- |
| CNN-BiLSTM + attention | 3 | **14.98** | 0.65 | **11.87** | **3.79** |
| CNN-BiLSTM + mean pooling | 3 | 15.47 | 0.17 | 12.08 | 3.97 |

Per-seed difference (attention − mean pooling): **−0.51, −1.06, +0.11**.

**Reading:** attention is better in 2 of 3 seeds with a mean advantage
of ≈0.49 cycles RMSE, which is smaller than the attention variant's own
seed spread (0.65). The honest conclusion is that **attention gives a
small, directionally positive but not statistically conclusive benefit
on FD001** — not that attention is required. The mean-pooling variant is
notably more stable across seeds (std 0.17).

## 16. Error analysis

Produced by `python -m src.evaluation.error_analysis` from the official
test predictions:

* actual vs predicted RUL (per model and combined),
* residual distribution,
* error vs true RUL,
* RMSE by RUL horizon (`Short ≤ 50`, `Medium 51–100`, `Long > 100`),
* worst 10 predictions,
* engine-level error ranking,
* model comparison.

Figures: `experiments/figures/error_analysis/`.
Tables: `experiments/results/error_analysis/`.

## 17. Computational cost

`python -m src.evaluation.cost_analysis` measures parameter count,
checkpoint size, training time (from the registry, with the device used
for each run) and inference latency for batch 1 and batch 100 — all on
the **same device** with identical input shape (T=30, F=17).

| Model | Params | Size (MB) | Train (s) | Device | Latency b=1 (ms) | Latency b=100 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | 56,641 | 0.22 | 38.1 | cuda | 2.50 | 2.70 |
| Transformer | 103,105 | 0.39 | 121.6 | cuda | 1.75 | 1.59 |
| CNN-BiLSTM-Attention | 198,209 | 0.76 | 74.3 | cuda | 4.92 | 6.20 |
| Random Forest | 606,408 nodes | 12.0 | 8.9 | cpu | 35.72 | 42.20 |

Hardware: RTX 4050 Laptop (6 GB), CUDA 12.6, PyTorch 2.14.1. The
Random Forest is CPU-trained and CPU-checkpointed, so its training time
is not directly comparable with the GPU runs; its inference latency was
measured on the same machine as the deep models.

## 18. Demo

Streamlit app (`demo/app.py`): home, input (sample engine or CSV
upload), engine monitoring, model selection, prediction, model
comparison and a limited explainability view.

* only models with an existing checkpoint are offered — that is the
  original and the optimized Random Forest plus the three deep models
  (and the mean-pooling ablation);
* uploaded CSVs are validated and produce a clear error message instead
  of a crash;
* predictions use the same preprocessing and inference code as the
  official evaluation.

Two scripts re-check the whole demo without a browser:

```bash
python scripts/verify_models.py   # every checkpoint loads and predicts
python scripts/verify_gui.py      # headless Streamlit end-to-end test
```

## 19. Installation

```bash
# Python 3.11+ (developed and verified on 3.13)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA builds of PyTorch, install it from the official index, e.g.:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## 20. Running preprocessing

```bash
# 1. place train_FD001.txt, test_FD001.txt, RUL_FD001.txt in data/raw/
python -m src.datasets.split          # creates the frozen split (only once)
python -m src.datasets.preprocessing  # scaler, features, processed CSVs
python -m src.datasets.validation     # validation endpoints (protocol file)
python -m src.datasets.windowing      # window sanity checks
```

`src/datasets/split.py` refuses to overwrite an existing split on
purpose: the split is part of the experimental protocol.

## 21. Training commands

```bash
# Random Forest baseline
python -m src.train_baseline --seed 42 --rul-cap 125

# One deep model
python -m src.train --model lstm --epochs 60 --patience 10 \
    --rul-cap 125 --seed 42 --experiment-id lstm_cap125_seed42

# Smoke test only (NOT a final result)
python -m src.train --model lstm --epochs 2 \
    --experiment-id lstm_smoke --tag smoke

# Whole suites (resumable: completed runs are skipped)
python scripts/run_suite.py configs/experiments/main.json
python scripts/run_suite.py configs/experiments/ablation.json
python scripts/run_suite.py configs/experiments/hyperparameters.json
python scripts/run_suite.py configs/experiments/data_strategy.json
```

## 22. Evaluation commands

```bash
# Official test set for the validation-selected run of every model
python -m src.evaluation.test_evaluation --auto

# Or specific runs
python -m src.evaluation.test_evaluation \
    --run-dir experiments/runs/lstm_cap125_seed42

# Error analysis + cost analysis + summary tables
python -m src.evaluation.error_analysis
python -m src.evaluation.cost_analysis
python scripts/make_summary_tables.py

# Full test suite
python -m pytest -q
```

## 23. Running the demo

```bash
streamlit run demo/app.py
```

Then open the printed local URL. Choose a sample test engine or upload a
CSV with the 26 C-MAPSS columns, pick a model that has a checkpoint, and
press **Estimate RUL**.

## 24. Project structure

```
.
├── configs/
│   ├── data/            # frozen split + validation endpoints
│   ├── experiments/     # experiment suites (JSON)
│   └── models/          # architecture configurations
├── data/
│   ├── raw/             # NASA files (not tracked)
│   └── processed/       # scaled CSVs, scaler, metadata (not tracked)
├── checkpoints/         # Random Forest baseline + optimized checkpoints
├── demo/app.py          # Streamlit application
├── experiments/
│   ├── registry.csv     # every completed run
│   ├── runs/            # per-run artifacts
│   ├── results/         # official test, error analysis, cost, tables
│   └── figures/         # all figures
├── report/              # technical report + presentation
├── scripts/             # suite runner, summary tables
├── src/
│   ├── datasets/        # split, preprocessing, windowing, datamodule
│   ├── models/          # baseline, LSTM, Transformer, hybrid, factory
│   ├── training/        # trainer
│   ├── evaluation/      # metrics, official test, error/cost analysis
│   ├── experiments/     # registry + runner
│   ├── inference/       # predictor + input pipeline
│   └── reporting/       # markdown helpers
└── tests/               # pytest suite
```

## 25. Reproducibility

* Seeds fixed for Python, NumPy and PyTorch; cuDNN deterministic mode on.
* Split, scaler, feature list, architecture config, training history
  and metrics are all persisted per run.
* `experiments/registry.csv` records the exact configuration and the
  device of every run.
* Determinism caveat: GPU kernels are not bit-exact across driver/CUDA
  versions, so a re-run on different hardware may differ slightly even
  with `cudnn.deterministic = True`.

## 26. Limitations

* Single sub-set (FD001: one operating condition, one fault mode); the
  results do not transfer automatically to FD002–FD004.
* Validation uses three fixed cut-offs per engine (60 samples), which is
  a strong smoothing of the validation signal compared with evaluating
  every cycle.
* RUL is capped at 125 cycles, so the models are not calibrated beyond
  that horizon.
* The attention weights are a diagnostic of where the model looks, not
  evidence of physical causation.
* The demo runs on CPU with single-engine input; it is not a
  maintenance-decision system.

## 27. Future work

* Evaluate on FD002/FD004 (multiple operating conditions) with
  condition-aware normalisation.
* Probabilistic RUL (quantile or distributional heads) instead of a
  point estimate.
* Remaining-life estimation with uncertainty-aware maintenance
  scheduling and cost-sensitive training.
* Sequence models with longer context and per-engine fine-tuning.

## 28. References

1. A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation
   Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008.
2. NASA PCoE, *Turbofan Engine Degradation Simulation Data Set*,
   NASA Ames Prognostics Data Repository.
3. F. O. Heimes, *Recurrent Neural Networks for Remaining Useful Life
   Estimation*, PHM 2008.
4. A. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
5. D. Bahdanau, K. Cho, Y. Bengio, *Neural Machine Translation by
   Jointly Learning to Align and Translate*, ICLR 2015.
6. S. Hochreiter, J. Schmidhuber, *Long Short-Term Memory*,
   Neural Computation 1997.
7. L. Breiman, *Random Forests*, Machine Learning 2001.

## 29. Team contributions

See `contributions/` for the individual logs and the combined summary.
