# Project Completion Checklist

Status legend: **Done** (implemented and executed), **Partial**
(implemented, not fully executed/verified), **Pending** (not started),
**Blocked** (cannot be completed here — reason given).

Priority legend: **C** critical, **I** important, **O** optional.

---

## Phase 1 — Audit

| # | Item | Pri | Status | Evidence |
| --- | --- | --- | --- | --- |
| 1.1 | Read every source file and test in the repository | C | Done | `FINAL_AUDIT.md` §1 |
| 1.2 | Verify which artifacts already existed (data, checkpoints, results) | C | Done | `FINAL_AUDIT.md` §2: no data, no checkpoints, no results existed |
| 1.3 | Identify bugs, design issues and missing requirements | C | Done | `FINAL_AUDIT.md` §3 |

## Phase 2 — Critical fixes

| # | Item | Pri | Status | Evidence |
| --- | --- | --- | --- | --- |
| 2.1 | Split is engine-wise, no engine in both train and val | C | Done | `configs/data/fd001_split.json`, `tests/test_preprocessing_contract.py` |
| 2.2 | Scaler fitted on train only | C | Done | `src/datasets/preprocessing.py`, scaler contract test |
| 2.3 | Causal windows, no future information | C | Done | `tests/test_windowing.py`, `tests/test_official_test_protocol.py` |
| 2.4 | Padding never enters attention/pooling | C | Done | padding-invariance tests for all four models |
| 2.5 | Unified metric implementation (RMSE/MAE/bias/R²/NASA) | C | Done | `src/evaluation/metrics.py`, `tests/test_metrics.py` |
| 2.6 | NASA score direction and asymmetry verified by hand | C | Done | hand-computed unit tests |
| 2.7 | Validation metrics computed against uncapped RUL | C | Done | `src/training/trainer.py` docstring + `metrics.json` |
| 2.8 | Reproducibility: seeds, deterministic flags, per-run config | C | Done | `set_seed`, `config.json` per run |
| 2.9 | Checkpoints reloadable without the training script | I | Done | `src/models/factory.py::load_checkpoint`, `tests` |
| 2.10 | Preprocessing extended to the official test set | C | Done | `test_FD001.csv`, `fd001_test_rul.csv` |

## Phase 3 — Data preparation

| # | Item | Pri | Status | Evidence |
| --- | --- | --- | --- | --- |
| 3.1 | Obtain the real NASA FD001 files | C | Done | `data/raw/` (checksums in `FINAL_AUDIT.md`) |
| 3.2 | Run `src.datasets.split` | C | Done | split JSON already frozen in the repo; `split.py` refuses to overwrite |
| 3.3 | Run `src.datasets.preprocessing` | C | Done | 17 features, 80/20 engines, test set scaled |
| 3.4 | Run `src.datasets.validation` | C | Done | 60 validation endpoints |
| 3.5 | Window generation sanity checks | C | Done | `python -m src.datasets.windowing` |

## Phase 4 — Testing

| # | Item | Pri | Status | Evidence |
| --- | --- | --- | --- | --- |
| 4.1 | Existing test suite runs and passes | C | Done | `python -m pytest -q` → **104 passed** (see `FINAL_AUDIT.md` §5) |
| 4.2 | Metric unit tests with hand-computed values | C | Done | `tests/test_metrics.py` |
| 4.3 | Leakage tests on real artifacts | C | Done | `tests/test_preprocessing_contract.py` |
| 4.4 | Official test protocol tests | C | Done | `tests/test_official_test_protocol.py` |
| 4.5 | Input-validation / demo inference tests | I | Done | `tests/test_inference_pipeline.py` |
| 4.6 | Attention ablation equivalence tests | I | Done | `tests/test_attention_ablation.py` |
| 4.7 | Automated Streamlit UI test | O | Pending | app driven manually in a real browser (Prediction → 122.9 cycles for test engine 1); no headless UI test is automated in `tests/` |

## Phase 5 — Experiments

| # | Item | Pri | Status | Evidence |
| --- | --- | --- | --- | --- |
| 5.1 | Random Forest baseline | C | Done | `experiments/runs/random_forest_cap125_seed42` (val RMSE 18.55) |
| 5.2 | LSTM seeds 42/123/3407 | C | Done | 13.68 / 13.12 / 14.32 |
| 5.3 | Transformer seeds 42/123/3407 | C | Done | 15.14 / 14.26 / 13.23 |
| 5.4 | CNN-BiLSTM-Attention seeds 42/123/3407 | C | Done | 14.77 / 14.47 / 15.72 |
| 5.5 | Attention ablation seeds 42/123/3407 | C | Done | 15.28 / 15.52 / 15.61 (tag `ablation`) |
| 5.6 | Hyper-parameter study (LR, hidden size) | I | Done | 4 runs, tag `hyperparameter` |
| 5.7 | Data strategy study (RUL cap, window length) | I | Done | 4 runs, tag `data_strategy` |
| 5.8 | Noise augmentation study | O | Pending | not run (out of scope for the available compute budget) |

## Phase 6 — Official evaluation

| # | Item | Pri | Status | Evidence |
| --- | --- | --- | --- | --- |
| 6.1 | Test pipeline: load, validate, scale, window, predict, align | C | Done | `src/evaluation/test_evaluation.py` |
| 6.2 | Model selection documented (validation only) | C | Done | `experiments/results/official_test/model_selection.json` |
| 6.3 | Test predictions saved per model | C | Done | `experiments/results/official_test/*_predictions.csv` |
| 6.4 | Test metrics saved per model | C | Done | `*_metrics.json`, `test_metrics.csv` |
| 6.5 | Test figures | I | Done | `experiments/figures/official_test/` |

## Phase 7 — Error analysis

| # | Item | Pri | Status | Evidence |
| --- | --- | --- | --- | --- |
| 7.1 | Actual vs predicted RUL | C | Done | `experiments/figures/error_analysis/actual_vs_predicted_all_models.png` |
| 7.2 | Residual distribution | C | Done | `residual_distribution.png` |
| 7.3 | Error vs actual RUL | C | Done | `error_vs_actual_rul.png` |
| 7.4 | Error by RUL horizon | C | Done | `error_by_horizon.png`, `metrics_by_horizon.csv` |
| 7.5 | Worst 10 predictions | C | Done | `worst_10_predictions.csv` |
| 7.6 | Engine-level analysis | I | Done | `engine_level_errors.png` |
| 7.7 | Model comparison | C | Done | `model_comparison.png` |

## Phase 8 — Demo

| # | Item | Pri | Status | Evidence |
| --- | --- | --- | --- | --- |
| 8.1 | Streamlit app with all required sections | C | Done | `demo/app.py` |
| 8.2 | Only available checkpoints are offered | C | Done | `available_checkpoints()` reads the registry |
| 8.3 | CSV validation with clear error messages | C | Done | `validate_raw_frame` + UI error path |
| 8.4 | App actually launched and exercised | C | Done | launched on :8501, Prediction page returned 122.9 cycles for test engine 1; error cases rejected — see `FINAL_AUDIT.md` §6 |
| 8.5 | Screenshots captured | O | Pending | no screenshots taken |

## Phase 9 — Documentation

| # | Item | Pri | Status | Evidence |
| --- | --- | --- | --- | --- |
| 9.1 | README complete | C | Done | `README.md` |
| 9.2 | Technical report draft | C | Done | `report/report.md` (all results tables filled from real runs) |
| 9.3 | Presentation draft (15 min) | I | Done | `report/presentation.md` |
| 9.4 | Experiment summary document | I | Done | `EXPERIMENTS_SUMMARY.md` |
| 9.5 | Contribution log templates | O | Done | `contributions/` |
| 9.6 | PDF export of the report | O | Pending | markdown source only, see report header |

## Phase 10 — Final audit

| # | Item | Pri | Status | Evidence |
| --- | --- | --- | --- | --- |
| 10.1 | Full pytest suite green | C | Done | `104 passed in 9.49s` (`FINAL_AUDIT.md` §5) |
| 10.2 | Repository structure clean, `.gitignore` complete | C | Done | `.gitignore`, `README.md` §24 |
| 10.3 | Known limitations documented | C | Done | `README.md` §26, `FINAL_AUDIT.md` §8 |
| 10.4 | Final ZIP archive | O | Pending | produced manually outside the repository |
