# Final Audit — Turbofan RUL Project

This document records what was found in the repository, what was
changed, what was executed and what is still open. Anything marked
**Verified** below was actually run in this environment; the exact
command and outcome are given.

---

## 1. What existed when the audit started

Repository state at `main` (11 commits, last commit
`9e37727 feat(models): integrate CNN BiLSTM with custom attention`):

* `src/datasets/`: `split.py`, `preprocessing.py`, `windowing.py`,
  `datamodule.py`, `validation.py`, `eda.py`, `visualize.py`
* `src/models/`: `baseline.py`, `lstm.py`, `transformer.py`,
  `cnn_bilstm_attention.py`, `components/additive_attention.py`
* `src/training/trainer.py`, `src/train.py`, `src/train_baseline.py`
* `src/evaluation/`: `metrics.py`, `baseline_analysis.py`
* `tests/`: 9 files
* `configs/data/fd001_split.json`, `configs/data/fd001_validation_endpoints.csv`
* everything else: `.gitkeep` only

**Empty or missing:** `README.md` (0 bytes), `.env.example` (0 bytes),
`src/inference/`, `src/explainability/`, `demo/`, `report/`,
`experiments/results/`, `experiments/runs/`, `experiments/registry.csv`,
`configs/experiments/`, `configs/models/`, any trained checkpoint, any
processed data.

## 2. Artifacts that really existed

| Artifact | Existed | Notes |
| --- | --- | --- |
| Raw NASA data | **No** | `data/raw/` empty; downloaded during this work |
| Processed data | **No** | produced during this work |
| Split file | Yes | 80/20 engine-wise, seed 42, 20 validation engines |
| Validation endpoints | Yes | 60 endpoints (3 horizons × 20 engines) |
| Checkpoints | **No** | `checkpoints/` empty |
| Results / figures | **No** | only `.gitkeep` |
| Tests | Yes | 9 files, hard-coded to processed-data sizes |

## 3. Findings

Priority: **C** critical (blocks correctness), **I** important,
**O** polish.

### 3.1 Missing requirements

| # | P | Finding |
| --- | --- | --- |
| F1 | C | No official test pipeline: `test_FD001.txt` and `RUL_FD001.txt` were never read anywhere in the codebase. |
| F2 | C | No inference layer and no demo, so trained models could not be used. |
| F3 | C | No experiment registry, no experiment configs, no multi-seed aggregation, no ablation, no hyper-parameter or data-strategy study. |
| F4 | C | `README.md` and `.env.example` were empty files. |
| F5 | I | No mean-pooling variant of the hybrid model, so the attention ablation described in the project spec was not implementable without duplicating the model. |
| F6 | I | No report, no presentation, no contribution logs. |

### 3.2 Bugs and correctness risks

| # | P | Finding | Fix |
| --- | --- | --- | --- |
| B1 | C | `train_baseline.py` used its own metric implementation (`rmse`, `mae`, `r2` only), so baseline numbers were not comparable with deep-model numbers (no bias, no NASA score). | baseline now calls `src.evaluation.metrics.regression_metrics` |
| B2 | C | `baseline_analysis.py` implemented a *third* metric function; per-horizon `r2` was computed on constant targets (undefined). | analysis now reuses the unified metrics; `r2` returns `None` for constant targets |
| B3 | I | Trainer computed the loss on **capped** targets but validation metrics on **uncapped** labels, with no documentation. Harmless at cap = 125 (validation endpoints are 100/60/30) but silently wrong for smaller caps. | documented in `RULTrainer`; validation always uncapped and this is stated in README/report |
| B4 | I | `set_seed` did not enable `cudnn.deterministic`, did not disable `benchmark`, and the DataLoader generator was never seeded by the trainer. | `set_seed` + `_loader_generator` |
| B5 | C | Checkpoints stored only `model_state_dict` + metrics — not reloadable without hard-coding the architecture. | checkpoint now stores `model_name`, `model_kwargs`, metadata; `load_checkpoint` rebuilds it |
| B6 | I | `config.json` was written *after* training finished, so a crashed run left no configuration on disk. | written before training starts |
| B7 | I | No `metrics.json`, no per-epoch/second timing, no peak GPU memory. | added to `summary.json` / `metrics.json` |
| B8 | I | NASA score had no documented reference and no mean variant, making it ambiguous whether a number was a sum or an average. | formula documented with reference; both `nasa_score` (sum) and `nasa_score_mean` are returned |
| B9 | O | `src/datasets/visualize.py` executed figures at import time (side effects on import) and used cwd-relative paths. | wrapped in `main()`, project-root paths |
| B10 | O | `src/datasets/eda.py` used a cwd-relative path. | project-root path + file check |

### 3.3 Design / process issues

| # | P | Finding | Fix |
| --- | --- | --- | --- |
| D1 | C | `src/train.py` defaulted to 3 epochs, i.e. the default was a smoke test that could be mistaken for a final result. | default kept for smoke only; `--experiment-id`/`--tag` required for real runs, final suites live in `configs/experiments/*.json` with 60 epochs, and the smoke run is tagged `smoke` |
| D2 | I | Architectures were hard-coded inside `build_model`. | `configs/models/*.json` + `src/models/factory.py` |
| D3 | C | `.gitignore` ignored `checkpoints/` entirely, so no model could ever be delivered. | `.gitignore` rewritten; checkpoints and runs are tracked (they are < 2 MB each) |
| D4 | I | `requirements.txt` had no demo dependency and an untested pandas bound. | `streamlit` added; bounds verified against the installed versions |
| D5 | O | Validation is only 60 endpoints (3 cut-offs per engine) — a noisy selection signal. | kept (it is the project protocol) and documented as a limitation |

## 4. What was added

| Area | Files |
| --- | --- |
| Data | `preprocessing.py` extended to the official test set, `windowing.py` gained `create_last_windows` |
| Models | `factory.py`, `cnn_bilstm_attention.py` ablation switch, `configs/models/*.json` |
| Training | `trainer.py` hardened, `src/train.py` rewritten, `src/train_baseline.py` rewritten |
| Experiments | `src/experiments/registry.py`, `src/experiments/runner.py`, `configs/experiments/*.json`, `scripts/run_suite.py` |
| Evaluation | `test_evaluation.py`, `error_analysis.py`, `cost_analysis.py` |
| Inference | `src/inference/predictor.py`, `src/inference/pipeline.py` |
| Demo | `demo/app.py` |
| Reporting | `src/reporting/tables.py`, `scripts/make_summary_tables.py` |
| Tests | `test_metrics.py`, `test_preprocessing_contract.py`, `test_official_test_protocol.py`, `test_inference_pipeline.py`, `test_attention_ablation.py` |
| Docs | `README.md`, this file, `PROJECT_COMPLETION_CHECKLIST.md`, `EXPERIMENTS_SUMMARY.md`, `report/report.md`, `report/presentation.md`, `contributions/*` |

## 5. Execution log

Environment: Windows 11, Python 3.13.0, PyTorch 2.14.1+cu126
(CUDA 12.6) on an NVIDIA GeForce RTX 4050 Laptop GPU (6 GB),
NumPy 2.1.2, pandas 3.0.3, scikit-learn 1.8.0, matplotlib 3.10.9,
Streamlit 1.57.0.

| # | Command | Outcome |
| --- | --- | --- |
| 1 | `python -m src.datasets.preprocessing` | **Passed.** 17 features (7 constant dropped), 80 train / 20 validation engines, 16,561 train rows, 4,070 validation rows, 13,096 test rows, 100 test engines; scaler reproduces mean 0 / std 1 on train. |
| 2 | `python -m pytest -q` | **104 passed** (three failures in the newly added tests were test-side bugs — `state_dict` tensors compared instead of shapes, a hook capturing the wrong tensor, a pandas-3 dtype assignment — and were fixed before the suite went green). |
| 3 | `python -m src.train_baseline --seed 42 --rul-cap 125` | **Completed.** Val RMSE 18.55, MAE 13.35, bias +4.65, R² 0.582, NASA 608.1. |
| 4 | `python scripts/run_suite.py configs/experiments/main.json` | **9/9 completed**, no crash (LSTM 13.12–14.32, Transformer 13.23–15.14, hybrid 14.47–15.72). |
| 5 | `python scripts/run_suite.py configs/experiments/ablation.json` | **3/3 completed** (15.28 / 15.52 / 15.61). |
| 6 | `python scripts/run_suite.py configs/experiments/hyperparameters.json` | **4/4 completed** (14.12 / 14.69 / 13.87 / 14.14). |
| 7 | `python scripts/run_suite.py configs/experiments/data_strategy.json` | **4/4 completed** (29.96 / 11.34 / 16.14 / 11.60). |
| 8 | `python -m src.evaluation.test_evaluation --auto` | **4 models evaluated** on the official test set (Transformer 15.54, LSTM 16.93, hybrid 17.32, RF 18.35). |
| 9 | `python -m src.evaluation.error_analysis` | Figures and tables written to `experiments/figures/error_analysis/` and `experiments/results/error_analysis/`. |
| 10 | `python -m src.evaluation.cost_analysis` | Cost table + figure written; latencies measured on CUDA. |
| 11 | `python scripts/make_summary_tables.py` | 6 markdown tables regenerated from disk artifacts. |
| 12 | `streamlit run demo/app.py` | HTTP 200, health `ok`; UI exercised (see §6). |

No run was reported as completed without a checkpoint, history,
predictions and metrics on disk; `experiments/registry.csv` contains
22 rows, all `status = completed`.

## 6. Demo verification

Verified by running the app (`streamlit run demo/app.py --server.port
8501 --server.headless true`) and driving it in a browser:

| Check | Result |
| --- | --- |
| Server starts and answers | HTTP 200, `/_stcore/health` → `ok` |
| Home page renders | Title, RUL definition, dataset facts (80/20/100 engines), model table |
| Model list only shows real checkpoints | `random_forest`, `lstm`, `transformer`, `cnn_bilstm_mean` listed while `cnn_bilstm_attention` was still training; hybrid appeared after its checkpoint existed |
| Sample-engine input | Test engine 1 loaded (31 available cycles) |
| Prediction | Random Forest → **122.9 cycles**, cycles used 30, reference label 112 shown |
| Invalid CSV — missing columns | Rejected with "Missing required columns: sensor_7…" |
| Invalid CSV — empty file | Rejected with "The input file contains no rows." |
| Invalid CSV — non-numeric | Rejected with "Some sensor values are not numeric or are empty." |
| Invalid CSV — duplicate cycles | Rejected with "Duplicate (engine_id, cycle) rows were found." |
| Invalid CSV — negative engine id | Rejected with "engine_id and cycle must be positive integers." |
| Multi-engine upload | 3 predictions returned for engines 1/2/3 (118.3 / 122.2 / 42.7 vs truth 112 / 98 / 69) |

Not verified: mobile/responsive layout at narrow widths, and the
Comparison/Explainability pages were only smoke-checked rather than
exercised end-to-end. No screenshots were captured.

## 7. Data provenance

Source: NASA Prognostics Center of Excellence, *Turbofan Engine
Degradation Simulation Data Set*, obtained from the NASA PHM archive
(`https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`,
12,429,152 bytes). The archive contains a nested `CMAPSSData.zip`; the
three FD001 files were extracted from it into `data/raw/`.

| File | Rows | Engines | SHA-256 |
| --- | --- | --- | --- |
| `train_FD001.txt` | 20,631 | 100 | `963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8` |
| `test_FD001.txt` | 13,096 | 100 | `3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851` |
| `RUL_FD001.txt` | 100 | 100 | `a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca` |

The files are git-ignored and not redistributed. `RUL_FD001.txt` has
min 7, median 86, max 145 cycles (mean 75.52, std 41.76).

## 8. Remaining limitations

* FD001 only.
* Validation uses 60 fixed endpoints, not full trajectories.
* RUL calibration is only meaningful up to the 125-cycle cap.
* No uncertainty quantification.
* No automated UI test for the Streamlit app.
* GPU determinism is not bit-exact across driver versions.
