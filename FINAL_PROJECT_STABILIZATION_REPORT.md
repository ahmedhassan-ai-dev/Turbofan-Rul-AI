# FINAL PROJECT STABILIZATION REPORT

**Project:** Turbofan Predictive Maintenance — RUL Estimation (NASA C-MAPSS FD001)
**Date:** 2026-11-18
**Scope:** implementation + verification (not a documentation-only audit)
**Result:** 5/5 models load and predict, Streamlit GUI verified end-to-end, 111/111
tests pass, submission archive built.

---

## 0. Protection of the existing project

| Item | Action |
| --- | --- |
| Complete repository backup | `backups/Turbofan_RUL_repo_backup_pre_stabilization_2026-11-18.zip` (83 MB, full tree including `.git`) |
| Existing submission ZIP | `turbofan_rul_submission.zip` — **untouched** |
| Existing audit/report ZIPs | `repo/FINAL_AUDIT_REPORTS.zip`, `repo/final_model_results.zip` — untouched, excluded from the new archive to avoid duplicate archives |
| Original Random Forest checkpoint | `checkpoints/random_forest.joblib` — untouched (11,850,552 bytes) |
| Optimized Random Forest checkpoint | `checkpoints/random_forest_optimized.joblib` — untouched (121,432,897 bytes) |
| Deep-model checkpoints | `experiments/runs/*/best_model.pt` — untouched |
| Retraining | **none performed** |
| Model weights / metrics | **not modified** (verified by exact metric reproduction, §5) |

Git state recorded **before** this work (branch `rul-project-integration`):

```
 M demo/app.py
 M experiments/registry.csv
 M experiments/results/official_test/random_forest_metrics.json
 M experiments/results/official_test/random_forest_predictions.csv
 M experiments/results/official_test/test_metrics.csv
 M experiments/results/official_test/test_summary.json
 M src/inference/predictor.py
?? FINAL_MODEL_COMPARISON_FAIRNESS_AUDIT.pdf, INTEGRATION_PLAN.md,
   RANDOM_FOREST_ORIGINAL_WORK_AUDIT.pdf, audit_report.md,
   checkpoints/random_forest_optimized.joblib, experiments/random_forest_optimization/,
   generate_pdfs.py, report/*, src/inference/predictor.py.bak,
   streamlit_gui_state.png, test_gui_verify.py
```

Git state **after** this work is listed in §2; every remaining modification is
accounted for in this report.

---

## 1. Confirmed issues (root causes, not hypotheses)

### Issue 1 — CRITICAL: the shared inference path had lost all three PyTorch branches

`src/inference/predictor.py::RULPredictor.predict_windows` had been rewritten to
always run `extract_window_features(...)` and `self.model.predict(...)`.

* The Random Forest (both checkpoints) therefore worked, but **LSTM,
  Transformer and CNN-BiLSTM-Attention could never predict**: a `torch.nn.Module`
  has no `.predict()`, so the GUI's *Estimate RUL* button raised
  `AttributeError: 'RULLSTM' object has no attribute 'predict'` (wrapped by the
  GUI as `Inference failed: ...`).
* `metadata` for joblib checkpoints was reset to `{}`, so `display_name` and the
  training metadata (`rul_cap`, `seed`, `features`, `statistics`) were lost.
* Predictions were cast to `float32` instead of `float64`, which perturbed the
  saved official-test predictions in the 12th decimal and produced a spurious
  diff in four committed result files.
* The `else` branch still held debug scaffolding: a second `joblib.load`, five
  `print("[LOG] ...")` statements and a local `import os`. (The original
  `UnboundLocalError: joblib` root cause had been removed earlier, but the
  scaffolding that printed it stayed behind.)

**Why it was necessary to fix:** without it four of five models are unusable in
the app, and the GUI error the user reported before ("Unsupported baseline
checkpoint") was only one symptom of the same broken constructor block.

### Issue 2 — `experiments/registry.csv` was not portable and misrepresented the RF study

The hyper-parameter sweep appended **40 rows**, every one of them containing

* an **absolute Windows path** (`C:\Users\...\checkpoints\random_forest_optimized.joblib`,
  same file for all 40 rows),
* an absolute `config_path`,
* Windows `\` separators in `run_dir` (the file mixes `\\` and `/`).

The registry is the project's declared single source of truth, so absolute paths
break reproducibility on any other machine, and 40 rows pointing at one file
misrepresents which checkpoint is shipped.

### Issue 3 — Committed official-test results were perturbed by Issue 1

`random_forest_metrics.json`, `random_forest_predictions.csv`, `test_metrics.csv`
and `test_summary.json` differed from the committed values only in the ~1e-12
range — a pure float32 artifact. No value changed materially, but a submitted
report must reproduce bit-for-bit.

### Issue 4 — `demo/app.py` regressions

* `@st.cache_resource` was followed by a blank line before `def` (works, but
  fragile/incorrect style) and the blank line separating two functions was lost.
* Unused `import joblib` / `import torch` were added while fixing the RF issue.
* The optimized-RF registration carried a misleading comment ("wrap it for the
  RULPredictor" — nothing is wrapped).

### Issue 5 — Attention ablation violated its own initialization contract (2 failing tests)

`tests/test_cnn_bilstm_attention.py` (2 of 111) failed:

```
test_hybrid_without_attention_uses_uniform_valid_weights
test_shared_initialization_matches
AssertionError: Mismatched elements: 8192 / 8192 (100.0%)
```

Cause: `CNNBiLSTMAttention.__init__` built `MaskedAdditiveAttention` **only when**
`use_attention=True`, i.e. between the BiLSTM and the regression head. With
`torch.manual_seed(42)` the attention block consumed random numbers, so every
parameter created *after* it (the regression head) received different values in
the ablated model. The ablation is documented as “identical CNN + BiLSTM +
head, mean pooling instead of attention”; the code did not honour that.

Fix (in the model, not the tests): the attention block is now always
constructed and simply discarded when `use_attention=False`. The full model's
construction order and random draws are **unchanged** (so nothing about the
trained checkpoints changes) while the ablated model now gets an identical
shared initialization and a strictly smaller parameter count.

### Issue 6 — Operational: seven stale Streamlit servers on port 8501

Several `streamlit run` processes from earlier sessions were still bound to
8501, each serving a *stale* version of the app. They were stopped and one
fresh server was started.

---

## 2. Source files changed

| File | Change | Why |
| --- | --- | --- |
| `src/inference/predictor.py` | Rewritten `__init__`/loading into `_load_sklearn` + `_load_torch`; restored batched PyTorch inference (`_predict_torch`) and float64 RF inference (`_predict_sklearn`); removed debug prints, duplicate `joblib.load` and local `import os`; restored `metadata`, `display_name` and the `np.maximum(..., 0)` floor; added clear errors for corrupted/unsupported checkpoints and an explicit `n_features_in_` compatibility check | Issue 1 |
| `demo/app.py` | Fixed decorator/blank-line regressions, removed unused `import joblib` / `import torch`, clarified the optimized-RF registration and description | Issue 4 |
| `experiments/registry.csv` | 40 absolute-path sweep rows → 1 canonical, repo-relative row (`rf_150e_sqrt_msl1_md20_seed42`, tag `rf_optimization`, checkpoint `checkpoints/random_forest_optimized.joblib`); 62 → 23 rows | Issue 2 |
| `experiments/results/official_test/random_forest_metrics.json`, `random_forest_predictions.csv`, `test_metrics.csv`, `test_summary.json` | Restored to the committed values (`git checkout`), then re-derived and matched exactly | Issue 3 |
| `src/models/cnn_bilstm_attention.py` | Always construct the attention block; keep it only when `use_attention=True` | Issue 5 |
| `scripts/consolidate_rf_registry.py` | **New** one-off script that preserves the sweep and writes the single canonical registry row | Issue 2 |
| `scripts/verify_models.py` | **New** reproducible per-model verification + metric reproduction | §4–5 |
| `scripts/verify_gui.py` | **New** headless Streamlit end-to-end test (`streamlit.testing.v1.AppTest`) | §6 |
| `experiments/random_forest_optimization/sweep_results.csv` | **New** artifact: all 40 sweep rows with relative paths (no data lost) | Issue 2 |
| `README.md` | Added the optimized-RF row, the checkpoint list, and the two verification commands | delivery |
| `SUBMISSION_MANIFEST.md` | Test count 104 → 111, optimized-RF checkpoint documented, verification step added | delivery |

Removed (my own temporary artifacts, superseded): `src/inference/predictor.py.bak`,
`test_gui_verify.py` (root-level ad-hoc script replaced by `scripts/verify_gui.py`),
`streamlit_gui_state.png`.

Not touched: model weights, `src/train*.py`, `src/training/`, feature selection,
scaler, splits, any reported metric value.

---

## 3. Model loading architecture after the fix

| Model | Checkpoint | Format | Window | Model input |
| --- | --- | --- | --- | --- |
| random_forest | `checkpoints/random_forest.joblib` | joblib dict `{model, features, window_size, rul_cap, ...}` | 30 | 68 statistical features (17 × last/mean/std/change) |
| optimized_random_forest | `checkpoints/random_forest_optimized.joblib` | joblib **bare estimator** | 30 (protocol default) | 68 statistical features |
| lstm | `experiments/runs/lstm_cap125_seed123/best_model.pt` | torch dict (`model_name`, `model_kwargs`, `model_state_dict`, `metadata`) | from metadata (30) | raw scaled window `[N,30,17]` + mask |
| transformer | `experiments/runs/transformer_cap125_seed3407/best_model.pt` | torch dict | 30 | raw scaled window + mask |
| cnn_bilstm_attention | `experiments/runs/cnn_bilstm_attention_cap125_seed123/best_model.pt` | torch dict | 30 | raw scaled window + mask |

The two families are deliberately **not** assumed to share an internal
representation: Random Forest consumes the 68 hand-built statistics, the deep
models consume the scaled 17-channel window with the padding mask. Exactly one
predictor implementation exists (`RULPredictor`, used by the GUI, the official
test evaluation and the error analysis).

Missing / corrupted checkpoints now raise `FileNotFoundError` (missing) or
`ValueError("Could not read/load ... : <cause>")` (corrupted), both shown in the
GUI as a readable message. No path can silently return a default prediction.

---

## 4. Per-model inference results (real data, real pipeline)

`python scripts/verify_models.py` — input: official FD001 test engine 1, last 60
cycles, scaled with the training scaler, 100 % through `prepare_scaled_frame` →
`create_last_windows`:

```
random_forest              kind=sklearn  window=30 RUL= 122.909 cycles  OK
optimized_random_forest    kind=sklearn  window=30 RUL= 118.512 cycles  OK
cnn_bilstm_attention       kind=torch    window=30 RUL= 113.426 cycles  OK
lstm                       kind=torch    window=30 RUL= 120.680 cycles  OK
transformer                kind=torch    window=30 RUL= 122.551 cycles  OK
```

All predictions are finite, non-negative and expressed in cycles (reference RUL
for engine 1 is 112 cycles). Exit code 0.

Warnings: none at inference. (Streamlit itself prints two deprecation notices
about `use_container_width`, see §8.)

---

## 5. Metric / comparison validation

Recomputed from the shipped checkpoints, not copied from the result files:

```
random_forest            val RMSE=18.549619037 expected=18.549619037 OK
optimized_random_forest  val RMSE=16.517202247 expected=16.517202247 OK
random_forest test RMSE=18.348562770 expected=18.348562770 OK   (exact, 1e-9)
```

* Validation is scored on the **60 predefined endpoints**
  (`configs/data/fd001_validation_endpoints.csv`), the same protocol used by
  training — the earlier attempt to score all 4,070 validation windows was a
  check error, not a metric error.
* The official-test number reproduces the committed value exactly, which also
  proves the float32 regression of Issue 1 is gone.
* The optimized RF validation RMSE reproducing 16.5172 proves that
  `checkpoints/random_forest_optimized.joblib` really is the
  `rf_150e_sqrt_msl1_md20_seed42` winner (n_estimators=150, max_features='sqrt',
  min_samples_leaf=1, max_depth=20) and not one of the other 39 trials.

**Original vs optimized Random Forest are kept separate everywhere:**

| | Validation RMSE (60 endpoints) | Official test RMSE | Checkpoint |
| --- | --- | --- | --- |
| Original RF | 18.5496 | 18.3486 | `checkpoints/random_forest.joblib` |
| Optimized RF | **16.5172** | *not evaluated on test* | `checkpoints/random_forest_optimized.joblib` |

The optimized RF was selected on validation only; it has **no** official-test
entry, and `experiments/results/official_test/test_metrics.csv` still contains
exactly the four original models (transformer 15.5414, lstm 16.9256,
cnn_bilstm_attention 17.3178, random_forest 18.3486). Validation numbers are
never mixed into the test files, and no metric value was rewritten. The
optimized row is tagged `rf_optimization` (not `main`) so it can never enter the
validation-based model selection of `test_evaluation --auto`.

---

## 6. Streamlit GUI verification

Server started from the repository root:

```
python -m streamlit run demo/app.py --server.port 8501
```

`http://127.0.0.1:8501` → HTTP 200, `/_stcore/health` → `ok`.

Browser automation in this environment could not drive the widgets reliably
(the Chromium tab throws `Failed to fetch dynamically imported module`), so the
GUI was verified with `streamlit.testing.v1.AppTest`, which executes
`demo/app.py` itself, builds the real widget tree and exercises the same
callbacks a user triggers:

`python scripts/verify_gui.py` →

```
GUI STARTUP
error elements on load: none
navigation options: ['Home','Input','Monitoring','Prediction','Comparison','Explainability']

CHECKPOINT DISCOVERY (sidebar)
  random_forest              listed=True
  optimized_random_forest    listed=True
  lstm                       listed=True
  transformer                listed=True
  cnn_bilstm_attention       listed=True

PREDICTION PAGE      (input: sample test engine 1, 60 cycles)
random_forest              RUL =  122.909 cycles   OK
optimized_random_forest    RUL =  118.512 cycles   OK
lstm                       RUL =  120.680 cycles   OK
transformer                RUL =  122.551 cycles   OK
cnn_bilstm_attention       RUL =  113.426 cycles   OK

MODEL COMPARISON PAGE
                  model  predicted_rul
          random_forest     122.908706
optimized_random_forest     118.511834
   cnn_bilstm_attention     113.426056
                   lstm     120.680084
            transformer     122.550575
        cnn_bilstm_mean      96.245895

GUI VERIFICATION PASSED: 5/5 models predicted, comparison OK
```

Covered: startup, navigation, sidebar checkpoint discovery, model selection and
checkpoint switching, Predict button per model, result rendering (metric +
table + figure), Comparison page, no error elements.

**Not covered by automation:** the *Upload a sensor CSV* input, the
Monitoring/Explainability plots, and clicking through the real browser tab.
Those use unchanged code paths (`validate_raw_frame`, matplotlib) and are
covered by unit tests, but they were not clicked through. They can be checked
manually in the running app: sidebar → *Input* → upload a 26-column C-MAPSS CSV
(or keep the sample engine) → *Prediction* → pick a model → **Estimate RUL**.

---

## 7. Test suite

```
python -m pytest -q
111 passed in 15.66s
```

Before the fix: `2 failed, 109 passed` (Issue 5). No test was modified, skipped
or weakened. The ablated model still loads its real checkpoint
(`cnn_bilstm_mean_cap125_seed123`: 189,889 parameters) and the full model still
has 198,209 parameters — both identical to the numbers in the registry, which
confirms no architecture/weight change.

---

## 8. Remaining limitations

1. **Budgeted, not re-trained:** every metric in the report is the original one.
   Nothing was retrained; `python -m src.train ...` re-run from scratch would
   produce its own numbers.
2. **Optimized RF has no official-test score.** Adding one would require a new
   evaluation pass; that is a reporting decision, not a bug, and the test files
   were deliberately left untouched.
3. **GUI automation gap** (see §6): CSV upload and the two plotting pages were
   not driven by an automated click-through.
4. **Streamlit deprecation warnings**: `use_container_width` is deprecated in
   Streamlit 1.57 and will be removed after 2025-12-31. Harmless today; the
   replacement is `width="stretch"`. Left unchanged to avoid a cosmetic sweep.
5. **NASA raw data**: `data/raw/` is shipped in this archive (as stated by
   `SUBMISSION_MANIFEST.md`); `data/raw/cmapss_raw.zip` was excluded as a
   duplicate of the three text files.
6. **Model disk size**: `random_forest_optimized.joblib` is 121 MB (1.6 M tree
   nodes). It is required for the optimized model to be loadable without
   retraining.

---

## 9. Final archive

**Path:** `turbofan_rul_submission_FINAL_STABILIZED_2026-11-18.zip`
(workspace root, next to — not over — the untouched `turbofan_rul_submission.zip`)

Contents: all source, configs, tests, demo, scripts, `report/`, the full
`experiments/` tree (runs, results, figures, registry, RF sweep), both
Random-Forest checkpoints, all deep-model checkpoints, processed data and the
three NASA raw text files, `requirements.txt`, `README.md`, `.env.example`,
`.gitignore`, `SUBMISSION_MANIFEST.md` and this report.
Excluded: `.git/`, `__pycache__/`, `.pytest_cache/`, `*.log`, `*.bak`,
nested archives (`*.zip`), screenshots.

---

## 10. How to run

```bash
cd repo
pip install -r requirements.txt

# optional, only if data/processed is missing
python -m src.datasets.preprocessing

# the app
python -m streamlit run demo/app.py --server.port 8501
# -> http://127.0.0.1:8501
```

Reproduce every claim in this report (from `repo/`):

```bash
python scripts/verify_models.py     # 5/5 checkpoints predict + metric reproduction
python scripts/verify_gui.py        # headless Streamlit end-to-end test
python -m pytest -q                 # 111 tests
```
