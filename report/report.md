# Remaining Useful Life Prediction for Turbofan Engines with Deep Learning

**Technical report — NASA C-MAPSS FD001**

> Results are final for the runs in `experiments/registry.csv`. Every
table was filled from files on disk (`experiments/results/`,
`experiments/runs/`), never typed by hand. Regenerate the aggregate
tables with `python scripts/make_summary_tables.py`.

---

## 1. Abstract

We study Remaining Useful Life (RUL) prediction on the NASA C-MAPSS
FD001 turbofan degradation benchmark. RUL is formulated as supervised
regression on causal sensor windows of 30 cycles. Four models are
compared under an identical protocol: a Random Forest baseline on
statistical window features, a stacked LSTM, a padding-aware Transformer
encoder, and a CNN-BiLSTM with additive temporal attention. Training and
validation are separated by engine unit, the feature standardizer is
fitted on training engines only, and the official truncated test set is
evaluated once after model selection on validation. In addition to
RMSE, MAE, bias and R², we report the asymmetric NASA scoring function,
a per-horizon error analysis, a controlled attention ablation, a
hyper-parameter study, a data-strategy study and a computational cost
benchmark. An interactive Streamlit demo exposes the trained models.

## 2. Introduction

Predictive maintenance replaces fixed maintenance schedules with
estimates of how long a component will keep working. For turbofan
engines the quantity of interest is the remaining number of operational
cycles before failure. Because sensor traces are long, noisy and
multi-channel, sequence models are a natural fit, but a fair comparison
requires strict control of the data protocol: engine-wise splits, causal
windows, and a test set that is touched only once.

This report documents the full pipeline: data preparation, model
architectures, training configuration, experiments, official test
evaluation, error analysis and cost analysis.

## 3. Problem statement

Let an engine have sensor observations
`x_t ∈ R^F` at cycle `t`, and let `t_fail` be its failure cycle. The
target is

```
y_t = t_fail - t
```

and the model is a function `f_θ(x_{t-T+1}, ..., x_t) → ŷ_t` with
`T = 30`. Only the past and the present may enter `f_θ`.

The benchmark's scoring function is asymmetric: under-estimating
remaining life (early replacement) is penalised less than
over-estimating it (a failure in service).

## 4. Motivation

* A model that is accurate "on average" but systematically late is
  dangerous, which is why the asymmetric score matters.
* Reported RUL numbers are only comparable when the split, the scaler
  and the window protocol are identical; much of the published variance
  comes from protocol differences rather than architectures.
* A reproducible, leakage-free baseline is therefore as valuable as a
  new architecture.

## 5. Related work

* **Classical:** Random Forests and gradient boosting on window
  statistics remain strong baselines on C-MAPSS.
* **Recurrent:** LSTM/GRU RUL estimators (Heimes, 2008) established the
  sequence formulation used here.
* **Attention:** Transformer encoders (Vaswani et al., 2017) and
  additive attention (Bahdanau et al., 2015) provide global temporal
  weighting; on short windows (30 steps) their advantage over recurrent
  models is not guaranteed.
* **Hybrids:** CNN front-ends followed by recurrent layers exploit local
  degradation patterns before temporal aggregation.

We do not claim state-of-the-art performance; we claim a controlled,
reproducible comparison on FD001.

## 6. Dataset description

FD001 of the NASA C-MAPSS Turbofan Engine Degradation Simulation Data
Set: one operating condition, one fault mode (HPC degradation).

| Split | Engines | Rows / samples | Notes |
| --- | --- | --- | --- |
| Train | 80 | 16,561 windows | complete run-to-failure trajectories |
| Validation | 20 | 60 endpoints | cut-offs at 100 / 60 / 30 cycles before failure |
| Official test | 100 | 100 windows | truncated trajectories + `RUL_FD001.txt` |

Each row has 2 identifiers, 3 operational settings and 21 sensors.

## 7. Data preprocessing

1. Schema validation and duplicate/consistency checks.
2. Constant-channel removal using the training split only:
   `setting_3`, `sensor_1`, `sensor_5`, `sensor_10`, `sensor_16`,
   `sensor_18`, `sensor_19` (7 of 24), leaving **17 features**.
3. `StandardScaler` fitted on training engines, applied unchanged to
   validation and test.
4. RUL labels: `max_cycle - cycle` for complete trajectories; official
   labels for the truncated test set.
5. Training target capping at 125 cycles.

Leakage control: engine-wise split, train-only scaling, train-only
feature selection, causal windows, and unit tests that mutate future
readings to prove earlier windows are unaffected.

## 8. Methodology

### 8.1 Windowing

One window per `(engine, cycle)`, length 30, left-padded by repeating
the earliest observed reading, with a boolean mask. The official test
uses one window per engine ending at its last recorded cycle.

### 8.2 Training

Adam (LR 1e-3), MSE loss on capped targets, batch 64, up to 60 epochs,
early stopping with patience 10 on validation RMSE, gradient clipping at
norm 1.0, three seeds (42, 123, 3407).

### 8.3 Model selection

The run with the lowest validation RMSE per model family is selected
**before** the official test set is evaluated. Test results never
influence architecture or hyper-parameter choices.

## 9. Baseline

Random Forest (150 trees, `min_samples_leaf=3`) on four statistics per
feature (last value, mean, standard deviation, change) computed over the
valid (unpadded) part of each window: 68 input features.

* Rationale: a strong, cheap, non-sequential reference.
* Same split, same cap (125), same metric implementation.

## 10. Deep learning architectures

**Stacked LSTM.** Two LSTM layers, 64 hidden units, dropout 0.2.
Left-padded windows are converted to right-padded sequences and packed,
so padded steps never reach the recurrence.

**Transformer encoder.** Linear projection to `d_model = 64`, learned
positional embeddings numbered from the first *real* reading, two
encoder layers (4 heads, FFN 128, GELU, pre-norm), padding mask on the
attention, masked mean pooling, regression head.

**CNN-BiLSTM-Attention.** Two Conv1d layers (17→64→64, kernel 3), a
bidirectional LSTM (64 per direction, 2 layers), masked additive
attention over time, context vector into a `128→64→1` head.

**Ablation variant.** Same CNN, BiLSTM and head, with masked mean
pooling instead of attention (no attention parameters at all).

## 11. Experimental setup

* Hardware/software: recorded per run in
  `experiments/runs/<id>/config.json` (`device_name`, `torch_version`).
* Metrics: RMSE, MAE, bias (`pred − true`), R², NASA score (sum) and
  mean NASA score.
* Artifacts per run: `config.json`, `training_history.csv`,
  `best_predictions.csv`, `metrics.json`, `summary.json`,
  `best_model.pt`.

## 12. Experimental hypotheses

| ID | Hypothesis | Test |
| --- | --- | --- |
| H1 | Deep sequence models beat the Random Forest baseline on validation RMSE | main suite vs baseline |
| H2 | Attention pooling improves the CNN-BiLSTM over mean pooling | ablation suite (same seeds) |
| H3 | The RUL cap changes accuracy on short horizons but not long ones | data-strategy suite |
| H4 | Window length 30 is a reasonable compromise between context and sample count | data-strategy suite |
| H5 | Seed variance is small relative to between-model differences | main suite, 3 seeds |

## 13. Hyper-parameter study

LSTM with one factor changed at a time relative to the baseline
(LR 0.001, hidden 64): LR ∈ {0.0005, 0.002}, hidden ∈ {32, 128},
seed 42 fixed.

| Configuration | Val RMSE | Val MAE |
| --- | --- | --- |
| LR 5e-4 | 14.12 | 11.24 |
| LR 1e-3 (baseline) | **13.68** | **10.57** |
| LR 2e-3 | 14.69 | 11.63 |
| hidden 32 | 13.87 | 10.30 |
| hidden 64 (baseline) | **13.68** | **10.57** |
| hidden 128 | 14.14 | 10.85 |

Neither factor produced an improvement larger than the seed-to-seed
spread measured in §16.

## 14. Data strategy study

LSTM, seed 42, one factor per run: RUL cap ∈ {50, 100} (baseline 125)
and window length ∈ {20, 50} (baseline 30). Dataset, split and scaler
are unchanged.

| Strategy | Val RMSE | Val bias |
| --- | --- | --- |
| RUL cap 50 | 29.96 | −19.36 |
| RUL cap 100 | **11.34** | +0.19 |
| RUL cap 125 (baseline) | 13.68 | +3.41 |
| Window 20 | 16.14 | +4.08 |
| Window 30 (baseline) | 13.68 | +3.41 |
| Window 50 | **11.60** | +3.03 |

Cap 50 illustrates a protocol trap rather than a modelling result: the
validation endpoints sit at 100/60/30 cycles and a model capped at 50
cannot represent them, producing a −19-cycle bias. Cap 100 and a 50-step
window are promising but single-seed.

## 15. Model comparison

Validation (3 seeds, mean ± std):

| Model | Val RMSE | Std | Val MAE |
| --- | --- | --- | --- |
| LSTM | **13.71** | 0.60 | 10.65 |
| Transformer | 14.21 | 0.96 | 11.03 |
| CNN-BiLSTM-Attention | 14.98 | 0.65 | 11.87 |
| Random Forest | 18.55 | — | 13.35 |

Official test (100 engines, evaluated once):

| Model | RMSE | MAE | Bias | R² | NASA sum | NASA mean |
| --- | --- | --- | --- | --- | --- | --- |
| Transformer | **15.54** | **11.96** | +1.67 | **0.860** | **363.2** | **3.63** |
| LSTM | 16.93 | 12.13 | −3.37 | 0.834 | 454.0 | 4.54 |
| CNN-BiLSTM-Attention | 17.32 | 13.21 | −0.38 | 0.826 | 507.2 | 5.07 |
| Random Forest | 18.35 | 13.79 | +3.07 | 0.805 | 733.9 | 7.34 |

The ranking flips between validation (LSTM first) and test (Transformer
first) by a margin (≈1.4 cycles) that is comparable to the seed standard
deviation, so the two are best described as statistically close.

## 16. Multi-seed analysis

Each deep model is trained with seeds 42, 123 and 3407 under an
identical configuration. Mean and standard deviation of validation RMSE
are reported; seed variance is the floor below which differences
between architectures should not be interpreted.

| Model | Seed 42 | Seed 123 | Seed 3407 | Mean | Std |
| --- | --- | --- | --- | --- | --- |
| LSTM | 13.68 | 13.12 | 14.32 | 13.71 | 0.60 |
| Transformer | 15.14 | 14.26 | 13.23 | 14.21 | 0.96 |
| CNN-BiLSTM-Attn | 14.77 | 14.47 | 15.72 | 14.98 | 0.65 |

Seed spread is 0.6–1.0 cycles RMSE, i.e. of the same order as the gap
between LSTM and Transformer. H5 (seed variance smaller than model
differences) therefore only holds for the baseline gap: every deep model
is clearly better than the Random Forest (18.55), but the ordering among
the three deep models is not resolved by three seeds.

## 17. Ablation study

| Variant | Seeds | Val RMSE mean | Std | Val MAE mean |
| --- | --- | --- | --- | --- |
| CNN-BiLSTM + attention | 3 | **14.98** | 0.65 | **11.87** |
| CNN-BiLSTM + mean pooling | 3 | 15.47 | 0.17 | 12.08 |

Per-seed differences (attention − mean pooling): −0.51, −1.06, +0.11.
The mean advantage is ≈0.49 cycles RMSE, smaller than the attention
variant's own standard deviation, and it reverses in one of three seeds.
**H2 is therefore only partially supported:** attention helps on average
but the evidence is not conclusive at this sample size, and the
mean-pooling variant is markedly more stable across seeds (0.17 vs 0.65).

The comparison is only meaningful because the two variants share the
same CNN, BiLSTM, head, split, cap, window length and seeds; only the
pooling operator differs.

## 18. Official test results

The official test set is evaluated exactly once per selected model,
after selection on validation. Ground truth comes from
`RUL_FD001.txt` and each prediction is aligned by engine id.

Per-horizon RMSE:

| Horizon | CNN-BiLSTM-Attn | LSTM | Random Forest | Transformer |
| --- | --- | --- | --- | --- |
| Short (≤ 50) | 12.14 | **6.53** | 11.85 | 7.01 |
| Medium (51–100) | 20.13 | 20.31 | 21.86 | **19.48** |
| Long (> 100) | 18.55 | 20.01 | 19.68 | **17.08** |

Validation-selected checkpoints, one evaluation pass, no re-selection.
Predictions: `experiments/results/official_test/<model>_predictions.csv`.

## 19. Error analysis

Horizons: short ≤ 50, medium 51–100, long > 100 cycles.

| Horizon | Model | n | RMSE | MAE | Bias |
| --- | --- | --- | --- | --- | --- |
| Short (≤ 50) | LSTM | 33 | **6.53** | 4.75 | −1.01 |
| Short (≤ 50) | Transformer | 33 | 7.01 | 5.69 | +3.41 |
| Short (≤ 50) | Random Forest | 33 | 11.85 | 7.76 | +5.99 |
| Short (≤ 50) | CNN-BiLSTM-Attn | 33 | 12.14 | 9.47 | +5.52 |
| Medium (51–100) | Transformer | 34 | **19.48** | 16.57 | +6.36 |
| Medium (51–100) | CNN-BiLSTM-Attn | 34 | 20.13 | 15.60 | +3.99 |
| Medium (51–100) | LSTM | 34 | 20.31 | 15.47 | +1.06 |
| Medium (51–100) | Random Forest | 34 | 21.86 | 17.79 | +13.14 |
| Long (> 100) | Transformer | 33 | **17.08** | 13.49 | −4.90 |
| Long (> 100) | CNN-BiLSTM-Attn | 33 | 18.55 | 14.49 | −10.77 |
| Long (> 100) | Random Forest | 33 | 19.68 | 15.70 | −10.23 |
| Long (> 100) | LSTM | 33 | 20.01 | 16.06 | −10.30 |

Short-horizon accuracy is where the models separate: the LSTM reaches
RMSE 6.53 near failure, while the hybrid and baseline stay above 11.
The long horizon shows the opposite pattern — every model except the
Transformer under-predicts remaining life by about 10 cycles, which is
the *safer* direction but inflates the long-horizon RMSE.

Worst-case behaviour is concentrated in the medium horizon. Engine 93
(true RUL 85) is under-predicted by 48.9 cycles by the hybrid and 41.3
by the LSTM; engine 79 (true 63) is over-predicted by 42.4 by the hybrid
and 50.9 by the Random Forest. The same engines recur in the worst-10
lists of every model, which points at intrinsically hard trajectories
rather than model-specific failures.

Full tables and figures: `experiments/results/error_analysis/`,
`experiments/figures/error_analysis/`.

## 20. Computational cost

| Model | Params | Size (MB) | Train (s) | Device | Latency b=1 (ms) | Latency b=100 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| LSTM | 56,641 | 0.22 | 38.1 | cuda | 2.50 | 2.70 |
| Transformer | 103,105 | 0.39 | 121.6 | cuda | 1.75 | 1.59 |
| CNN-BiLSTM-Attn | 198,209 | 0.76 | 74.3 | cuda | 4.92 | 6.20 |
| Random Forest | 606,408 nodes | 12.0 | 8.9 | cpu | 35.72 | 42.20 |

The Transformer is both the most accurate on test and the fastest per
batch among the deep models; the hybrid is 3.5× larger than the LSTM and
2–4× slower while scoring worse. The Random Forest is cheapest to train
but ≈15–26× slower at inference than the deep models, because it
evaluates 150 trees per window.

All latencies are measured on one device with identical input shape, so
the comparison is like-for-like. Training times come from the registry
and record the device of each run.

## 21. Live demo

A Streamlit application (`demo/app.py`) provides: home/overview, input
selection (sample engine or CSV upload with validation), engine
monitoring, model selection restricted to existing checkpoints,
prediction with an input summary, model comparison and a limited
explainability view.

The explainability view shows sensor trends and relative changes; it
explicitly does **not** treat attention weights as causal evidence.

## 22. Limitations

* FD001 only (single condition, single fault mode).
* Validation is three fixed cut-offs per engine, not a full-trajectory
  evaluation.
* Capped targets limit calibration beyond 125 cycles.
* Point predictions only — no uncertainty estimate.
* Attention maps are diagnostic, not causal.

## 23. Conclusion

On FD001, a leakage-free pipeline with engine-wise splits, train-only
standardization and strictly causal windows produces consistent results
across four model families. All three deep models clearly outperform the
Random Forest baseline (official-test RMSE 15.54–17.32 vs 18.35). The
best official-test configuration is the Transformer encoder
(RMSE 15.54, MAE 11.96, R² 0.860, NASA score 363.2), although it is
statistically close to the LSTM (16.93), because the seed-to-seed spread
(0.6–1.0 cycles) is of the same order as the difference between them.

The attention ablation gives a small and directionally positive but
inconclusive result: attention improves validation RMSE by ≈0.49 cycles
on average over three seeds, and the improvement reverses in one seed.
We therefore do not claim that attention is necessary for this task.

Errors concentrate in the medium RUL horizon and in the same handful of
test engines for every model, which points at intrinsically ambiguous
trajectories rather than architecture-specific failures. Two protocol
findings are worth carrying forward: the RUL cap must match the
evaluation horizons (a 50-cycle cap inflates RMSE to 30 against
endpoints at 100/60/30), and a longer window (50 cycles) improved
validation RMSE in a single-seed probe (11.60 vs 13.68) and deserves a
full multi-seed study.

## 24. Future work

* Multi-condition sub-sets (FD002/FD004) with condition-aware scaling.
* Probabilistic RUL and cost-sensitive training aligned with the NASA
  scoring asymmetry.
* Longer context, per-engine adaptation, and deployment-oriented
  latency budgets.

## 25. References

1. A. Saxena, K. Goebel, D. Simon, N. Eklund, *Damage Propagation
   Modeling for Aircraft Engine Run-to-Failure Simulation*, PHM 2008.
2. NASA PCoE, *Turbofan Engine Degradation Simulation Data Set*.
3. F. O. Heimes, *Recurrent Neural Networks for Remaining Useful Life
   Estimation*, PHM 2008.
4. A. Vaswani et al., *Attention Is All You Need*, NeurIPS 2017.
5. D. Bahdanau, K. Cho, Y. Bengio, *Neural Machine Translation by
   Jointly Learning to Align and Translate*, ICLR 2015.
6. S. Hochreiter, J. Schmidhuber, *Long Short-Term Memory*, 1997.
7. L. Breiman, *Random Forests*, Machine Learning, 2001.
