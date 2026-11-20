# Turbofan Predictive Maintenance — RUL Estimation
### 15-minute presentation deck (draft)

Slide structure: 20 slides, ~45 s each. Figures are the ones produced by
the project (`experiments/figures/`). Numbers must be taken from
`experiments/results/summary_tables/all_tables.md`; anything not in a run
directory is marked as pending.

---

## Slide 1 — Title

**Turbofan Predictive Maintenance — RUL Estimation**
Deep learning on NASA C-MAPSS FD001
Team / course / date

> Visual: engine silhouette + one sensor degradation curve.

---

## Slide 2 — Problem

* Predictive maintenance asks: *how many cycles until failure?*
* Too late → in-service failure. Too early → wasted part.
* Input: multivariate sensor history. Output: RUL in cycles.

---

## Slide 3 — Motivation

* Safety and cost both depend on the two error directions.
* C-MAPSS scoring is **asymmetric**: late predictions cost more than
  early ones.
* Comparable numbers need a strict protocol, not just a good model.

---

## Slide 4 — Dataset (FD001)

| Item | Value |
| --- | --- |
| Train engines | 100 run-to-failure trajectories |
| Test engines | 100 truncated trajectories |
| Columns | 2 ids + 3 settings + 21 sensors |
| Fault modes | 1 (HPC degradation), 1 operating condition |

> Visual: `experiments/figures/eda/engine_lifetimes.png`

---

## Slide 5 — Data pipeline

Raw → validate → drop constant channels → scale (train only) →
causal windows (30 steps) → mask → model

* 17 of 24 channels kept.
* Split **by engine**: 80 train / 20 validation.
* Test set sealed until the end.

---

## Slide 6 — RUL definition

```
RUL = max_cycle(engine) - cycle          (train/val)
RUL from RUL_FD001.txt                   (official test)
```

* Training targets capped at 125 cycles.
* Validation/test metrics on **uncapped** labels.

> Visual: RUL curve for one engine with the 125-cap line.

---

## Slide 7 — Baseline

**Random Forest** on four statistics per sensor (last, mean, std, change)
→ 68 features.

* Cheap, strong, non-sequential reference.
* Same split, cap and metrics as the deep models.

---

## Slide 8 — Models

| Model | Key idea |
| --- | --- |
| LSTM (2×64) | packed sequences, padding-aware |
| Transformer (2 layers, 4 heads) | padding mask + positional ids from first real reading |
| CNN-BiLSTM-Attention | local conv features + bidirectional context + additive attention |
| CNN-BiLSTM-Mean | ablation: mean pooling instead of attention |

> Visual: three small architecture diagrams.

---

## Slide 9 — Experimental setup

* Adam 1e-3, MSE, batch 64, ≤60 epochs, early stopping (patience 10).
* Seeds 42 / 123 / 3407 for every main model.
* Selection: lowest **validation** RMSE; test set evaluated once.
* Every run stores config, history, predictions, metrics, checkpoint.

---

## Slide 10 — Hypotheses

| ID | Hypothesis |
| --- | --- |
| H1 | Deep models beat the Random Forest |
| H2 | Attention beats mean pooling in the hybrid |
| H3 | The RUL cap changes short-horizon accuracy |
| H4 | Window length 30 is a reasonable compromise |
| H5 | Seed variance is smaller than model differences |

---

## Slide 11 — Results: validation

<!-- Insert table: experiments/results/summary_tables/validation_summary.md -->

> Note: state whether H1 holds, based on the numbers only.

---

## Slide 12 — Model comparison (official test)

<!-- Insert table: experiments/results/summary_tables/official_test.md -->

> Visual: `experiments/figures/official_test/*_actual_vs_predicted.png`

---

## Slide 13 — Ablation

<!-- Insert table: experiments/results/summary_tables/attention_ablation.md -->

* Same CNN, BiLSTM, head, split, cap and seeds.
* Only the pooling operator changes.
* Conclusion must follow the numbers; if the gap is inside seed
  variance, say so.

---

## Slide 14 — Error analysis

* Residual distribution: early vs late bias.
* RMSE by horizon (short ≤50 / medium 51–100 / long >100).
* Worst 10 engines.

> Visual: `error_analysis/residual_distribution.png`,
> `error_analysis/error_by_horizon.png`

---

## Slide 15 — Computational cost

<!-- Insert table: experiments/results/summary_tables/cost.md -->

* Latency measured on one device, same input shape.
* Training time with the device of each run.

---

## Slide 16 — Demo

Streamlit app: sample engine or CSV → monitoring → model choice →
RUL estimate → model comparison → sensor trends.

* Only models with real checkpoints are offered.
* Invalid CSV → clear error, no crash.

> Visual: screenshot of the prediction page.

---

## Slide 17 — Limitations

* Single sub-set (one condition, one fault mode).
* Validation = 3 cut-offs per engine.
* Capped targets: no calibration beyond 125 cycles.
* Point predictions, no uncertainty.

---

## Slide 18 — Conclusion

* Reproducible, leakage-free pipeline for FD001.
* Four models compared under one protocol, selected on validation only.
* Ablation answers the attention question with evidence.
* Demo + documented commands for every step.

---

## Slide 19 — Future work

* FD002/FD004 with condition-aware scaling.
* Probabilistic RUL and cost-sensitive training.
* Longer context and per-engine adaptation.

---

## Slide 20 — References

1. Saxena et al., PHM 2008 — C-MAPSS data set and scoring.
2. Heimes, PHM 2008 — LSTM for RUL.
3. Vaswani et al., NeurIPS 2017 — Transformer.
4. Bahdanau et al., ICLR 2015 — additive attention.
5. Breiman, 2001 — Random Forests.

---

### Speaker notes

* Do not quote a metric that is not in `experiments/results/`.
* If a slide table is empty, say "pending" out loud rather than
  estimating.
* The ablation slide is the scientific centre of the talk: keep the
  claim limited to what the seeds support.
