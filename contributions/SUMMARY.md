# Contribution Summary

> Template. Fill after the individual logs are complete. Only list work
> that exists in the repository.

| Member | Role | Main deliverables | Experiments owned | Figures / tables |
| --- | --- | --- | --- | --- |
| _TBD_ | Data pipeline | `src/datasets/*` | — | `figures/eda/*` |
| _TBD_ | Modelling | `src/models/*` | — | — |
| _TBD_ | Training / MLOps | `src/training/*`, `src/experiments/*` | — | `registry.csv` |
| _TBD_ | Evaluation | `src/evaluation/*` | — | official test + error analysis |
| _TBD_ | Demo | `demo/app.py` | — | demo screenshots |
| _TBD_ | Documentation | `README.md`, `report/` | — | report + slides |

## How to verify a claimed contribution

1. The file or experiment id must exist in the repository.
2. `git log -- <path>` (or the PR history) must show the author.
3. Experiment ids must appear in `experiments/registry.csv` with
   `status = completed`.
