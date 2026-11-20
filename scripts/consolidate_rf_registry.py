"""One-off maintenance script: consolidate the Random Forest hyper-parameter
sweep in experiments/registry.csv.

Why this is needed
------------------
experiments/random_forest_optimization/run_experiments.py appended one
registry row per grid point (40 rows) and each row stored an ABSOLUTE
Windows path to the same checkpoint file plus an absolute config path.
That is not portable (the registry is the single source of truth for a
reproducible submission) and it claimed forty different checkpoints where
only the winner is shipped.

What this script does
---------------------
1. Writes the full sweep to experiments/random_forest_optimization/sweep_results.csv
   (relative run paths) so no experiment data is lost.
2. Replaces the 40 rows with ONE canonical row for the winning
   configuration, using repository-relative paths.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "experiments" / "registry.csv"
SWEEP_OUT = ROOT / "experiments" / "random_forest_optimization" / "sweep_results.csv"

WINNER_ID = "rf_150e_sqrt_msl1_md20_seed42"
WINNER_RUN_DIR = f"experiments/random_forest_optimization/runs/{WINNER_ID}"


def portable(value: str) -> str:
    return str(value).replace("\\", "/")


def main() -> None:
    registry = pd.read_csv(REGISTRY)

    sweep = registry[registry["model"] == "random_forest_optimized"].copy()

    if sweep.empty:
        print("No optimization rows in the registry; nothing to consolidate.")
        return

    # 1. keep the sweep as its own artifact
    preserved = sweep.drop(columns=["checkpoint", "config_path"]).copy()
    preserved["run_dir"] = preserved["run_dir"].map(portable)
    preserved = preserved.sort_values("val_rmse")
    SWEEP_OUT.write_text(preserved.to_csv(index=False), encoding="utf-8")

    # 2. one canonical row for the shipped checkpoint
    winner = sweep[sweep["experiment_id"] == WINNER_ID]

    if len(winner) != 1:
        raise SystemExit(f"Expected exactly one {WINNER_ID} row, got {len(winner)}")

    row = winner.iloc[0].to_dict()
    row["tag"] = "rf_optimization"
    row["run_dir"] = WINNER_RUN_DIR
    row["checkpoint"] = "checkpoints/random_forest_optimized.joblib"
    row["config_path"] = f"{WINNER_RUN_DIR}/config.json"
    row["notes"] = (
        "Random Forest hyper-parameter winner (validation RMSE "
        f"{float(row['val_rmse']):.4f}); selected on the validation split only. "
        "Full grid in experiments/random_forest_optimization/sweep_results.csv"
    )

    kept = registry[registry["model"] != "random_forest_optimized"].copy()
    updated = pd.concat([kept, pd.DataFrame([row])], ignore_index=True)
    updated = updated.sort_values(["tag", "model", "seed"]).reset_index(drop=True)
    updated.to_csv(REGISTRY, index=False)

    print(f"Preserved {len(preserved)} sweep rows -> {SWEEP_OUT.relative_to(ROOT)}")
    print(f"Registry rows: {len(registry)} -> {len(updated)}")


if __name__ == "__main__":
    main()
