"""Random Forest baseline.

The baseline is deliberately simple: statistics extracted from each
causal window (last value, mean, standard deviation, change) are fed to
a Random Forest. It exists to answer "does deep learning actually add
value on FD001?".

The baseline uses the same split, the same validation endpoints, the
same RUL cap (125) and the same metric implementation as the deep
models.
"""

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.datasets.datamodule import CMapssDataset
from src.evaluation.metrics import regression_metrics
from src.experiments.registry import ROOT, append_record
from src.models.baseline import build_baseline, extract_window_features


CHECKPOINT_PATH = ROOT / "checkpoints" / "random_forest.joblib"
RUNS_DIR = ROOT / "experiments" / "runs"
RESULTS_DIR = ROOT / "experiments" / "results"


def get_features(dataset):
    return extract_window_features(
        dataset.x.numpy(),
        dataset.mask.numpy(),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train the Random Forest baseline"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--rul-cap", type=float, default=125.0)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--tag", default="main")
    parser.add_argument("--n-estimators", type=int, default=150)

    args = parser.parse_args()

    experiment_id = (
        args.experiment_id
        or f"random_forest_cap{int(args.rul_cap)}_seed{args.seed}"
    )

    run_dir = RUNS_DIR / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== RANDOM FOREST BASELINE ===")

    train = CMapssDataset("train", window_size=args.window_size)
    validation = CMapssDataset(
        "val_endpoints", window_size=args.window_size
    )

    X_train = get_features(train)
    y_train = train.y.numpy().ravel()

    X_val = get_features(validation)
    y_val = validation.y.numpy().ravel()

    if args.rul_cap is not None:
        y_train_capped = np.minimum(y_train, args.rul_cap)
    else:
        y_train_capped = y_train

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("RUL cap:", args.rul_cap)

    model = build_baseline(seed=args.seed)
    model.set_params(n_estimators=args.n_estimators)

    start = time.perf_counter()
    model.fit(X_train, y_train_capped)
    train_seconds = time.perf_counter() - start

    predictions = np.maximum(model.predict(X_val), 0.0)

    metrics = regression_metrics(y_val, predictions)

    print("\n=== VALIDATION RESULTS ===")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name.upper()}: {value:.4f}")
        else:
            print(f"{name.upper()}: {value}")

    predictions_df = pd.DataFrame({
        "engine_id": validation.engine_ids.numpy(),
        "cycle": validation.cycles.numpy(),
        "true_rul": y_val,
        "predicted_rul": predictions,
        "error": predictions - y_val,
    })

    predictions_df.to_csv(
        run_dir / "best_predictions.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Checkpoint (model + inference metadata in one file)
    # ---------------------------------------------------------

    joblib.dump(
        {
            "model": model,
            "model_name": "random_forest",
            "seed": args.seed,
            "window_size": args.window_size,
            "rul_cap": args.rul_cap,
            "features": train.features,
            "statistics": ["last", "mean", "std", "change"],
            "model_kwargs": {
                "n_estimators": args.n_estimators,
                "min_samples_leaf": 3,
                "max_features": 1.0,
            },
        },
        CHECKPOINT_PATH,
        compress=3,
    )

    config = {
        "experiment_id": experiment_id,
        "tag": args.tag,
        "model": "random_forest",
        "model_kwargs": {
            "n_estimators": args.n_estimators,
            "min_samples_leaf": 3,
            "max_features": 1.0,
        },
        "window_size": args.window_size,
        "rul_cap": args.rul_cap,
        "seed": args.seed,
        "learning_rate": None,
        "weight_decay": None,
        "batch_size": None,
        "epochs": 1,
        "patience": None,
        "features": train.features,
        "feature_count": len(train.features),
        "train_windows": len(train),
        "val_samples": len(validation),
        "notes": "statistical window features + RandomForestRegressor",
    }

    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    summary = {
        "best_epoch": 1,
        "epochs_ran": 1,
        "best_val_rmse": metrics["rmse"],
        "best_metrics": metrics,
        "training_seconds": train_seconds,
        "seconds_per_epoch": train_seconds,
        "parameter_count": int(
            sum(
                estimator.tree_.node_count
                for estimator in model.estimators_
            )
        ),
        "device": "cpu",
        "rul_cap": args.rul_cap,
        "peak_gpu_memory_mb": None,
        "seed": args.seed,
    }

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    (run_dir / "metrics.json").write_text(
        json.dumps(
            {"validation": metrics, "model": "random_forest"},
            indent=2,
        ),
        encoding="utf-8",
    )

    pd.DataFrame([{
        "epoch": 1,
        "train_mse": float(
            np.mean((model.predict(X_train) - y_train_capped) ** 2)
        ),
        "val_rmse": metrics["rmse"],
        "val_mae": metrics["mae"],
        "val_bias": metrics["bias"],
        "val_nasa_score": metrics["nasa_score"],
        "duration_seconds": train_seconds,
    }]).to_csv(run_dir / "training_history.csv", index=False)

    append_record({
        "experiment_id": experiment_id,
        "tag": args.tag,
        "model": "random_forest",
        "dataset": "FD001",
        "split": "engine_wise_80_20_seed42",
        "window_size": args.window_size,
        "rul_cap": args.rul_cap,
        "seed": args.seed,
        "learning_rate": None,
        "weight_decay": None,
        "batch_size": None,
        "epochs_requested": 1,
        "epochs_ran": 1,
        "best_epoch": 1,
        "optimizer": "closed_form",
        "loss": "MSE",
        "patience": None,
        "train_windows": len(train),
        "val_samples": len(validation),
        "train_seconds": round(train_seconds, 2),
        "seconds_per_epoch": round(train_seconds, 3),
        "parameters": summary["parameter_count"],
        "device": "cpu",
        "val_rmse": metrics["rmse"],
        "val_mae": metrics["mae"],
        "val_bias": metrics["bias"],
        "val_r2": metrics["r2"],
        "val_nasa_score": metrics["nasa_score"],
        "val_nasa_score_mean": metrics["nasa_score_mean"],
        "early_stopped": False,
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "checkpoint": CHECKPOINT_PATH.relative_to(ROOT).as_posix(),
        "config_path": (run_dir / "config.json").relative_to(ROOT).as_posix(),
        "status": "completed",
        "notes": "statistical window features + RandomForestRegressor",
    })

    print("\nBaseline training completed!")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Run directory:", run_dir)


if __name__ == "__main__":
    main()
