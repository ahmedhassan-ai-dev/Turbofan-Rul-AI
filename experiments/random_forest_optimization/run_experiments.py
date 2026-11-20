"""Random Forest hyperparameter optimization experiments.

Runs a controlled grid search over the specified parameter grid using only
the training/validation engine-wise split. Preserves the original baseline
configuration and data pipeline unchanged.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.datasets.datamodule import CMapssDataset
from src.evaluation.metrics import regression_metrics
from src.experiments.registry import append_record, ROOT
from src.models.baseline import build_baseline, extract_window_features


# Fixed configuration (same as original baseline)
WINDOW_SIZE = 30
RUL_CAP = 125.0
SEED = 42

# Experiment grid
PARAM_GRID = {
    "n_estimators": [150, 300, 500],
    "max_features": [1.0, "sqrt", 0.7],
    "min_samples_leaf": [1, 2, 3, 5],
    "max_depth": [None, 10, 20],
}

# Derive the full grid
import itertools

param_names = list(PARAM_GRID.keys())
param_values = list(PARAM_GRID.values())
full_grid = [dict(zip(param_names, combo)) for combo in itertools.product(*param_values)]

print(f"Total experiments: {len(full_grid)}")
print(f"Parameter grid: {PARAM_GRID}")


def get_features(dataset):
    return extract_window_features(
        dataset.x.numpy(),
        dataset.mask.numpy(),
    )


def run_experiment(config):
    """Run a single RF experiment with given hyperparameters."""
    experiment_id = config.get(
        "experiment_id", f"rf_{config['n_estimators']}e_{config['max_features']}_msl{config['min_samples_leaf']}_md{config['max_depth']}_seed{SEED}"
    )

    run_dir = Path("experiments/random_forest_optimization/runs") / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    RESULTS_DIR = Path("experiments/random_forest_optimization/results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    CHECKPOINT_PATH = Path("checkpoints/random_forest_optimized.joblib")

    print(f"\n=== RF EXPERIMENT: {experiment_id} ===")

    # Data: use the same engine-wise split as the baseline
    train = CMapssDataset("train", window_size=WINDOW_SIZE)
    validation = CMapssDataset("val_endpoints", window_size=WINDOW_SIZE)

    X_train = get_features(train)
    y_train = train.y.numpy().ravel()

    X_val = get_features(validation)
    y_val = validation.y.numpy().ravel()

    if RUL_CAP is not None:
        y_train_capped = np.minimum(y_train, RUL_CAP)
    else:
        y_train_capped = y_train

    print(f"Training shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")

    # Build model with config params
    model = build_baseline(seed=SEED)

    # Set params
    set_kwargs = {}
    if "n_estimators" in config:
        set_kwargs["n_estimators"] = config["n_estimators"]
    if "max_depth" in config:
        set_kwargs["max_depth"] = config["max_depth"]
    if "min_samples_leaf" in config:
        set_kwargs["min_samples_leaf"] = config["min_samples_leaf"]
    if "max_features" in config:
        set_kwargs["max_features"] = config["max_features"]

    model.set_params(**set_kwargs)

    start = time.perf_counter()
    model.fit(X_train, y_train_capped)
    train_seconds = time.perf_counter() - start

    predictions = np.maximum(model.predict(X_val), 0.0)

    metrics = regression_metrics(y_val, predictions)

    print(f"Validation results:")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name.upper()}: {value:.6f}")
        else:
            print(f"  {name.upper()}: {value}")

    # Save predictions
    predictions_df = pd.DataFrame({
        "engine_id": validation.engine_ids.numpy(),
        "cycle": validation.cycles.numpy(),
        "true_rul": y_val,
        "predicted_rul": predictions,
        "error": predictions - y_val,
    })

    pred_path = run_dir / "best_predictions.csv"
    predictions_df.to_csv(pred_path, index=False)

    # Save checkpoint
    joblib.dump({
        "model": model,
        "model_name": "random_forest_optimized",
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "rul_cap": RUL_CAP,
        "features": train.features,
        "statistics": ["last", "mean", "std", "change"],
        "model_kwargs": {
            "n_estimators": config["n_estimators"],
            "min_samples_leaf": config["min_samples_leaf"],
            "max_features": config["max_features"],
        },
    }, CHECKPOINT_PATH, compress=3)

    # Save config
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Save summary
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
        "rul_cap": RUL_CAP,
        "peak_gpu_memory_mb": None,
        "seed": SEED,
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps({"validation": metrics, "model": "random_forest_optimized"}, indent=2), encoding="utf-8")

    training_history = pd.DataFrame([{
        "epoch": 1,
        "train_mse": float(np.mean((model.predict(X_train) - y_train_capped) ** 2)),
        "val_rmse": metrics["rmse"],
        "val_mae": metrics["mae"],
        "val_bias": metrics["bias"],
        "val_nasa_score": metrics["nasa_score"],
        "duration_seconds": train_seconds,
    }])
    training_history.to_csv(run_dir / "training_history.csv", index=False)

    # Append to registry
    append_record({
        "experiment_id": experiment_id,
        "tag": f"rf_optimization_{config['n_estimators']}e_{config['max_features']}_msl{config['min_samples_leaf']}_md{config['max_depth']}",
        "model": "random_forest_optimized",
        "dataset": "FD001",
        "split": "engine_wise_80_20_seed42",
        "window_size": WINDOW_SIZE,
        "rul_cap": RUL_CAP,
        "seed": SEED,
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
        "notes": f"RF hyperparameter experiment: {config}",
    })

    print(f"  -> val_rmse: {metrics['rmse']:.6f}")

    return {
        "experiment_id": experiment_id,
        "config": config,
        "metrics": metrics,
        "train_seconds": train_seconds,
    }


results = []

for i, config in enumerate(full_grid):
    result = run_experiment(config)
    results.append(result)

# After all experiments, produce summary CSV
df = pd.DataFrame([{
    "experiment_id": r["experiment_id"],
    "n_estimators": r["config"]["n_estimators"],
    "max_features": str(r["config"]["max_features"]),
    "min_samples_leaf": r["config"]["min_samples_leaf"],
    "max_depth": r["config"]["max_depth"],
    "val_rmse": r["metrics"]["rmse"],
    "val_mae": r["metrics"]["mae"],
    "val_r2": r["metrics"]["r2"],
    "val_nasa_score": r["metrics"]["nasa_score"],
    "val_nasa_score_mean": r["metrics"]["nasa_score_mean"],
    "train_seconds": r["train_seconds"],
} for r in results])

csv_path = Path("experiments/random_forest_optimization/rf_optimization_results.csv")
df.to_csv(csv_path, index=False)

print(f"\n=== ALL EXPERIMENTS COMPLETE ===")
print(f"Results saved to: {csv_path}")
print(f"\nSummary table (sorted by val_rmse):")
df_sorted = df.sort_values("val_rmse")
print(df_sorted.to_string(index=False))

# Find best config
best = df_sorted.nsmallest(1, "val_rmse").iloc[0]
print(f"\nBest config by val_rmse:")
print(f"  experiment_id={best['experiment_id']}")
print(f"  n_estimators={best['n_estimators']}, max_features={best['max_features']}, min_samples_leaf={best['min_samples_leaf']}, max_depth={best['max_depth']}")
print(f"  val_rmse={best['val_rmse']:.6f}, val_mae={best['val_mae']:.6f}, val_r2={best['val_r2']:.4f}, val_nasa_score_mean={best['val_nasa_score_mean']:.4f}")