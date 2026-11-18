
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from src.datasets.datamodule import CMapssDataset
from src.models.baseline import (
    build_baseline,
    extract_window_features,
)


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "experiments" / "results"

SEED = 42
WINDOW_SIZE = 30


def get_features(dataset):
    x = dataset.x.numpy()
    masks = dataset.mask.numpy()

    return extract_window_features(x, masks)


def calculate_metrics(y_true, y_pred):
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main():
    print("\n=== RANDOM FOREST BASELINE ===")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load established data protocol.
    train = CMapssDataset(
        "train",
        window_size=WINDOW_SIZE,
    )

    validation = CMapssDataset(
        "val_endpoints",
        window_size=WINDOW_SIZE,
    )

    # Convert windows to statistical features.
    X_train = get_features(train)
    y_train = train.y.numpy().ravel()

    X_val = get_features(validation)
    y_val = validation.y.numpy().ravel()

    print("Training shape:", X_train.shape)
    print("Validation shape:", X_val.shape)

    # Train baseline.
    model = build_baseline(seed=SEED)

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # Evaluate ONLY on validation endpoints.
    predictions = np.maximum(
        model.predict(X_val),
        0.0,
    )

    metrics = calculate_metrics(y_val, predictions)

    print("\n=== VALIDATION RESULTS ===")
    for name, value in metrics.items():
        print(f"{name.upper()}: {value:.4f}")

    # Save predictions for error analysis.
    predictions_df = pd.DataFrame({
        "engine_id": validation.engine_ids.numpy(),
        "cycle": validation.cycles.numpy(),
        "true_rul": y_val,
        "predicted_rul": predictions,
        "error": predictions - y_val,
    })

    predictions_df.to_csv(
        RESULTS_DIR / "random_forest_predictions.csv",
        index=False,
    )

    # Report performance for each RUL horizon.
    print("\n=== PER-HORIZON RESULTS ===")

    for horizon, group in predictions_df.groupby("true_rul"):
        horizon_metrics = calculate_metrics(
            group["true_rul"].to_numpy(),
            group["predicted_rul"].to_numpy(),
        )

        # R2 is undefined for constant ground-truth targets.
        print(
            f"RUL {int(horizon):3d} | "
            f"RMSE: {horizon_metrics['rmse']:.4f} | "
            f"MAE: {horizon_metrics['mae']:.4f}"
        )

    # Save model with inference metadata.
    checkpoint = {
        "model": model,
        "model_name": "random_forest",
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "features": train.features,
        "statistics": ["last", "mean", "std", "change"],
    }

    joblib.dump(
        checkpoint,
        CHECKPOINT_DIR / "random_forest.joblib",
    )

    results = {
        "model": "random_forest",
        "dataset": "FD001",
        "seed": SEED,
        "window_size": WINDOW_SIZE,
        "train_samples": len(train),
        "validation_samples": len(validation),
        "metrics": metrics,
    }

    (RESULTS_DIR / "random_forest_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\nBaseline training completed!")
    print("Model and validation results saved.")


if __name__ == "__main__":
    main()
