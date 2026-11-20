"""Verify that every shipped checkpoint loads and predicts for real.

Run from the repository root::

    python scripts/verify_models.py

For each model this script
  1. resolves the checkpoint exactly like the Streamlit app does,
  2. builds a real window from the official FD001 test trajectory with
     the real preprocessing pipeline (scaler + feature selection),
  3. runs inference through ``RULPredictor``,
  4. checks the prediction is finite, non-negative and in cycles.

It also re-derives two numbers that are quoted in the report, so a
mismatch is caught instead of silently shipped:

  * validation RMSE of the original and the optimized Random Forest,
  * official test RMSE of the original Random Forest.
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.datamodule import CMapssDataset  # noqa: E402
from src.datasets.windowing import create_last_windows  # noqa: E402
from src.evaluation.metrics import regression_metrics  # noqa: E402
from src.inference.pipeline import load_artifacts, load_sample_engine, prepare_scaled_frame  # noqa: E402
from src.inference.predictor import RULPredictor  # noqa: E402
from src.models.baseline import extract_window_features  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
CHECKPOINTS = ROOT / "checkpoints"


def app_checkpoint_map() -> dict[str, Path]:
    """Mirror demo/app.py::available_checkpoints()."""

    found: dict[str, Path] = {}

    baseline = CHECKPOINTS / "random_forest.joblib"
    if baseline.is_file():
        found["random_forest"] = baseline

    optimized = CHECKPOINTS / "random_forest_optimized.joblib"
    if optimized.is_file():
        found["optimized_random_forest"] = optimized

    registry = pd.read_csv(ROOT / "experiments" / "registry.csv")
    main = registry[(registry["tag"] == "main") & (registry["status"] == "completed")]

    for model, group in main.groupby("model"):
        best = group.sort_values("val_rmse").iloc[0]
        checkpoint = ROOT / str(best["checkpoint"])
        if checkpoint.is_file():
            found[str(model)] = checkpoint

    return found


def sample_windows(metadata: dict, engine: int = 1, cycles: int = 60):
    raw = load_sample_engine(engine, "test", cycles)
    scaled = prepare_scaled_frame(raw, metadata, joblib.load(
        PROCESSED / metadata["scaler_file"]
    ))

    return create_last_windows(
        scaled,
        features=metadata["selected_features"],
        window_size=30,
    )


def main() -> int:
    metadata, scaler = load_artifacts()
    windows = sample_windows(metadata)
    failures: list[str] = []

    print("=" * 74)
    print("PER-MODEL INFERENCE (official FD001 test engine 1, last 60 cycles)")
    print("=" * 74)

    for label, path in app_checkpoint_map().items():
        try:
            predictor = RULPredictor(path)
            predictions = predictor.predict_windows(windows["X"], windows["mask"])
            value = float(predictions[0])

            ok = (
                np.isfinite(predictions).all()
                and (predictions >= 0).all()
                and len(predictions) == 1
            )

            print(
                f"{label:26s} kind={predictor.kind:8s} "
                f"window={predictor.window_size:2d} "
                f"RUL={value:8.3f} cycles  "
                f"{'OK' if ok else 'FAIL'}"
            )

            if not ok:
                failures.append(f"{label}: invalid prediction {predictions}")
        except Exception as error:  # noqa: BLE001
            failures.append(f"{label}: {type(error).__name__}: {error}")
            print(f"{label:26s} FAILED -> {type(error).__name__}: {error}")

    # ------------------------------------------------------------------
    # reproduce reported numbers
    # ------------------------------------------------------------------
    # Validation is scored on the 60 predefined endpoints, exactly like
    # src/train_baseline.py does (split="val_endpoints").
    validation = CMapssDataset("val_endpoints", window_size=30)
    x_val = extract_window_features(
        validation.x.numpy(),
        validation.mask.numpy(),
    )
    y_val = validation.y.numpy().ravel()

    print()
    print("=" * 74)
    print("VALIDATION RMSE CHECK (recomputed from the shipped checkpoint)")
    print("=" * 74)

    expected_val = {
        "random_forest": 18.549619037185813,
        "optimized_random_forest": 16.517202247166853,
    }

    for label, expected in expected_val.items():
        path = app_checkpoint_map()[label]
        predictor = RULPredictor(path)
        predicted = np.maximum(predictor.model.predict(x_val), 0.0)
        rmse = regression_metrics(y_val, predicted)["rmse"]
        delta = abs(rmse - expected)
        status = "OK" if delta < 1e-6 else "MISMATCH"
        print(f"{label:26s} val RMSE={rmse:.9f} expected={expected:.9f} {status}")
        if delta >= 1e-6:
            failures.append(f"{label}: validation RMSE drift {delta}")

    print()
    print("=" * 74)
    print("OFFICIAL TEST RMSE CHECK (original Random Forest)")
    print("=" * 74)

    from src.evaluation.test_evaluation import evaluate_checkpoint  # noqa: E402

    payload = evaluate_checkpoint(
        CHECKPOINTS / "random_forest.joblib",
        model_label="random_forest",
        window_size=30,
        save=False,
    )
    expected_test = 18.348562769729035
    delta = abs(payload["metrics_uncapped"]["rmse"] - expected_test)
    print(
        f"random_forest test RMSE={payload['metrics_uncapped']['rmse']:.9f} "
        f"expected={expected_test:.9f} "
        f"{'OK' if delta < 1e-9 else 'MISMATCH'}"
    )
    if delta >= 1e-9:
        failures.append(f"random_forest: official test RMSE drift {delta}")

    print()
    if failures:
        print("FAILURES:")
        for failure in failures:
            print(" -", failure)
        return 1

    print(f"ALL {len(app_checkpoint_map())} MODELS PREDICT SUCCESSFULLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
