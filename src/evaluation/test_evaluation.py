"""Official FD001 test-set evaluation.

Protocol
--------
1. Test trajectories (``test_FD001.txt``) are truncated: the last
   recorded cycle is the prediction point.
2. Sensor values are scaled with the *training* scaler and restricted to
   the training feature list (``data/processed/fd001_metadata.json``).
3. Exactly one window per engine is built: the final ``window_size``
   cycles, left-padded when the trajectory is shorter.
4. Ground truth comes from ``RUL_FD001.txt``; row *i* is engine *i+1*.
5. Model selection is done on the validation split only (lowest
   validation RMSE inside the ``main`` experiment tag). The test set is
   used once, after selection, and never for tuning.

Usage
-----
    python -m src.evaluation.test_evaluation --auto
    python -m src.evaluation.test_evaluation \
        --run-dir experiments/runs/lstm_cap125_seed42
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets.preprocessing import load_test_rul
from src.datasets.windowing import create_last_windows
from src.evaluation.metrics import regression_metrics
from src.experiments.registry import ROOT, read_registry
from src.inference.predictor import RULPredictor


PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "experiments" / "results" / "official_test"
FIGURES_DIR = ROOT / "experiments" / "figures" / "official_test"

CAP_FOR_REPORT = 125.0


def load_test_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    metadata_path = PROCESSED_DIR / "fd001_metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Run `python -m src.datasets.preprocessing` first."
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    test_df = pd.read_csv(PROCESSED_DIR / "test_FD001.csv")

    test_rul = load_test_rul(
        ROOT / "data" / "raw" / "RUL_FD001.txt"
    )

    engines = sorted(
        int(x) for x in test_df["engine_id"].unique()
    )

    if engines != test_rul["engine_id"].tolist():
        raise ValueError(
            "Test engine order does not match RUL_FD001.txt"
        )

    return test_df, test_rul, metadata


def build_test_windows(
    window_size: int = 30,
) -> tuple[dict, pd.DataFrame, dict]:
    test_df, test_rul, metadata = load_test_artifacts()

    features = metadata["selected_features"]

    windows = create_last_windows(
        test_df,
        features=features,
        window_size=window_size,
    )

    engines = windows["engines"].tolist()

    if engines != test_rul["engine_id"].tolist():
        raise ValueError(
            "Window engine order does not match test labels"
        )

    return windows, test_rul, metadata


def evaluate_checkpoint(
    checkpoint_path: Path,
    model_label: str,
    window_size: int = 30,
    device: str = "cpu",
    save: bool = True,
) -> dict:
    """Score one checkpoint on the official test set."""

    windows, test_rul, metadata = build_test_windows(window_size)

    predictor = RULPredictor(checkpoint_path, device=device)

    predictions = predictor.predict_windows(
        windows["X"],
        windows["mask"],
    )

    if len(predictions) != len(test_rul):
        raise ValueError(
            "Prediction/label length mismatch on the test set"
        )

    table = pd.DataFrame({
        "engine_id": windows["engines"],
        "last_cycle": windows["cycles"],
        "window_length": windows["lengths"],
        "true_rul": test_rul["rul"].to_numpy(),
        "predicted_rul": predictions,
        "error": predictions - test_rul["rul"].to_numpy(),
    })

    metrics = regression_metrics(
        table["true_rul"],
        table["predicted_rul"],
    )

    capped_true = np.minimum(
        table["true_rul"].to_numpy(), CAP_FOR_REPORT
    )
    capped_pred = np.minimum(
        table["predicted_rul"].to_numpy(), CAP_FOR_REPORT
    )

    capped_metrics = regression_metrics(capped_true, capped_pred)

    payload = {
        "model": model_label,
        "checkpoint": Path(checkpoint_path).as_posix(),
        "dataset": "FD001 official test",
        "window_size": window_size,
        "test_engines": int(len(table)),
        "scaler": metadata["scaler_fit_on"],
        "features": metadata["selected_features"],
        "metrics_uncapped": metrics,
        "metrics_capped_125": capped_metrics,
    }

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        table.to_csv(
            RESULTS_DIR / f"{model_label}_predictions.csv",
            index=False,
        )

        (RESULTS_DIR / f"{model_label}_metrics.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        plot_official_test(table, model_label)

    return payload


def plot_official_test(table: pd.DataFrame, model_label: str) -> None:
    """Two standard views of the official test performance."""

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(
        table["true_rul"],
        table["predicted_rul"],
        alpha=0.7,
        edgecolor="black",
        linewidth=0.4,
    )

    limit = max(
        table["true_rul"].max(),
        table["predicted_rul"].max(),
    ) * 1.05

    ax.plot([0, limit], [0, limit], "--", color="grey", label="Ideal")
    ax.set_xlabel("True RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"Official test — {model_label}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{model_label}_actual_vs_predicted.png",
        dpi=180,
    )
    plt.close(fig)

    ordered = table.sort_values("true_rul").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))

    ax.plot(
        ordered.index,
        ordered["true_rul"],
        marker="o",
        markersize=3,
        label="True RUL",
    )
    ax.plot(
        ordered.index,
        ordered["predicted_rul"],
        marker="x",
        markersize=3,
        label="Predicted RUL",
    )
    ax.set_xlabel("Test engine (sorted by true RUL)")
    ax.set_ylabel("RUL (cycles)")
    ax.set_title(f"Official test — {model_label}: per-engine RUL")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{model_label}_per_engine.png",
        dpi=180,
    )
    plt.close(fig)


def select_main_runs(
    registry: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """Pick one run per model using validation RMSE only."""

    registry = read_registry() if registry is None else registry

    if registry.empty:
        return {}

    main = registry[
        (registry["tag"] == "main")
        & (registry["status"] == "completed")
    ]

    selection: dict[str, dict] = {}

    for model, group in main.groupby("model"):
        best = group.sort_values("val_rmse").iloc[0]

        selection[str(model)] = {
            "experiment_id": best["experiment_id"],
            "run_dir": best["run_dir"],
            "checkpoint": best["checkpoint"],
            "val_rmse": float(best["val_rmse"]),
            "seed": int(best["seed"]),
            "rul_cap": float(best["rul_cap"]),
            "window_size": int(best["window_size"]),
        }

    return selection


def main():
    parser = argparse.ArgumentParser(
        description="Official FD001 test evaluation"
    )
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--run-dir", nargs="*", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    targets: list[tuple[str, Path, int]] = []

    if args.run_dir:
        for run in args.run_dir:
            run_dir = Path(run)

            if not run_dir.is_absolute():
                run_dir = ROOT / run_dir

            config = json.loads(
                (run_dir / "config.json").read_text(encoding="utf-8")
            )

            targets.append((
                config["experiment_id"],
                run_dir / "best_model.pt",
                config.get("window_size", 30),
            ))
    else:
        selection = select_main_runs()

        if not selection:
            raise SystemExit(
                "No completed main runs in the registry. "
                "Train models first or pass --run-dir."
            )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        (RESULTS_DIR / "model_selection.json").write_text(
            json.dumps(
                {
                    "rule": (
                        "lowest validation RMSE among runs tagged "
                        "'main'; the official test set was not used "
                        "for selection"
                    ),
                    "selected": selection,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        for model, info in selection.items():
            checkpoint = ROOT / info["checkpoint"]

            if not checkpoint.is_file():
                print(
                    f"SKIP {model}: missing checkpoint {checkpoint}"
                )
                continue

            window_size = args.window_size or info["window_size"]

            targets.append((model, checkpoint, window_size))

    if not targets:
        raise SystemExit("Nothing to evaluate.")

    summary = []

    for label, checkpoint, window_size in targets:
        print(f"\n=== OFFICIAL TEST: {label} ===")

        payload = evaluate_checkpoint(
            checkpoint,
            model_label=label,
            window_size=window_size,
            device=args.device,
        )

        metrics = payload["metrics_uncapped"]

        print(
            f"RMSE {metrics['rmse']:.3f} | MAE {metrics['mae']:.3f} | "
            f"Bias {metrics['bias']:.3f} | R2 {metrics['r2']:.3f} | "
            f"NASA {metrics['nasa_score']:.2f}"
        )

        summary.append({
            "model": label,
            "experiment_id": label,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "bias": metrics["bias"],
            "r2": metrics["r2"],
            "nasa_score": metrics["nasa_score"],
            "nasa_score_mean": metrics["nasa_score_mean"],
            "rmse_capped_125": payload["metrics_capped_125"]["rmse"],
            "mae_capped_125": payload["metrics_capped_125"]["mae"],
        })

    summary_df = pd.DataFrame(summary).sort_values("rmse")

    summary_df.to_csv(
        RESULTS_DIR / "test_metrics.csv",
        index=False,
    )

    (RESULTS_DIR / "test_summary.json").write_text(
        json.dumps(
            {
                "dataset": "FD001 official test",
                "engines": 100,
                "results": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n=== OFFICIAL TEST SUMMARY ===")
    print(summary_df.to_string(index=False))
    print("\nSaved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
