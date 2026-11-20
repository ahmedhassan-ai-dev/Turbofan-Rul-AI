"""Computational cost analysis.

All latency numbers are measured on the SAME device, with the SAME
input shape (T=30, F=17) and the same number of repetitions, so the
comparison is fair. Training times are taken from the experiment
registry, which stores the device used for each run.

Usage
-----
    python -m src.evaluation.cost_analysis
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.experiments.registry import ROOT, read_registry
from src.inference.predictor import RULPredictor
from src.models.factory import model_size_mb
from src.reporting.tables import to_markdown


RESULTS_DIR = ROOT / "experiments" / "results" / "cost"
FIGURES_DIR = ROOT / "experiments" / "figures" / "cost"

WINDOW_SIZE = 30
FEATURE_COUNT = 17
BATCH_SIZES = [1, 100]
WARMUP = 5
REPETITIONS = 20


def benchmark_torch(
    model,
    device: torch.device,
    batch_size: int,
) -> float:
    x = torch.randn(
        batch_size, WINDOW_SIZE, FEATURE_COUNT,
        device=device,
    )
    mask = torch.ones(
        batch_size, WINDOW_SIZE,
        dtype=torch.bool,
        device=device,
    )

    model.eval()

    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x, mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        for _ in range(REPETITIONS):
            model(x, mask)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    return (elapsed / REPETITIONS) * 1000.0


def benchmark_sklearn(predictor, batch_size: int) -> float:
    x = np.random.randn(
        batch_size, WINDOW_SIZE, FEATURE_COUNT
    ).astype(np.float32)
    mask = np.ones((batch_size, WINDOW_SIZE), dtype=bool)

    for _ in range(WARMUP):
        predictor.predict_windows(x, mask)

    start = time.perf_counter()

    for _ in range(REPETITIONS):
        predictor.predict_windows(x, mask)

    elapsed = time.perf_counter() - start

    return (elapsed / REPETITIONS) * 1000.0


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    registry = read_registry()

    if registry.empty:
        raise SystemExit(
            "Empty registry: train at least one model first."
        )

    main_runs = registry[
        (registry["tag"] == "main")
        & (registry["status"] == "completed")
    ]

    if main_runs.empty:
        raise SystemExit("No completed 'main' experiments found.")

    best = (
        main_runs.sort_values("val_rmse")
        .groupby("model", as_index=False)
        .first()
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    rows = []

    for record in best.itertuples(index=False):
        checkpoint = ROOT / record.checkpoint

        if not checkpoint.is_file():
            print(f"SKIP {record.model}: missing {checkpoint}")
            continue

        predictor = RULPredictor(
            checkpoint,
            device=str(device),
        )

        if predictor.kind == "torch":
            parameters = sum(
                p.numel() for p in predictor.model.parameters()
            )
            size_mb = model_size_mb(predictor.model)
            latency = {
                batch: benchmark_torch(
                    predictor.model, device, batch
                )
                for batch in BATCH_SIZES
            }
        else:
            parameters = int(
                sum(
                    tree.tree_.node_count
                    for tree in predictor.model.estimators_
                )
            )
            size_mb = checkpoint.stat().st_size / (1024 ** 2)
            latency = {
                batch: benchmark_sklearn(predictor, batch)
                for batch in BATCH_SIZES
            }

        throughput = (
            1000.0 / latency[1] if latency[1] > 0 else float("nan")
        )

        rows.append({
            "model": record.model,
            "experiment_id": record.experiment_id,
            "parameters": parameters,
            "size_mb": round(size_mb, 3),
            "training_seconds": float(record.train_seconds),
            "training_device": record.device,
            "latency_ms_batch1": round(latency[1], 3),
            "latency_ms_batch100": round(latency[100], 3),
            "throughput_engines_per_s": round(throughput, 1),
            "benchmark_device": str(device),
            "val_rmse": float(record.val_rmse),
        })

    table = pd.DataFrame(rows).sort_values("parameters")

    table.to_csv(RESULTS_DIR / "cost_table.csv", index=False)

    (RESULTS_DIR / "cost_table.md").write_text(
        to_markdown(table),
        encoding="utf-8",
    )

    (RESULTS_DIR / "benchmark_config.json").write_text(
        json.dumps({
            "window_size": WINDOW_SIZE,
            "feature_count": FEATURE_COUNT,
            "batch_sizes": BATCH_SIZES,
            "warmup": WARMUP,
            "repetitions": REPETITIONS,
            "device": str(device),
            "device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu"
            ),
            "note": (
                "Latency for every model was measured on the same "
                "device. Training times come from the registry and "
                "record the device each run used."
            ),
        }, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    table.plot(
        kind="bar", x="model", y="parameters",
        ax=axes[0], legend=False, color="#55A868",
    )
    axes[0].set_title("Parameters")
    axes[0].set_ylabel("count")
    axes[0].set_xlabel("")

    table.plot(
        kind="bar", x="model", y="latency_ms_batch100",
        ax=axes[1], legend=False, color="#4C72B0",
    )
    axes[1].set_title("Inference latency (batch 100)")
    axes[1].set_ylabel("ms")
    axes[1].set_xlabel("")

    for ax in axes:
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(
        FIGURES_DIR / "cost_comparison.png",
        dpi=180,
    )
    plt.close(fig)

    print("\n=== COMPUTATIONAL COST ===")
    print(table.to_string(index=False))
    print("\nSaved:", RESULTS_DIR)


if __name__ == "__main__":
    main()
