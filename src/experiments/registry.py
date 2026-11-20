"""Experiment registry.

Every completed training run appends exactly one row here. The registry
is the single source of truth for "what was actually run", so the report
can never quote a number that has no run behind it.
"""

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "experiments" / "registry.csv"
RUNS_DIR = ROOT / "experiments" / "runs"

FIELDS = [
    "experiment_id",
    "tag",
    "model",
    "dataset",
    "split",
    "window_size",
    "rul_cap",
    "seed",
    "learning_rate",
    "weight_decay",
    "batch_size",
    "epochs_requested",
    "epochs_ran",
    "best_epoch",
    "optimizer",
    "loss",
    "patience",
    "train_windows",
    "val_samples",
    "train_seconds",
    "seconds_per_epoch",
    "parameters",
    "device",
    "val_rmse",
    "val_mae",
    "val_bias",
    "val_r2",
    "val_nasa_score",
    "val_nasa_score_mean",
    "early_stopped",
    "run_dir",
    "checkpoint",
    "config_path",
    "status",
    "notes",
]


def read_registry(path: Path = REGISTRY_PATH) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=FIELDS)

    df = pd.read_csv(path)
    df["experiment_id"] = df["experiment_id"].astype(str)

    return df


def append_record(record: dict, path: Path = REGISTRY_PATH) -> Path:
    """Insert or replace one record, keeping a stable column order."""

    missing = set(FIELDS) - set(record)

    if missing:
        raise ValueError(f"Missing registry fields: {sorted(missing)}")

    path.parent.mkdir(parents=True, exist_ok=True)

    df = read_registry(path)

    if not df.empty:
        df = df[df["experiment_id"] != str(record["experiment_id"])]

    df = pd.concat(
        [df, pd.DataFrame([{key: record[key] for key in FIELDS}])],
        ignore_index=True,
    )

    df = df.sort_values(["tag", "model", "seed"]).reset_index(drop=True)

    df.to_csv(path, index=False)

    return path


def record_from_run(run_dir: Path, extra: dict | None = None) -> dict:
    """Rebuild a registry record from a run directory on disk."""

    run_dir = Path(run_dir)

    config = json.loads(
        (run_dir / "config.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (run_dir / "summary.json").read_text(encoding="utf-8")
    )
    metrics = summary["best_metrics"]

    history_path = run_dir / "training_history.csv"
    early_stopped = False

    if history_path.is_file():
        history = pd.read_csv(history_path)
        early_stopped = len(history) < config["epochs"]

    record = {
        "experiment_id": config["experiment_id"],
        "tag": config.get("tag", "main"),
        "model": config["model"],
        "dataset": config.get("dataset", "FD001"),
        "split": config.get("split", "engine_wise_80_20_seed42"),
        "window_size": config["window_size"],
        "rul_cap": config["rul_cap"],
        "seed": config["seed"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config.get("weight_decay", 0.0),
        "batch_size": config["batch_size"],
        "epochs_requested": config["epochs"],
        "epochs_ran": summary["epochs_ran"],
        "best_epoch": summary["best_epoch"],
        "optimizer": "Adam",
        "loss": "MSE",
        "patience": config.get("patience", 5),
        "train_windows": config.get("train_windows"),
        "val_samples": config.get("val_samples"),
        "train_seconds": round(summary["training_seconds"], 2),
        "seconds_per_epoch": round(summary["seconds_per_epoch"], 3),
        "parameters": summary["parameter_count"],
        "device": summary["device"],
        "val_rmse": metrics["rmse"],
        "val_mae": metrics["mae"],
        "val_bias": metrics["bias"],
        "val_r2": metrics["r2"],
        "val_nasa_score": metrics["nasa_score"],
        "val_nasa_score_mean": metrics["nasa_score_mean"],
        "early_stopped": early_stopped,
        "run_dir": run_dir.relative_to(ROOT).as_posix(),
        "checkpoint": (run_dir / "best_model.pt").relative_to(ROOT).as_posix(),
        "config_path": (run_dir / "config.json").relative_to(ROOT).as_posix(),
        "status": "completed",
        "notes": config.get("notes", ""),
    }

    if extra:
        record.update(extra)

    return record
