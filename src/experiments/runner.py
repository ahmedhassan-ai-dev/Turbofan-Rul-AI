"""Single-experiment runner.

A run is only registered after it finishes and writes its checkpoint,
history, predictions and metrics. Configuration is written before
training starts so a crashed run still leaves a trace of what was
attempted.
"""

import json
import sys
from pathlib import Path

import torch

from src.datasets.datamodule import create_dataloader
from src.experiments.registry import append_record, record_from_run
from src.models.factory import (
    build_model,
    model_size_mb,
    resolve_architecture,
)
from src.training.trainer import RULTrainer, set_seed


ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "experiments" / "runs"


DEFAULT_CONFIG = {
    "experiment_id": "lstm_smoke",
    "tag": "smoke",
    "model": "lstm",
    "model_kwargs": {},
    "dataset": "FD001",
    "split": "engine_wise_80_20_seed42",
    "window_size": 30,
    "rul_cap": 125,
    "seed": 42,
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "batch_size": 64,
    "epochs": 60,
    "patience": 10,
    "device": None,
    "notes": "",
}


def build_loaders(config: dict) -> tuple:
    train_loader = create_dataloader(
        "train",
        batch_size=config["batch_size"],
        window_size=config["window_size"],
        seed=config["seed"],
    )

    val_loader = create_dataloader(
        "val_endpoints",
        batch_size=config["batch_size"],
        window_size=config["window_size"],
        seed=config["seed"],
    )

    return train_loader, val_loader


def run_experiment(config: dict, register: bool = True) -> dict:
    """Train one configuration and persist every artifact."""

    config = {**DEFAULT_CONFIG, **config}

    experiment_id = config["experiment_id"]

    set_seed(config["seed"])

    run_dir = RUNS_DIR / experiment_id

    train_loader, val_loader = build_loaders(config)

    input_size = len(train_loader.dataset.features)

    model = build_model(
        config["model"],
        input_size=input_size,
        **config["model_kwargs"],
    )

    architecture, model_kwargs = resolve_architecture(
        config["model"],
        **config["model_kwargs"],
    )

    full_config = {
        **config,
        "input_size": input_size,
        "architecture": architecture,
        "resolved_model_kwargs": model_kwargs,
        "train_windows": len(train_loader.dataset),
        "val_samples": len(val_loader.dataset),
        "features": train_loader.dataset.features,
        "feature_count": input_size,
        "parameter_count": sum(
            p.numel() for p in model.parameters()
        ),
        "model_size_mb": round(model_size_mb(model), 4),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "cpu"
        ),
        "python": sys.version.split()[0],
    }

    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(full_config, indent=2),
        encoding="utf-8",
    )

    trainer = RULTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["learning_rate"],
        epochs=config["epochs"],
        patience=config["patience"],
        device=config["device"],
        rul_cap=config["rul_cap"],
        output_dir=str(run_dir),
        model_name=config["model"],
        model_kwargs=model_kwargs,
        seed=config["seed"],
        config=full_config,
        weight_decay=config["weight_decay"],
    )

    print(f"\n=== {experiment_id} ===")
    print("Model:", config["model"], "| params:", full_config["parameter_count"])
    print("Windows:", len(train_loader.dataset), "| cap:", config["rul_cap"])
    print("Output:", run_dir)

    summary = trainer.fit()

    if register:
        append_record(record_from_run(run_dir))

    summary["run_dir"] = str(run_dir)

    return summary


def run_from_config_file(path: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding="utf-8"))

    return run_experiment(config)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python -m src.experiments.runner <config.json>"
        )

    run_from_config_file(Path(sys.argv[1]))
