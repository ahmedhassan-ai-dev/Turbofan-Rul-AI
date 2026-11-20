"""CLI training entry point.

Examples
--------
Smoke test (few epochs, not a final result):

    python -m src.train --model lstm --epochs 2 --rul-cap 125 \
        --experiment-id lstm_smoke --tag smoke

Final run:

    python -m src.train --model lstm --epochs 60 --patience 10 \
        --rul-cap 125 --seed 42 --experiment-id lstm_cap125_seed42
"""

import argparse
import json
from pathlib import Path

from src.experiments.runner import run_experiment
from src.models.factory import build_model


ROOT = Path(__file__).resolve().parents[1]

MODEL_CHOICES = [
    "lstm",
    "transformer",
    "cnn_bilstm_attention",
    "cnn_bilstm_mean",
]


def main():
    parser = argparse.ArgumentParser(
        description="Train RUL prediction models"
    )

    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="lstm",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rul-cap", type=float, default=125.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--experiment-id", type=str, default=None)
    parser.add_argument("--tag", type=str, default="main")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default="{}",
        help="JSON overrides for the architecture",
    )
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    model_kwargs = json.loads(args.model_kwargs)

    if args.experiment_id is None:
        cap_label = (
            f"cap{int(args.rul_cap)}"
            if args.rul_cap is not None
            else "uncapped"
        )
        experiment_id = (
            f"{args.model}_{cap_label}_seed{args.seed}"
        )
    else:
        experiment_id = args.experiment_id

    config = {
        "experiment_id": experiment_id,
        "tag": args.tag,
        "model": args.model,
        "model_kwargs": model_kwargs,
        "window_size": args.window_size,
        "rul_cap": args.rul_cap,
        "seed": args.seed,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": args.device,
        "notes": args.notes,
    }

    if args.output_dir is not None:
        config["experiment_id"] = experiment_id

    summary = run_experiment(config)

    print("\nRun directory:", summary["run_dir"])


if __name__ == "__main__":
    main()
