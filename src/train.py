
import argparse
import json
from pathlib import Path

import torch

from src.datasets.datamodule import create_dataloader
from src.models.cnn_bilstm_attention import CNNBiLSTMAttention
from src.models.lstm import RULLSTM
from src.models.transformer import RULTransformer
from src.training.trainer import RULTrainer, set_seed


ROOT = Path(__file__).resolve().parents[1]


def build_model(model_name: str, input_size: int):
    """Create a model using the shared training interface."""

    if model_name == "lstm":
        return RULLSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
        )

    if model_name == "transformer":
        return RULTransformer(
            input_size=input_size,
            d_model=64,
            num_heads=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.1,
        )

    if model_name == "cnn_bilstm_attention":
        return CNNBiLSTMAttention(
            input_size=input_size,
            cnn_channels=64,
            hidden_size=64,
            num_layers=2,
            attention_dim=64,
            dropout=0.2,
        )

    raise ValueError(f"Unsupported model: {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Train RUL prediction models"
    )

    parser.add_argument(
        "--model",
        choices=[
            "lstm",
            "transformer",
            "cnn_bilstm_attention",
        ],
        default="lstm",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--rul-cap",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    train_loader = create_dataloader(
        "train",
        batch_size=args.batch_size,
        window_size=30,
        seed=args.seed,
    )

    val_loader = create_dataloader(
        "val_endpoints",
        batch_size=args.batch_size,
        window_size=30,
        seed=args.seed,
    )

    input_size = len(train_loader.dataset.features)

    model = build_model(
        model_name=args.model,
        input_size=input_size,
    )

    output_dir = args.output_dir

    if output_dir is None:
        cap_label = (
            f"cap{int(args.rul_cap)}"
            if args.rul_cap is not None
            else "uncapped"
        )
        output_dir = (
            f"experiments/runs/"
            f"{args.model}_{cap_label}_seed{args.seed}"
        )

    output_path = Path(output_dir)

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    trainer = RULTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=5,
        rul_cap=args.rul_cap,
        output_dir=str(output_path),
    )

    print(f"\n=== {args.model.upper()} TRAINING ===")
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Training windows:", len(train_loader.dataset))
    print("Validation endpoints:", len(val_loader.dataset))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
    print("RUL cap:", args.rul_cap)
    print("Output:", output_path)

    summary = trainer.fit()

    # Store the experiment configuration separately.
    config = {
        "model": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "seed": args.seed,
        "rul_cap": args.rul_cap,
        "window_size": 30,
        "input_size": input_size,
        "parameters": summary["parameter_count"],
    }

    (output_path / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
