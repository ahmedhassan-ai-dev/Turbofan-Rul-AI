
import argparse
from pathlib import Path

import torch

from src.datasets.datamodule import create_dataloader
from src.models.lstm import RULLSTM
from src.training.trainer import RULTrainer, set_seed


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/runs/lstm_smoke",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    train_loader = create_dataloader(
        "train",
        batch_size=args.batch_size,
        seed=args.seed,
    )

    val_loader = create_dataloader(
        "val_endpoints",
        batch_size=args.batch_size,
        seed=args.seed,
    )

    input_size = len(train_loader.dataset.features)

    model = RULLSTM(
        input_size=input_size,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )

    output_path = Path(args.output_dir)

    if not output_path.is_absolute():
        output_path = ROOT / output_path

    trainer = RULTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.lr,
        epochs=args.epochs,
        patience=5,
        output_dir=str(output_path),
    )

    print("\n=== LSTM TRAINING ===")
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Training windows:", len(train_loader.dataset))
    print("Validation endpoints:", len(val_loader.dataset))
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    trainer.fit()


if __name__ == "__main__":
    main()
