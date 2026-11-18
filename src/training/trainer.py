
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluation.metrics import regression_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RULTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        learning_rate: float = 0.001,
        epochs: int = 3,
        patience: int = 5,
        device: str | None = None,
        output_dir: str = "experiments/runs/lstm_smoke",
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs
        self.patience = patience

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
        )

        self.criterion = nn.MSELoss()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch in self.train_loader:
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            predictions = self.model(x, mask)
            loss = self.criterion(predictions, y)

            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )

            self.optimizer.step()

            count = x.size(0)
            total_loss += loss.item() * count
            total_samples += count

        return total_loss / total_samples

    @torch.inference_mode()
    def evaluate(self):
        self.model.eval()

        targets = []
        predictions = []
        engine_ids = []
        cycles = []

        for batch in self.val_loader:
            x = batch["x"].to(self.device)
            mask = batch["mask"].to(self.device)

            output = self.model(x, mask)

            # RUL cannot be negative at evaluation time.
            output = output.clamp(min=0)

            predictions.extend(output.cpu().numpy().reshape(-1))
            targets.extend(batch["y"].numpy().reshape(-1))
            engine_ids.extend(batch["engine_id"].numpy().reshape(-1))
            cycles.extend(batch["cycle"].numpy().reshape(-1))

        y_true = np.asarray(targets)
        y_pred = np.asarray(predictions)

        metrics = regression_metrics(y_true, y_pred)

        prediction_table = pd.DataFrame({
            "engine_id": engine_ids,
            "cycle": cycles,
            "true_rul": y_true,
            "predicted_rul": y_pred,
            "error": y_pred - y_true,
        })

        return metrics, prediction_table

    def fit(self):
        history = []
        best_rmse = float("inf")
        best_epoch = 0
        no_improvement = 0
        best_state = None

        start_time = time.perf_counter()

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.perf_counter()

            train_loss = self.train_epoch()
            metrics, predictions = self.evaluate()

            duration = time.perf_counter() - epoch_start

            history.append({
                "epoch": epoch,
                "train_mse": train_loss,
                "val_rmse": metrics["rmse"],
                "val_mae": metrics["mae"],
                "val_bias": metrics["bias"],
                "val_nasa_score": metrics["nasa_score"],
                "duration_seconds": duration,
            })

            print(
                f"Epoch {epoch:03d}/{self.epochs} | "
                f"Train MSE: {train_loss:.4f} | "
                f"Val RMSE: {metrics['rmse']:.4f} | "
                f"MAE: {metrics['mae']:.4f} | "
                f"Time: {duration:.2f}s",
                flush=True,
            )

            if metrics["rmse"] < best_rmse - 1e-6:
                best_rmse = metrics["rmse"]
                best_epoch = epoch
                no_improvement = 0

                best_state = copy.deepcopy(
                    self.model.state_dict()
                )

                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": best_epoch,
                        "val_metrics": metrics,
                    },
                    self.output_dir / "best_model.pt",
                )

                predictions.to_csv(
                    self.output_dir / "best_predictions.csv",
                    index=False,
                )
            else:
                no_improvement += 1

            if no_improvement >= self.patience:
                print("Early stopping triggered.")
                break

        elapsed = time.perf_counter() - start_time

        pd.DataFrame(history).to_csv(
            self.output_dir / "training_history.csv",
            index=False,
        )

        if best_state is None:
            raise RuntimeError("No valid checkpoint produced")

        self.model.load_state_dict(best_state)

        best_metrics, _ = self.evaluate()

        summary = {
            "best_epoch": best_epoch,
            "best_val_rmse": best_rmse,
            "best_metrics": best_metrics,
            "training_seconds": elapsed,
            "parameter_count": sum(
                p.numel() for p in self.model.parameters()
            ),
            "device": str(self.device),
        }

        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("\n=== TRAINING SUMMARY ===")
        print(json.dumps(summary, indent=2))

        return summary

