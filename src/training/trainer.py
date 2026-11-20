"""Training loop for RUL regression.

Protocol decisions (documented, not tuned on the test set):

* Optimizer: Adam with a constant learning rate.
* Loss: MSE on RUL targets, optionally capped at ``rul_cap``
  (standard C-MAPSS practice uses 125 cycles: beyond that the exact
  remaining life is not identifiable from the sensors).
* Validation metrics are always computed against the *uncapped* true
  RUL of the validation endpoints, so reports are comparable across
  capping strategies.
* Model selection and early stopping use validation RMSE only.
* Gradient clipping at norm 1.0.
"""

import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.evaluation.metrics import regression_metrics


def prepare_training_targets(y, rul_cap=None):
    """Optionally cap training targets without modifying the original dataset labels."""
    if rul_cap is None:
        return y
    if rul_cap <= 0:
        raise ValueError("rul_cap must be positive")
    return y.clamp(max=float(rul_cap))


def set_seed(seed: int) -> None:
    """Best-effort determinism across Python, NumPy and PyTorch."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _loader_generator(loader, seed: int) -> None:
    """Seed the sampler of an existing DataLoader in place."""

    generator = torch.Generator()
    generator.manual_seed(seed)

    if getattr(loader, "sampler", None) is not None and hasattr(
        loader.sampler, "generator"
    ):
        loader.sampler.generator = generator

    loader.generator = generator


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
        rul_cap: float | None = None,
        model_name: str | None = None,
        model_kwargs: dict | None = None,
        seed: int = 42,
        config: dict | None = None,
        weight_decay: float = 0.0,
        grad_clip: float = 1.0,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.epochs = epochs
        self.patience = patience
        if rul_cap is not None and rul_cap <= 0:
            raise ValueError("rul_cap must be positive")
        self.rul_cap = rul_cap

        self.model_name = model_name
        self.model_kwargs = model_kwargs or {}
        self.seed = seed
        self.grad_clip = grad_clip

        _loader_generator(train_loader, seed)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.criterion = nn.MSELoss()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if config is not None:
            (self.output_dir / "config.json").write_text(
                json.dumps(config, indent=2),
                encoding="utf-8",
            )

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
            training_targets = prepare_training_targets(
                y,
                self.rul_cap,
            )
            loss = self.criterion(predictions, training_targets)

            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")

            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip,
                )

            self.optimizer.step()

            count = x.size(0)
            total_loss += loss.item() * count
            total_samples += count

        return total_loss / total_samples

    @torch.inference_mode()
    def evaluate(self, loader=None):
        loader = loader or self.val_loader

        self.model.eval()

        targets = []
        predictions = []
        engine_ids = []
        cycles = []

        for batch in loader:
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

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

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
                "val_nasa_score_mean": metrics["nasa_score_mean"],
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
                        "model_name": self.model_name,
                        "model_kwargs": {
                            "input_size": self.model.input_size,
                            **self.model_kwargs,
                        },
                        "model_state_dict": best_state,
                        "epoch": best_epoch,
                        "val_metrics": metrics,
                        "rul_cap": self.rul_cap,
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

        best_metrics, best_predictions = self.evaluate()

        peak_memory_mb = None
        if torch.cuda.is_available():
            peak_memory_mb = (
                torch.cuda.max_memory_allocated() / (1024 ** 2)
            )

        summary = {
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "best_val_rmse": best_rmse,
            "best_metrics": best_metrics,
            "training_seconds": elapsed,
            "seconds_per_epoch": elapsed / max(len(history), 1),
            "parameter_count": sum(
                p.numel() for p in self.model.parameters()
            ),
            "device": str(self.device),
            "rul_cap": self.rul_cap,
            "peak_gpu_memory_mb": peak_memory_mb,
            "seed": self.seed,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        (self.output_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "validation": best_metrics,
                    "best_epoch": best_epoch,
                    "rul_cap": self.rul_cap,
                    "model": self.model_name,
                    "seed": self.seed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print("\n=== TRAINING SUMMARY ===")
        print(json.dumps(summary, indent=2))

        return summary
