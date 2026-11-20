"""Uniform inference interface for every trained model.

The demo, the official test evaluation and the error analysis all use
this class, so a prediction computed in the app is produced by exactly
the same code path as a prediction in the report.

Two families of checkpoints are supported:

``.joblib`` (scikit-learn Random Forest)
    * dict payload ``{"model": ..., "features": ..., "window_size": ...}``
      as written by ``src/train_baseline.py``;
    * a bare estimator object (``RandomForestRegressor``) as written by
      the Random Forest hyper-parameter study.
    Both were trained on the 4 statistics per feature produced by
    :func:`src.models.baseline.extract_window_features` (17 features ->
    68 model inputs).

``.pt`` (PyTorch)
    * ``{"model_name", "model_kwargs", "model_state_dict", "metadata"}``
      written by :func:`src.models.factory.save_checkpoint`. These models
      consume the raw ``[N, T, F]`` scaled sensor windows directly.
"""

from pathlib import Path

import joblib
import numpy as np
import torch

from src.models.baseline import extract_window_features
from src.models.factory import load_checkpoint


BASELINE_WINDOW_SIZE = 30
STATISTICS_PER_FEATURE = 4


class RULPredictor:
    """Wrap a torch checkpoint or the Random Forest baseline."""

    def __init__(self, checkpoint_path: Path, device: str = "cpu"):
        self.path = Path(checkpoint_path)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {self.path}"
            )

        self.device = device
        self.kind = "unknown"
        self.window_size = BASELINE_WINDOW_SIZE
        self.features = None
        self.model = None
        self.metadata = {}

        if self.path.suffix == ".joblib":
            self._load_sklearn()
        else:
            self._load_torch(device)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    def _load_sklearn(self) -> None:
        """Load a joblib Random Forest in either supported format."""

        try:
            payload = joblib.load(self.path)
        except Exception as error:  # noqa: BLE001
            raise ValueError(
                f"Could not read baseline checkpoint {self.path}: {error}"
            ) from error

        if isinstance(payload, dict) and "model" in payload:
            model = payload["model"]

            if not hasattr(model, "predict"):
                raise ValueError(
                    f"Unsupported baseline checkpoint: {self.path} "
                    "(the 'model' entry has no predict() method)"
                )

            self.kind = "sklearn"
            self.model = model
            self.features = payload.get("features")
            self.window_size = int(
                payload.get("window_size", BASELINE_WINDOW_SIZE)
            )
            self.metadata = {
                key: value
                for key, value in payload.items()
                if key != "model"
            }
            return

        if hasattr(payload, "predict"):
            # Bare estimator, e.g. random_forest_optimized.joblib.
            # The hyper-parameter study saved the estimator alone, so the
            # window size falls back to the protocol default (30).
            self.kind = "sklearn"
            self.model = payload
            self.features = None
            self.window_size = BASELINE_WINDOW_SIZE
            self.metadata = {"checkpoint_format": "bare_estimator"}
            return

        raise ValueError(
            f"Unsupported baseline checkpoint: {self.path} "
            f"(loaded a {type(payload).__name__}, expected a "
            "RandomForestRegressor or a dict with a 'model' entry)"
        )

    def _load_torch(self, device: str) -> None:
        """Load one of the three deep models from a .pt checkpoint."""

        try:
            model, metadata = load_checkpoint(self.path, device=device)
        except FileNotFoundError:
            raise
        except Exception as error:  # noqa: BLE001
            raise ValueError(
                f"Could not load model checkpoint {self.path}: {error}"
            ) from error

        self.kind = "torch"
        self.model = model
        self.metadata = metadata
        self.window_size = int(
            metadata.get("window_size", BASELINE_WINDOW_SIZE)
        )
        self.features = metadata.get("features")

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    @property
    def display_name(self) -> str:
        if self.kind == "sklearn":
            return "random_forest"

        return self.metadata.get("model", "torch_model")

    def predict_windows(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        batch_size: int = 256,
    ) -> np.ndarray:
        """Predict RUL from [N, T, F] windows and a [N, T] mask."""

        x = np.asarray(x, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)

        if x.ndim != 3:
            raise ValueError("Expected windows [N, T, F]")

        if mask.shape != x.shape[:2]:
            raise ValueError("Mask shape mismatch")

        if self.kind == "sklearn":
            return self._predict_sklearn(x, mask)

        return self._predict_torch(x, mask, batch_size)

    def _predict_sklearn(
        self,
        x: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Statistical window features -> Random Forest."""

        features = extract_window_features(x, mask)

        expected = getattr(self.model, "n_features_in_", None)

        if expected is not None and features.shape[1] != expected:
            raise ValueError(
                "Checkpoint expects "
                f"{expected} input features but the pipeline produced "
                f"{features.shape[1]}."
            )

        predictions = np.asarray(
            self.model.predict(features),
            dtype=np.float64,
        )

        return np.maximum(predictions, 0.0)

    def _predict_torch(
        self,
        x: np.ndarray,
        mask: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Scaled windows -> deep sequence model, clipped at zero."""

        predictions = []

        self.model.eval()

        with torch.inference_mode():
            for start in range(0, len(x), batch_size):
                xb = torch.from_numpy(
                    x[start:start + batch_size]
                ).to(self.device)
                mb = torch.from_numpy(
                    mask[start:start + batch_size]
                ).to(self.device)

                output = self.model(xb, mb).clamp(min=0)
                predictions.append(output.cpu().numpy().reshape(-1))

        return np.maximum(
            np.concatenate(predictions).astype(np.float64),
            0.0,
        )
