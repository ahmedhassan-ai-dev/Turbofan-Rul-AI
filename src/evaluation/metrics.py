
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true, y_pred):
    """Evaluate RUL predictions using consistent metrics."""

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Invalid prediction or target shapes")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Non-finite predictions or targets")

    error = y_pred - y_true

    # Conventional C-MAPSS asymmetric score:
    # Early prediction: error < 0, denominator 13.
    # Late prediction: error >= 0, denominator 10.
    score = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    ).sum()

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(error.mean()),
        "r2": (
            float(r2_score(y_true, y_pred))
            if np.unique(y_true).size > 1
            else None
        ),
        "nasa_score": float(score),
    }
