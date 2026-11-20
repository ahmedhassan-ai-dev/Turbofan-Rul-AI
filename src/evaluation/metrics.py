"""Unified regression metrics for NASA C-MAPSS RUL prediction.

All models (Random Forest baseline, LSTM, Transformer,
CNN-BiLSTM-Attention) are scored with the functions in this module so
that every number in the report comes from one implementation.

NASA scoring function
---------------------
The asymmetric score used by the C-MAPSS benchmark penalises late RUL
predictions (predicting more remaining life than the engine really has)
more heavily than early predictions:

    d = y_pred - y_true
    s(d) = exp(-d / 13) - 1   if d < 0   (early prediction)
    s(d) = exp( d / 10) - 1   if d >= 0  (late prediction)

Reference: A. Saxena, K. Goebel, D. Simon and N. Eklund,
"Damage propagation modeling for aircraft engine run-to-failure
simulation", PHM 2008, and the C-MAPSS scoring description distributed
with the NASA PCoE Turbofan Engine Degradation data set.
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

EARLY_DENOMINATOR = 13.0
LATE_DENOMINATOR = 10.0


def _validate_pair(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("Invalid prediction or target shapes")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("Non-finite predictions or targets")

    return y_true, y_pred


def nasa_scores(y_true, y_pred) -> np.ndarray:
    """Per-sample asymmetric NASA penalty terms."""

    y_true, y_pred = _validate_pair(y_true, y_pred)

    error = y_pred - y_true

    return np.where(
        error < 0,
        np.exp(-error / EARLY_DENOMINATOR) - 1.0,
        np.exp(error / LATE_DENOMINATOR) - 1.0,
    )


def nasa_score(y_true, y_pred) -> float:
    """Total (summed) NASA score, as defined by the benchmark."""

    return float(nasa_scores(y_true, y_pred).sum())


def regression_metrics(y_true, y_pred) -> dict:
    """Evaluate RUL predictions using consistent metrics.

    Returns
    -------
    rmse, mae, bias : float
    r2 : float | None  (None when the targets are constant)
    nasa_score : float  (sum over samples, official definition)
    nasa_score_mean : float  (sum / N, the value usually reported in papers)
    n_samples : int
    """

    y_true, y_pred = _validate_pair(y_true, y_pred)

    error = y_pred - y_true
    scores = nasa_scores(y_true, y_pred)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "bias": float(error.mean()),
        "r2": (
            float(r2_score(y_true, y_pred))
            if np.unique(y_true).size > 1
            else None
        ),
        "nasa_score": float(scores.sum()),
        "nasa_score_mean": float(scores.mean()),
        "n_samples": int(y_true.size),
    }
