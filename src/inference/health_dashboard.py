"""Causal, inference-only helpers for the Engine Health Intelligence page.

These functions never fit a scaler, update model weights or read test labels.
The plotted intervals describe disagreement between saved training seeds,
NOT statistically calibrated predictive confidence intervals.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.inference.pipeline import InputValidationError


COMPARABLE_SEED_FAMILIES = frozenset({
    "lstm", "transformer", "cnn_bilstm_attention",
})


def build_causal_timeline_windows(
    scaled_frame: pd.DataFrame,
    features: list[str],
    window_size: int,
    max_points: int = 100,
) -> dict[str, np.ndarray]:
    """Construct complete, trailing windows ending at observed cycles only.

    All plotted windows contain exactly `window_size` observed rows. Each
    window for cycle t ends at t; it cannot see sensor readings after t.
    A trailing sample whose cycle numbering does not start at 1 is valid.
    """
    if window_size < 1 or max_points < 1:
        raise ValueError("window_size and max_points must be positive.")
    if scaled_frame is None or scaled_frame.empty:
        raise InputValidationError("No sensor history available.")
    if scaled_frame["engine_id"].nunique() != 1:
        raise InputValidationError("Analyze one engine at a time.")
    if len(scaled_frame) < window_size:
        raise InputValidationError(
            f"The timeline requires at least {window_size} observed cycles "
            f"for complete model windows; received {len(scaled_frame)}. "
            "Increase 'Cycles to load' or upload a longer history."
        )

    group = scaled_frame.sort_values("cycle")
    cycles = group["cycle"].to_numpy(dtype=np.int64)
    if np.any(np.diff(cycles) != 1):
        raise InputValidationError("Cycle history must be consecutive.")

    values = group[features].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise InputValidationError("Scaled sensor values must be finite.")

    first = max(window_size - 1, len(group) - max_points)
    ends = range(first, len(group))
    windows = np.stack([
        values[end - window_size + 1:end + 1]
        for end in ends
    ]).astype(np.float32)

    return {
        "X": windows,
        "mask": np.ones(windows.shape[:2], dtype=bool),
        "cycles": cycles[first:].astype(np.int32),
    }


def predict_rul_timeline(
    scaled_frame: pd.DataFrame,
    features: list[str],
    predictor,
    max_points: int = 100,
) -> pd.DataFrame:
    """Predict RUL for each displayed cycle using the existing model API."""
    windows = build_causal_timeline_windows(
        scaled_frame, features, predictor.window_size, max_points,
    )
    estimates = np.asarray(
        predictor.predict_windows(windows["X"], windows["mask"]),
        dtype=np.float64,
    ).reshape(-1)

    if len(estimates) != len(windows["cycles"]) or not np.isfinite(estimates).all():
        raise ValueError("Model returned an invalid timeline prediction array.")

    return pd.DataFrame({
        "cycle": windows["cycles"],
        "predicted_rul": estimates,
    })


def main_seed_checkpoints(
    registry: pd.DataFrame,
    family: str,
    project_root: Path,
) -> dict[int, str]:
    """Return available, comparable main-experiment seed checkpoints only."""
    if registry.empty or family not in COMPARABLE_SEED_FAMILIES:
        return {}

    rows = registry.loc[
        (registry["tag"] == "main")
        & (registry["model"] == family)
        & (registry["status"] == "completed")
    ]
    if rows.empty:
        return {}

    # Different target caps or window lengths are different protocols,
    # not independent seed replicates of one model configuration.
    if len(rows[["window_size", "rul_cap"]].drop_duplicates()) != 1:
        return {}

    root = Path(project_root).resolve()
    found: dict[int, str] = {}
    for row in rows.itertuples(index=False):
        checkpoint = (root / str(row.checkpoint)).resolve()
        if checkpoint.is_relative_to(root) and checkpoint.is_file():
            seed = int(row.seed)
            if seed not in found:
                found[seed] = str(checkpoint)
    return dict(sorted(found.items()))


def summarize_seed_disagreement(
    timelines: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Compute descriptive min/mean/max across aligned seed predictions.

    Not a probabilistic interval or a calibrated uncertainty estimate.
    """
    if len(timelines) < 2:
        raise ValueError("At least two seed timelines are needed.")

    curves = list(timelines.values())
    cycles = curves[0]["cycle"].to_numpy()
    estimates = []

    for curve in curves:
        if not np.array_equal(cycles, curve["cycle"].to_numpy()):
            raise ValueError("Seed timelines must share the exact same cycles.")
        values = curve["predicted_rul"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Seed predictions must be finite.")
        estimates.append(values)

    matrix = np.stack(estimates)
    low = matrix.min(axis=0)
    high = matrix.max(axis=0)
    return pd.DataFrame({
        "cycle": cycles,
        "mean_rul": matrix.mean(axis=0),
        "min_rul": low,
        "max_rul": high,
        "spread_rul": high - low,
    })
