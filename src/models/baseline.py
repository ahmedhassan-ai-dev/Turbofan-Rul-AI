
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def extract_window_features(
    windows: np.ndarray,
    masks: np.ndarray,
) -> np.ndarray:
    """
    Extract statistical features from temporal windows.

    windows: [N, T, F]
    masks:   [N, T], True for real observations

    Returns: [N, F * 4]

    Features per sensor:
        1. Last observed value
        2. Mean
        3. Standard deviation
        4. Change (last - first)
    """

    windows = np.asarray(windows, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)

    if windows.ndim != 3:
        raise ValueError("Expected windows [N, T, F]")

    if masks.shape != windows.shape[:2]:
        raise ValueError("Mask shape mismatch")

    if not masks.any(axis=1).all():
        raise ValueError("Every window needs valid observations")

    if not np.isfinite(windows).all():
        raise ValueError("Non-finite sensor values")

    # Exclude padding from statistics.
    weights = masks[..., None]
    counts = masks.sum(axis=1, keepdims=True).astype(np.float32)

    means = (windows * weights).sum(axis=1) / counts

    variance = (
        ((windows - means[:, None, :]) ** 2) * weights
    ).sum(axis=1) / counts

    stds = np.sqrt(np.maximum(variance, 0.0))

    first_indices = masks.argmax(axis=1)
    last_indices = (
        masks.shape[1] - 1
        - masks[:, ::-1].argmax(axis=1)
    )

    indices = np.arange(len(windows))

    first = windows[indices, first_indices]
    last = windows[indices, last_indices]

    changes = last - first

    return np.concatenate(
        [last, means, stds, changes],
        axis=1,
    ).astype(np.float32)


def build_baseline(seed: int = 42) -> RandomForestRegressor:
    """Build initial Random Forest baseline."""

    return RandomForestRegressor(
        n_estimators=150,
        max_depth=None,
        min_samples_leaf=3,
        max_features=1.0,
        random_state=seed,
        n_jobs=-1,
    )
