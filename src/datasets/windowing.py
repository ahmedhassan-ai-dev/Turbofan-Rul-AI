"""Causal sliding-window generation for C-MAPSS.

Design contract
---------------
* Windows never cross engine boundaries.
* A window for target cycle ``t`` contains cycles
  ``max(1, t - window_size + 1) .. t`` only: no future sensor readings.
* Windows shorter than ``window_size`` are left-padded by repeating the
  earliest *observed* reading of that engine. Padding is flagged in
  ``mask`` (True = real observation) so every model can ignore it.
* The target of a window is the RUL at its last (current) cycle.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

WINDOW_SIZE = 30


def _pad_left(
    window: np.ndarray,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Left-pad a [T, F] window by repeating its first row.

    Returns the padded window and a boolean mask where True marks real
    observations.
    """

    valid_length = len(window)
    pad_length = window_size - valid_length

    if pad_length > 0:
        padding = np.repeat(window[:1], pad_length, axis=0)
        window = np.concatenate([padding, window], axis=0)

    mask = np.concatenate([
        np.zeros(pad_length, dtype=bool),
        np.ones(valid_length, dtype=bool),
    ])

    return window, mask


def _validate_frame(
    df: pd.DataFrame,
    features: list[str],
    require_target: bool,
) -> list[str]:
    if not features:
        raise ValueError("No features provided")

    if len(set(features)) != len(features):
        raise ValueError("Duplicate feature names")

    required = ["engine_id", "cycle"] + features

    if require_target:
        required = required + ["rul"]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty or df[required].isna().any().any():
        raise ValueError("Empty dataset or missing values")

    return required


def _check_engine_cycles(
    group: pd.DataFrame,
    engine_id,
    require_start_at_one: bool = True,
) -> np.ndarray:
    """Cycles must be positive and consecutive.

    Training/validation trajectories always start at cycle 1. Inference
    input may be a trailing slice of a trajectory (a user uploading only
    the recent history), so it only has to be consecutive.
    """

    cycle_values = group["cycle"].to_numpy()

    if np.any(cycle_values < 1) or np.any(np.diff(cycle_values) != 1):
        raise ValueError(f"Invalid cycle sequence for engine {engine_id}")

    if require_start_at_one and cycle_values[0] != 1:
        raise ValueError(f"Invalid cycle sequence for engine {engine_id}")

    return cycle_values


def create_windows(
    df: pd.DataFrame,
    features: list[str],
    window_size: int = WINDOW_SIZE,
) -> dict[str, np.ndarray]:
    """Generate one causal window per (engine, cycle) row.

    Returns
    -------
    X:       [N, window_size, num_features]
    y:       [N]            RUL at the last cycle of the window
    mask:    [N, window_size]
    engines: [N]
    cycles:  [N]
    """

    if window_size < 1:
        raise ValueError("window_size must be positive")

    _validate_frame(df, features, require_target=True)

    X, y, masks, engines, cycles = [], [], [], [], []

    for engine_id, group in df.groupby("engine_id", sort=True):
        group = group.sort_values("cycle")
        cycle_values = _check_engine_cycles(group, engine_id)

        values = group[features].to_numpy(dtype=np.float32)
        targets = group["rul"].to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError("Non-finite feature values")

        if not np.isfinite(targets).all() or (targets < 0).any():
            raise ValueError("Invalid RUL targets")

        for end in range(len(group)):
            start = max(0, end - window_size + 1)

            window, mask = _pad_left(
                values[start:end + 1],
                window_size,
            )

            X.append(window)
            y.append(targets[end])
            masks.append(mask)
            engines.append(engine_id)
            cycles.append(cycle_values[end])

    return {
        "X": np.stack(X).astype(np.float32),
        "y": np.asarray(y, dtype=np.float32),
        "mask": np.stack(masks),
        "engines": np.asarray(engines, dtype=np.int32),
        "cycles": np.asarray(cycles, dtype=np.int32),
    }


def create_last_windows(
    df: pd.DataFrame,
    features: list[str],
    window_size: int = WINDOW_SIZE,
) -> dict[str, np.ndarray]:
    """One window per engine, ending at its last available cycle.

    This is the official C-MAPSS test-time protocol: the model may only
    use the sensor history that the test trajectory provides.
    """

    if window_size < 1:
        raise ValueError("window_size must be positive")

    _validate_frame(df, features, require_target=False)

    X, masks, engines, cycles, lengths = [], [], [], [], []

    for engine_id, group in df.groupby("engine_id", sort=True):
        group = group.sort_values("cycle")
        cycle_values = _check_engine_cycles(
            group,
            engine_id,
            require_start_at_one=False,
        )

        values = group[features].to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError("Non-finite feature values")

        start = max(0, len(group) - window_size)

        window, mask = _pad_left(values[start:], window_size)

        X.append(window)
        masks.append(mask)
        engines.append(engine_id)
        cycles.append(cycle_values[-1])
        lengths.append(int(mask.sum()))

    return {
        "X": np.stack(X).astype(np.float32),
        "mask": np.stack(masks),
        "engines": np.asarray(engines, dtype=np.int32),
        "cycles": np.asarray(cycles, dtype=np.int32),
        "lengths": np.asarray(lengths, dtype=np.int32),
    }


def main():
    import json

    metadata_path = DATA_DIR / "fd001_metadata.json"
    metadata = json.loads(metadata_path.read_text("utf-8"))

    features = metadata["selected_features"]

    for split in ["train", "val"]:
        path = DATA_DIR / f"{split}_FD001.csv"
        df = pd.read_csv(path)

        data = create_windows(
            df,
            features=features,
            window_size=WINDOW_SIZE,
        )

        print(f"\n=== {split.upper()} WINDOWS ===")
        print("X shape:", data["X"].shape)
        print("y shape:", data["y"].shape)
        print("Mask shape:", data["mask"].shape)
        print("Unique engines:", len(np.unique(data["engines"])))
        print("First sample cycle:", data["cycles"][0])
        print("First sample RUL:", data["y"][0])
        print("First sample valid steps:", data["mask"][0].sum())
        print("Last sample cycle:", data["cycles"][-1])

        assert len(data["X"]) == len(df)
        assert data["X"].shape[1:] == (WINDOW_SIZE, len(features))
        assert np.isfinite(data["X"]).all()

    test_df = pd.read_csv(DATA_DIR / "test_FD001.csv")
    test_data = create_last_windows(
        test_df,
        features=features,
        window_size=WINDOW_SIZE,
    )

    print("\n=== TEST LAST WINDOWS ===")
    print("X shape:", test_data["X"].shape)
    print("Engines:", len(test_data["engines"]))

    assert len(test_data["X"]) == test_df["engine_id"].nunique()

    print("\nAll window generation checks passed!")


if __name__ == "__main__":
    main()
