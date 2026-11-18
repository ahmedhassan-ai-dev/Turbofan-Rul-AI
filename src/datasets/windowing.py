
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

WINDOW_SIZE = 30


def create_windows(
    df: pd.DataFrame,
    features: list[str],
    window_size: int = WINDOW_SIZE,
) -> dict[str, np.ndarray]:
    """
    Generate causal sliding windows for C-MAPSS.

    Input:
        DataFrame with engine_id, cycle, features, rul

    Output:
        X:       [N, window_size, num_features]
        y:       [N]
        mask:    [N, window_size]
        engines: [N]
        cycles:  [N]

    Each window contains only the current and past cycles.
    """

    if window_size < 1:
        raise ValueError("window_size must be positive")

    required = ["engine_id", "cycle", "rul"] + features

    if len(set(features)) != len(features):
        raise ValueError("Duplicate feature names")

    if not features:
        raise ValueError("No features provided")

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df.empty or df[required].isna().any().any():
        raise ValueError("Empty dataset or missing values")

    X, y, masks, engines, cycles = [], [], [], [], []

    for engine_id, group in df.groupby("engine_id", sort=True):
        group = group.sort_values("cycle")

        cycle_values = group["cycle"].to_numpy()

        if (
            np.any(cycle_values < 1)
            or np.any(np.diff(cycle_values) != 1)
            or cycle_values[0] != 1
        ):
            raise ValueError(
                f"Invalid cycle sequence for engine {engine_id}"
            )

        values = group[features].to_numpy(dtype=np.float32)
        targets = group["rul"].to_numpy(dtype=np.float32)

        if not np.isfinite(values).all():
            raise ValueError("Non-finite feature values")

        if not np.isfinite(targets).all() or (targets < 0).any():
            raise ValueError("Invalid RUL targets")

        for end in range(len(group)):
            start = max(0, end - window_size + 1)

            window = values[start:end + 1]
            valid_length = len(window)
            pad_length = window_size - valid_length

            # Repeat the earliest observed reading on the left.
            # This never uses future sensor measurements.
            if pad_length > 0:
                padding = np.repeat(
                    window[:1],
                    pad_length,
                    axis=0,
                )
                window = np.concatenate(
                    [padding, window],
                    axis=0,
                )

            # True = real observation, False = padding.
            mask = np.concatenate([
                np.zeros(pad_length, dtype=bool),
                np.ones(valid_length, dtype=bool),
            ])

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

    print("\nAll window generation checks passed!")


if __name__ == "__main__":
    main()
