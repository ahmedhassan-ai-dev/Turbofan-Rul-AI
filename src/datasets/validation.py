
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.windowing import create_windows


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"
CONFIG_DIR = ROOT / "configs" / "data"

VAL_PATH = PROCESSED_DIR / "val_FD001.csv"
METADATA_PATH = PROCESSED_DIR / "fd001_metadata.json"
ENDPOINTS_PATH = CONFIG_DIR / "fd001_validation_endpoints.csv"

WINDOW_SIZE = 30
RUL_HORIZONS = (100, 60, 30)


def create_validation_endpoints(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = RUL_HORIZONS,
) -> pd.DataFrame:
    """
    Define deterministic validation endpoints.

    For each validation engine, select earlier cycles
    with a known remaining life.

    The future is used only to define evaluation labels,
    never as model input.
    """

    records = []

    for engine_id, group in df.groupby("engine_id"):
        group = group.sort_values("cycle")

        if group["cycle"].duplicated().any():
            raise ValueError("Duplicate cycles detected")

        lifetime = int(group["cycle"].max())

        for horizon in horizons:
            cutoff_cycle = lifetime - horizon

            if cutoff_cycle < 1:
                raise ValueError(
                    f"Engine {engine_id} too short "
                    f"for RUL horizon {horizon}"
                )

            endpoint = group.loc[
                group["cycle"] == cutoff_cycle
            ]

            if len(endpoint) != 1:
                raise ValueError(
                    f"Missing endpoint for engine {engine_id}"
                )

            actual_rul = int(endpoint["rul"].iloc[0])

            if actual_rul != horizon:
                raise ValueError(
                    f"RUL mismatch: engine {engine_id}, "
                    f"expected {horizon}, got {actual_rul}"
                )

            records.append({
                "engine_id": int(engine_id),
                "cycle": cutoff_cycle,
                "rul": actual_rul,
            })

    return pd.DataFrame(records).sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)


def main():
    df = pd.read_csv(VAL_PATH)

    metadata = json.loads(
        METADATA_PATH.read_text(encoding="utf-8")
    )

    features = metadata["selected_features"]

    endpoints = create_validation_endpoints(df)

    if df["engine_id"].nunique() != 20:
        raise ValueError("Expected 20 validation engines")

    if len(endpoints) != 60:
        raise ValueError("Expected 60 validation endpoints")

    # Generate causal windows using the established pipeline.
    windows = create_windows(
        df,
        features=features,
        window_size=WINDOW_SIZE,
    )

    # Match each endpoint to its input window.
    window_lookup = {
        (int(engine), int(cycle)): idx
        for idx, (engine, cycle) in enumerate(
            zip(windows["engines"], windows["cycles"])
        )
    }

    indices = []

    for row in endpoints.itertuples(index=False):
        key = (row.engine_id, row.cycle)

        if key not in window_lookup:
            raise ValueError(f"Window not found: {key}")

        idx = window_lookup[key]

        if not np.isclose(windows["y"][idx], row.rul):
            raise ValueError(f"Target mismatch: {key}")

        indices.append(idx)

    X = windows["X"][indices]
    y = windows["y"][indices]
    masks = windows["mask"][indices]

    assert X.shape == (60, WINDOW_SIZE, len(features))
    assert y.shape == (60,)
    assert masks.shape == (60, WINDOW_SIZE)
    assert np.isfinite(X).all()

    # Save the evaluation protocol, not large tensors.
    ENDPOINTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    if ENDPOINTS_PATH.exists():
        existing = pd.read_csv(ENDPOINTS_PATH)

        if not existing.equals(endpoints):
            raise ValueError(
                "Existing validation protocol differs. "
                "Do not overwrite it silently."
            )
    else:
        endpoints.to_csv(ENDPOINTS_PATH, index=False)

    print("\n=== VALIDATION PROTOCOL ===")
    print("Engines:", endpoints["engine_id"].nunique())
    print("Endpoints:", len(endpoints))
    print("Window shape:", X.shape)
    print("Targets:", sorted(y.astype(int).tolist()[:3]))
    print("\nRUL horizon distribution:")
    print(endpoints["rul"].value_counts().sort_index())
    print("\nSaved:", ENDPOINTS_PATH)
    print("\nValidation endpoint checks passed!")


if __name__ == "__main__":
    main()
