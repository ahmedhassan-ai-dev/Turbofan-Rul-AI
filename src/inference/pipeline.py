"""Raw sensor input -> model-ready windows.

Used by the Streamlit demo and by the official test pipeline. All
validation errors raise ``InputValidationError`` with a message that can
be shown directly to a user, so a bad CSV never crashes the app.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.datasets.windowing import create_last_windows, create_windows


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
FEATURE_COLUMNS = SETTINGS + SENSORS

RAW_COLUMNS = ["engine_id", "cycle"] + FEATURE_COLUMNS


class InputValidationError(ValueError):
    """Raised when user-supplied sensor data cannot be used."""


def load_artifacts() -> tuple[dict, object]:
    """Load preprocessing metadata and the fitted training scaler."""

    metadata_path = PROCESSED_DIR / "fd001_metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Preprocessing artifacts missing. Run "
            "`python -m src.datasets.preprocessing` first."
        )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    scaler = joblib.load(
        PROCESSED_DIR / metadata["scaler_file"]
    )

    return metadata, scaler


def validate_raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a user-provided C-MAPSS style dataframe."""

    if df is None or len(df) == 0:
        raise InputValidationError("The input file contains no rows.")

    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    if "engine_id" not in df.columns:
        df["engine_id"] = 1

    if "cycle" not in df.columns:
        df["cycle"] = np.arange(1, len(df) + 1)

    missing = [
        column for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:
        preview = ", ".join(missing[:5])
        raise InputValidationError(
            "Missing required columns: "
            f"{preview}{'...' if len(missing) > 5 else ''}. "
            "Expected the 3 operational settings (setting_1..setting_3) "
            "and 21 sensors (sensor_1..sensor_21)."
        )

    numeric_columns = ["engine_id", "cycle"] + FEATURE_COLUMNS

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[list(FEATURE_COLUMNS)].isna().any().any():
        raise InputValidationError(
            "Some sensor values are not numeric or are empty."
        )

    if df[["engine_id", "cycle"]].isna().any().any():
        raise InputValidationError(
            "engine_id and cycle must be numeric."
        )

    df = df.dropna(subset=FEATURE_COLUMNS)

    if (df["engine_id"] <= 0).any() or (df["cycle"] <= 0).any():
        raise InputValidationError(
            "engine_id and cycle must be positive integers."
        )

    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    if df.duplicated(["engine_id", "cycle"]).any():
        raise InputValidationError(
            "Duplicate (engine_id, cycle) rows were found."
        )

    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise InputValidationError(
            "Sensor data contains non-finite values."
        )

    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    for engine_id, group in df.groupby("engine_id"):
        cycles = group["cycle"].to_numpy()
        if np.any(np.diff(cycles) != 1):
            raise InputValidationError(
                f"Engine {engine_id} has non-consecutive cycles."
            )

    return df


def prepare_scaled_frame(
    df: pd.DataFrame,
    metadata: dict,
    scaler,
) -> pd.DataFrame:
    """Apply the training feature selection and scaler to raw input."""

    df = validate_raw_frame(df)

    features = metadata["selected_features"]

    required = ["engine_id", "cycle"] + features

    if set(required) - set(df.columns):
        raise InputValidationError(
            "Input does not contain the training features."
        )

    df = df.copy()
    df[features] = scaler.transform(df[features])

    return df[required]


def windows_from_frame(
    df: pd.DataFrame,
    features: list[str],
    window_size: int,
    last_only: bool = True,
) -> dict[str, np.ndarray]:
    """Build inference windows from a scaled dataframe."""

    if last_only:
        return create_last_windows(
            df, features=features, window_size=window_size
        )

    return create_windows(df, features=features, window_size=window_size)


def load_sample_engine(
    engine_id: int,
    split: str = "test",
    last_cycles: int | None = None,
) -> pd.DataFrame:
    """Return one engine's history from the FD001 raw files."""

    raw_path = ROOT / "data" / "raw" / f"{split}_FD001.txt"

    if not raw_path.is_file():
        raise FileNotFoundError(
            f"Raw data not available: {raw_path}. "
            "Place the NASA C-MAPSS files in data/raw/."
        )

    df = pd.read_csv(
        raw_path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
    )

    subset = df[df["engine_id"] == engine_id]

    if subset.empty:
        raise InputValidationError(
            f"Engine {engine_id} is not present in {split}_FD001."
        )

    subset = subset.sort_values("cycle")

    if last_cycles is not None:
        subset = subset.tail(last_cycles)

    return subset.reset_index(drop=True)


def list_sample_engines(split: str = "test") -> list[int]:
    """Engine ids available for the demo sample picker."""

    for candidate in ("test", "train"):
        path = ROOT / "data" / "raw" / f"{candidate}_FD001.txt"

        if split != candidate or not path.is_file():
            continue

        ids = pd.read_csv(
            path,
            sep=r"\s+",
            header=None,
            usecols=[0],
        )[0].unique()

        return sorted(int(x) for x in ids)

    return []
