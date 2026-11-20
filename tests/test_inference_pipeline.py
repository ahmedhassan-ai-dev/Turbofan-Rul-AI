"""Input validation and demo inference tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.inference.pipeline import (
    FEATURE_COLUMNS,
    InputValidationError,
    load_sample_engine,
    validate_raw_frame,
)


ROOT = Path(__file__).resolve().parents[1]


def make_frame(rows: int = 5, engine_id: int = 1) -> pd.DataFrame:
    data = {
        "engine_id": [engine_id] * rows,
        "cycle": list(range(1, rows + 1)),
    }

    for index, column in enumerate(FEATURE_COLUMNS):
        data[column] = np.linspace(0.0, 1.0, rows) + index

    return pd.DataFrame(data)


def test_valid_frame_passes():
    df = validate_raw_frame(make_frame())

    assert len(df) == 5
    assert df["cycle"].tolist() == [1, 2, 3, 4, 5]


def test_missing_columns_rejected():
    df = make_frame().drop(columns=["sensor_7"])

    with pytest.raises(InputValidationError) as error:
        validate_raw_frame(df)

    assert "sensor_7" in str(error.value)


def test_non_numeric_values_rejected():
    df = make_frame().astype({"sensor_2": object})
    df.loc[2, "sensor_2"] = "broken"

    with pytest.raises(InputValidationError):
        validate_raw_frame(df)


def test_duplicate_engine_cycle_rejected():
    df = make_frame()
    df.loc[3, "cycle"] = 3

    with pytest.raises(InputValidationError):
        validate_raw_frame(df)


def test_non_consecutive_cycles_rejected():
    df = make_frame()
    df.loc[4, "cycle"] = 9

    with pytest.raises(InputValidationError):
        validate_raw_frame(df)


def test_empty_frame_rejected():
    with pytest.raises(InputValidationError):
        validate_raw_frame(pd.DataFrame())


def test_missing_ids_are_generated():
    df = make_frame().drop(columns=["engine_id", "cycle"])

    validated = validate_raw_frame(df)

    assert validated["engine_id"].unique().tolist() == [1]
    assert validated["cycle"].tolist() == [1, 2, 3, 4, 5]


def test_sample_engine_loader():
    path = ROOT / "data/raw/test_FD001.txt"

    if not path.is_file():
        pytest.skip("raw FD001 test data not available")

    df = load_sample_engine(3, "test", last_cycles=25)

    assert len(df) == 25
    assert df["engine_id"].unique().tolist() == [3]
    assert df["cycle"].is_monotonic_increasing

    validate_raw_frame(df)


def test_out_of_range_engine_rejected():
    path = ROOT / "data/raw/test_FD001.txt"

    if not path.is_file():
        pytest.skip("raw FD001 test data not available")

    with pytest.raises(InputValidationError):
        load_sample_engine(999, "test")
