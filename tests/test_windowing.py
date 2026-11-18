
import numpy as np
import pandas as pd
import pytest

from src.datasets.windowing import create_windows


FEATURES = ["sensor_2", "sensor_3"]


@pytest.fixture
def sample_data():
    """Small synthetic dataset with two engines."""

    return pd.DataFrame({
        "engine_id": [1, 1, 1, 2, 2],
        "cycle": [1, 2, 3, 1, 2],
        "sensor_2": [10, 20, 30, 100, 200],
        "sensor_3": [1, 2, 3, 10, 20],
        "rul": [2, 1, 0, 1, 0],
    })


def test_output_shapes(sample_data):
    result = create_windows(
        sample_data, FEATURES, window_size=3
    )

    assert result["X"].shape == (5, 3, 2)
    assert result["y"].shape == (5,)
    assert result["mask"].shape == (5, 3)


def test_no_engine_mixing(sample_data):
    result = create_windows(
        sample_data, FEATURES, window_size=3
    )

    # First three windows belong to Engine 1.
    assert np.array_equal(
        result["engines"][:3], [1, 1, 1]
    )

    # Remaining windows belong to Engine 2.
    assert np.array_equal(
        result["engines"][3:], [2, 2]
    )

    # Engine 2's first window must contain
    # only Engine 2's earliest sensor reading.
    assert np.array_equal(
        result["X"][3, :, 0], [100, 100, 100]
    )


def test_no_future_leakage(sample_data):
    result = create_windows(
        sample_data, FEATURES, window_size=3
    )

    # At Cycle 2, only Cycles 1 and 2 are available.
    assert np.array_equal(
        result["X"][1, :, 0], [10, 10, 20]
    )

    assert result["cycles"][1] == 2
    assert result["y"][1] == 1

    # Changing a future reading must not affect
    # the window used to predict Cycle 2.
    modified = sample_data.copy()
    modified.loc[
        (modified["engine_id"] == 1)
        & (modified["cycle"] == 3),
        "sensor_2"
    ] = 9999

    updated = create_windows(
        modified, FEATURES, window_size=3
    )

    np.testing.assert_array_equal(
        result["X"][1],
        updated["X"][1],
    )


def test_padding_mask(sample_data):
    result = create_windows(
        sample_data, FEATURES, window_size=3
    )

    assert np.array_equal(
        result["mask"][0], [False, False, True]
    )

    assert np.array_equal(
        result["mask"][1], [False, True, True]
    )

    assert np.array_equal(
        result["mask"][2], [True, True, True]
    )


def test_target_alignment(sample_data):
    result = create_windows(
        sample_data, FEATURES, window_size=3
    )

    assert np.array_equal(
        result["y"], [2, 1, 0, 1, 0]
    )


def test_invalid_window_size(sample_data):
    with pytest.raises(ValueError):
        create_windows(
            sample_data, FEATURES, window_size=0
        )


def test_missing_feature(sample_data):
    with pytest.raises(ValueError):
        create_windows(
            sample_data, ["unknown_sensor"], window_size=3
        )


def test_invalid_cycle_sequence(sample_data):
    modified = sample_data.copy()

    modified.loc[
        (modified["engine_id"] == 1)
        & (modified["cycle"] == 2),
        "cycle"
    ] = 5

    with pytest.raises(ValueError):
        create_windows(
            modified, FEATURES, window_size=3
        )
