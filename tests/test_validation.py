
import numpy as np
import pandas as pd
import pytest

from src.datasets.validation import (
    create_validation_endpoints,
)
from src.datasets.windowing import create_windows


FEATURES = ["sensor_2", "sensor_3"]


@pytest.fixture
def sample_data():
    """Two synthetic run-to-failure engine trajectories."""

    records = []

    for engine_id, lifetime in [(1, 12), (2, 15)]:
        for cycle in range(1, lifetime + 1):
            records.append({
                "engine_id": engine_id,
                "cycle": cycle,
                "sensor_2": float(engine_id * 100 + cycle),
                "sensor_3": float(cycle * 2),
                "rul": lifetime - cycle,
            })

    return pd.DataFrame(records)


def test_endpoint_count(sample_data):
    endpoints = create_validation_endpoints(
        sample_data,
        horizons=(8, 5, 2),
    )

    assert len(endpoints) == 6
    assert endpoints["engine_id"].nunique() == 2


def test_correct_rul_labels(sample_data):
    endpoints = create_validation_endpoints(
        sample_data,
        horizons=(8, 5, 2),
    )

    assert sorted(endpoints["rul"].tolist()) == [
        2, 2, 5, 5, 8, 8
    ]


def test_correct_cutoff_cycles(sample_data):
    endpoints = create_validation_endpoints(
        sample_data,
        horizons=(8, 5, 2),
    )

    engine_1 = endpoints[
        endpoints["engine_id"] == 1
    ]

    assert sorted(engine_1["cycle"].tolist()) == [
        4, 7, 10
    ]


def test_no_future_sensor_access(sample_data):
    endpoints = create_validation_endpoints(
        sample_data,
        horizons=(5,),
    )

    original = create_windows(
        sample_data,
        FEATURES,
        window_size=4,
    )

    modified = sample_data.copy()

    # Change sensor readings occurring AFTER the
    # selected validation endpoint for Engine 1.
    cutoff = int(
        endpoints.loc[
            endpoints["engine_id"] == 1,
            "cycle",
        ].iloc[0]
    )

    modified.loc[
        (modified["engine_id"] == 1)
        & (modified["cycle"] > cutoff),
        "sensor_2",
    ] = 9999.0

    updated = create_windows(
        modified,
        FEATURES,
        window_size=4,
    )

    idx = np.where(
        (original["engines"] == 1)
        & (original["cycles"] == cutoff)
    )[0][0]

    np.testing.assert_array_equal(
        original["X"][idx],
        updated["X"][idx],
    )


def test_short_engine_rejected(sample_data):
    with pytest.raises(ValueError):
        create_validation_endpoints(
            sample_data,
            horizons=(20,),
        )


def test_target_mismatch_rejected(sample_data):
    modified = sample_data.copy()

    modified.loc[
        (modified["engine_id"] == 1)
        & (modified["cycle"] == 7),
        "rul",
    ] = 999

    with pytest.raises(ValueError):
        create_validation_endpoints(
            modified,
            horizons=(5,),
        )
