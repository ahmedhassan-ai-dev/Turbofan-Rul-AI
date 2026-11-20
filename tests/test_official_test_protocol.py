"""Official FD001 test protocol tests."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets.windowing import create_last_windows


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

pytestmark = pytest.mark.skipif(
    not (PROCESSED / "fd001_metadata.json").is_file(),
    reason="processed FD001 artifacts not available",
)


@pytest.fixture(scope="module")
def protocol():
    from src.evaluation.test_evaluation import build_test_windows

    return build_test_windows(window_size=30)


@pytest.fixture(scope="module")
def test_frame():
    return pd.read_csv(PROCESSED / "test_FD001.csv")


@pytest.fixture(scope="module")
def features():
    metadata = json.loads(
        (PROCESSED / "fd001_metadata.json").read_text()
    )

    return metadata["selected_features"]


def test_one_window_per_test_engine(protocol):
    windows, labels, _ = protocol

    assert len(windows["X"]) == 100
    assert len(windows["X"]) == len(labels)
    assert windows["X"].shape == (100, 30, 17)


def test_engine_order_matches_label_file(protocol):
    windows, labels, _ = protocol

    assert windows["engines"].tolist() == labels["engine_id"].tolist()


def test_window_ends_at_last_recorded_cycle(protocol, test_frame):
    windows, _, _ = protocol

    last_cycles = (
        test_frame.groupby("engine_id")["cycle"].max()
        .reindex(windows["engines"]).to_numpy()
    )

    assert np.array_equal(windows["cycles"], last_cycles)


def test_last_timestep_is_always_real(protocol):
    windows, _, _ = protocol

    assert windows["mask"][:, -1].all()
    assert np.all(windows["mask"].sum(axis=1) >= 1)


def test_no_future_information_in_windows(protocol, test_frame, features):
    """A test window may only contain cycles up to the last recorded one."""

    windows, _, _ = protocol

    for index, engine_id in enumerate(windows["engines"]):
        engine = test_frame[
            test_frame["engine_id"] == engine_id
        ].sort_values("cycle")

        valid_length = int(windows["mask"][index].sum())
        expected_start = max(0, len(engine) - valid_length)

        expected = engine.iloc[
            expected_start:expected_start + valid_length
        ][features].to_numpy(dtype=np.float32)

        observed = windows["X"][index][-valid_length:]

        np.testing.assert_allclose(observed, expected, rtol=1e-6)


def test_targets_are_the_official_labels(protocol):
    _, labels, _ = protocol

    assert labels["rul"].min() >= 0
    assert labels["rul"].max() <= 200
    assert len(labels) == 100
