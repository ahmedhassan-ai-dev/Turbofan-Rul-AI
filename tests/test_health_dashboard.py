"""Causal time-series and cross-seed comparison regression tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets.windowing import create_last_windows
from src.inference.health_dashboard import (
    build_causal_timeline_windows,
    main_seed_checkpoints,
    predict_rul_timeline,
    summarize_seed_disagreement,
)
from src.inference.pipeline import InputValidationError


class LastSensorPredictor:
    window_size = 3

    def predict_windows(self, x, mask):
        assert np.all(mask)
        return x[:, -1, 0] * 2


def sample_scaled(n=8, first_cycle=40):
    return pd.DataFrame({
        "engine_id": [5] * n,
        "cycle": np.arange(first_cycle, first_cycle + n),
        "sensor_2": np.arange(n, dtype=float),
        "sensor_3": np.arange(n, dtype=float) * -2,
    })


def test_causal_windows_only_include_current_and_past_rows():
    result = build_causal_timeline_windows(
        sample_scaled(), ["sensor_2", "sensor_3"], window_size=3,
    )
    assert result["cycles"].tolist() == [42, 43, 44, 45, 46, 47]
    np.testing.assert_allclose(result["X"][0, :, 0], [0, 1, 2])
    np.testing.assert_allclose(result["X"][1, :, 0], [1, 2, 3])
    assert result["mask"].all()


def test_final_timeline_window_matches_official_last_window_protocol():
    frame = sample_scaled()
    timeline = build_causal_timeline_windows(frame, ["sensor_2"], 3)
    official = create_last_windows(frame, features=["sensor_2"], window_size=3)
    np.testing.assert_array_equal(timeline["X"][-1], official["X"][0])
    np.testing.assert_array_equal(timeline["mask"][-1], official["mask"][0])
    assert timeline["cycles"][-1] == official["cycles"][0]


def test_trailing_observation_and_max_points_obey_absolute_cycle_numbers():
    result = build_causal_timeline_windows(
        sample_scaled(), ["sensor_2"], window_size=3, max_points=2,
    )
    assert result["cycles"].tolist() == [46, 47]
    assert result["X"].shape == (2, 3, 1)


def test_timeline_predictions_match_last_cycle_prediction():
    timeline = predict_rul_timeline(
        sample_scaled(), ["sensor_2"], LastSensorPredictor(),
    )
    assert timeline["cycle"].iloc[-1] == 47
    assert timeline["predicted_rul"].tolist() == [4, 6, 8, 10, 12, 14]


def test_timeline_requires_full_observed_window():
    with pytest.raises(InputValidationError, match="at least 3"):
        build_causal_timeline_windows(sample_scaled(n=2), ["sensor_2"], 3)


def test_timeline_rejects_multiple_engines_and_gaps():
    mixed = sample_scaled()
    mixed.loc[mixed.index[-1], "engine_id"] = 7
    with pytest.raises(InputValidationError, match="one engine"):
        build_causal_timeline_windows(mixed, ["sensor_2"], 3)

    gap = sample_scaled()
    gap.loc[gap.index[-1], "cycle"] += 1
    with pytest.raises(InputValidationError, match="consecutive"):
        build_causal_timeline_windows(gap, ["sensor_2"], 3)


def test_disagreement_summary_is_descriptive_not_calibrated():
    curves = {
        42: pd.DataFrame({"cycle": [9, 10], "predicted_rul": [80.0, 76.0]}),
        123: pd.DataFrame({"cycle": [9, 10], "predicted_rul": [90.0, 73.0]}),
        3407: pd.DataFrame({"cycle": [9, 10], "predicted_rul": [85.0, 79.0]}),
    }
    result = summarize_seed_disagreement(curves)
    assert result["mean_rul"].tolist() == [85.0, 76.0]
    assert result["min_rul"].tolist() == [80.0, 73.0]
    assert result["max_rul"].tolist() == [90.0, 79.0]
    assert result["spread_rul"].tolist() == [10.0, 6.0]


def test_disagreement_rejects_nonmatching_cycles():
    a = pd.DataFrame({"cycle": [5, 6], "predicted_rul": [10.0, 9.0]})
    b = pd.DataFrame({"cycle": [5, 7], "predicted_rul": [10.0, 9.0]})
    with pytest.raises(ValueError, match="exact same cycles"):
        summarize_seed_disagreement({42: a, 123: b})


def test_seed_registry_filters_smoke_and_missing_checkpoints(tmp_path: Path):
    good1 = tmp_path / "main_42.pt"
    good2 = tmp_path / "main_123.pt"
    good1.write_bytes(b"dummy")
    good2.write_bytes(b"dummy")
    rows = pd.DataFrame([
        ("main", "lstm", 42, 125, 30, "main_42.pt", "completed"),
        ("main", "lstm", 123, 125, 30, "main_123.pt", "completed"),
        ("main", "lstm", 3407, 125, 30, "missing.pt", "completed"),
        ("smoke", "lstm", 88, 125, 30, "main_42.pt", "completed"),
        ("main", "transformer", 42, 125, 30, "main_42.pt", "completed"),
    ], columns=["tag", "model", "seed", "rul_cap", "window_size", "checkpoint", "status"])
    paths = main_seed_checkpoints(rows, "lstm", tmp_path)
    assert sorted(paths) == [42, 123]
    assert main_seed_checkpoints(rows, "random_forest", tmp_path) == {}


def test_seed_registry_rejects_mixed_training_protocols(tmp_path: Path):
    path = tmp_path / "model.pt"
    path.write_bytes(b"dummy")
    registry = pd.DataFrame([
        ("main", "lstm", 42, 125, 30, "model.pt", "completed"),
        ("main", "lstm", 123, 100, 30, "model.pt", "completed"),
    ], columns=["tag", "model", "seed", "rul_cap", "window_size", "checkpoint", "status"])
    assert main_seed_checkpoints(registry, "lstm", tmp_path) == {}
