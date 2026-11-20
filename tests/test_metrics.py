import math

import numpy as np
import pytest

from src.evaluation.metrics import (
    nasa_score,
    nasa_scores,
    regression_metrics,
)


def test_basic_errors():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])

    metrics = regression_metrics(y_true, y_pred)

    # errors: +2, -2, 0  -> mse = (4 + 4 + 0) / 3
    assert metrics["rmse"] == pytest.approx(math.sqrt(8 / 3))
    assert metrics["mae"] == pytest.approx(4 / 3)
    assert metrics["bias"] == pytest.approx(0.0)
    assert metrics["n_samples"] == 3


def test_r2_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])

    metrics = regression_metrics(y_true, y_pred)

    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(0.0)


def test_r2_none_for_constant_targets():
    y_true = np.array([5.0, 5.0, 5.0])
    y_pred = np.array([5.0, 4.0, 6.0])

    assert regression_metrics(y_true, y_pred)["r2"] is None


def test_nasa_early_error_hand_computed():
    # d = -10 -> exp(10/13) - 1
    expected = math.exp(10 / 13) - 1

    score = nasa_scores([100.0], [90.0])[0]

    assert score == pytest.approx(expected, rel=1e-12)


def test_nasa_late_error_hand_computed():
    # d = +10 -> exp(10/10) - 1
    expected = math.exp(1.0) - 1

    score = nasa_scores([90.0], [100.0])[0]

    assert score == pytest.approx(expected, rel=1e-12)


def test_nasa_penalises_late_more_than_early():
    early = nasa_scores([100.0], [90.0])[0]
    late = nasa_scores([100.0], [110.0])[0]

    assert late > early


def test_nasa_zero_for_perfect_prediction():
    assert nasa_score([50.0, 60.0], [50.0, 60.0]) == pytest.approx(0.0)


def test_nasa_score_is_sum_with_mean_variant():
    y_true = np.array([100.0, 90.0])
    y_pred = np.array([90.0, 100.0])

    metrics = regression_metrics(y_true, y_pred)

    expected_first = math.exp(10 / 13) - 1
    expected_second = math.exp(1.0) - 1

    assert metrics["nasa_score"] == pytest.approx(
        expected_first + expected_second
    )
    assert metrics["nasa_score_mean"] == pytest.approx(
        (expected_first + expected_second) / 2
    )


def test_invalid_inputs_rejected():
    with pytest.raises(ValueError):
        regression_metrics([1.0, 2.0], [1.0])

    with pytest.raises(ValueError):
        regression_metrics([], [])

    with pytest.raises(ValueError):
        regression_metrics([1.0, np.nan], [1.0, 2.0])


def test_metric_values_documented_in_report():
    """Guard the exact numbers quoted in the report."""

    metrics = regression_metrics([100.0], [90.0])

    assert metrics["rmse"] == pytest.approx(10.0)
    assert metrics["mae"] == pytest.approx(10.0)
    assert metrics["bias"] == pytest.approx(-10.0)
    assert metrics["nasa_score"] == pytest.approx(
        math.exp(10 / 13) - 1
    )
