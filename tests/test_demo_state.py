"""Regression tests for stale UI state and label provenance."""

import pandas as pd

from src.inference.demo_state import (
    SAMPLE_SOURCE,
    input_signature,
    is_official_reference_allowed,
)


def sample_input(engine=1, sensor=5.0, last_cycle=80):
    return pd.DataFrame({
        "engine_id": [engine, engine],
        "cycle": [last_cycle - 1, last_cycle],
        "sensor_2": [sensor, sensor + 0.1],
    })


def prediction(engine=1, cycle=80):
    return pd.DataFrame({
        "engine_id": [engine],
        "last_cycle": [cycle],
        "predicted_rul": [45.0],
    })


def test_result_invalidated_if_engine_changes():
    one = input_signature(sample_input(1), "transformer", SAMPLE_SOURCE)
    two = input_signature(sample_input(79), "transformer", SAMPLE_SOURCE)
    assert one != two


def test_result_invalidated_if_sensor_changes():
    one = input_signature(sample_input(sensor=5.0), "transformer", SAMPLE_SOURCE)
    two = input_signature(sample_input(sensor=5.001), "transformer", SAMPLE_SOURCE)
    assert one != two


def test_result_invalidated_if_model_or_source_changes():
    df = sample_input()
    original = input_signature(df, "lstm", SAMPLE_SOURCE)
    assert original != input_signature(df, "transformer", SAMPLE_SOURCE)
    assert original != input_signature(df, "lstm", "Upload a sensor CSV")


def test_official_label_only_for_sample_with_matching_endpoint():
    df = sample_input()
    assert is_official_reference_allowed(SAMPLE_SOURCE, df, prediction())
    assert not is_official_reference_allowed("Upload a sensor CSV", df, prediction())
    assert not is_official_reference_allowed(SAMPLE_SOURCE, df, prediction(engine=2))
    assert not is_official_reference_allowed(SAMPLE_SOURCE, df, prediction(cycle=79))
