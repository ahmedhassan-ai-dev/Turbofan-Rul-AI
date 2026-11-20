"""Pure, testable helpers for the Streamlit prediction display."""

import hashlib

import pandas as pd


SAMPLE_SOURCE = "Sample engine from the dataset"


def input_signature(frame: pd.DataFrame, model_name: str, source: str) -> str:
    """Identify the exact observed input and model that produced a result."""
    if frame is None or frame.empty:
        raise ValueError("Cannot fingerprint an empty input.")

    digest = hashlib.sha256()
    for value in (source, model_name):
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")
    digest.update(frame.to_csv(index=False, float_format="%.17g").encode("utf-8"))
    return digest.hexdigest()


def is_official_reference_allowed(
    source: str, input_frame: pd.DataFrame, predictions: pd.DataFrame
) -> bool:
    """Only the actual last cycle of a bundled official test engine has a known label."""
    if (
        source != SAMPLE_SOURCE
        or input_frame is None
        or predictions is None
        or input_frame.empty
        or len(predictions) != 1
        or "engine_id" not in input_frame
        or "cycle" not in input_frame
        or "engine_id" not in predictions
        or "last_cycle" not in predictions
        or input_frame["engine_id"].nunique() != 1
    ):
        return False

    return (
        int(input_frame["engine_id"].iloc[0])
        == int(predictions["engine_id"].iloc[0])
        and int(input_frame["cycle"].max())
        == int(predictions["last_cycle"].iloc[0])
    )
