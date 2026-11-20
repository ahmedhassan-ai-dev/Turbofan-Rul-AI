"""Leakage and protocol tests on the real FD001 artifacts.

These tests require `python -m src.datasets.preprocessing` to have been
run. They are skipped (not silently passed) when the artifacts are
missing.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
METADATA_PATH = PROCESSED / "fd001_metadata.json"


pytestmark = pytest.mark.skipif(
    not METADATA_PATH.is_file(),
    reason="processed FD001 artifacts not available",
)


@pytest.fixture(scope="module")
def metadata():
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def frames():
    return {
        "train": pd.read_csv(PROCESSED / "train_FD001.csv"),
        "val": pd.read_csv(PROCESSED / "val_FD001.csv"),
        "test": pd.read_csv(PROCESSED / "test_FD001.csv"),
    }


def test_feature_count_is_17(metadata):
    assert len(metadata["selected_features"]) == 17
    assert len(metadata["constant_features"]) == 7


def test_constant_features_are_not_selected(metadata):
    assert not set(metadata["constant_features"]) & set(
        metadata["selected_features"]
    )


def test_split_is_engine_wise_and_disjoint(metadata, frames):
    train_ids = set(frames["train"]["engine_id"])
    val_ids = set(frames["val"]["engine_id"])

    assert len(train_ids) == 80
    assert len(val_ids) == 20
    assert train_ids.isdisjoint(val_ids)

    split_config = json.loads(
        (ROOT / "configs/data/fd001_split.json").read_text()
    )

    assert train_ids == set(split_config["train_engine_ids"])
    assert val_ids == set(split_config["val_engine_ids"])
    assert train_ids.isdisjoint(set(split_config["val_engine_ids"]))


def test_scaler_fitted_on_train_only(metadata, frames):
    """The scaler must make train mean 0 / std 1 but not val or test."""

    features = metadata["selected_features"]

    train = frames["train"][features].to_numpy(dtype=float)
    val = frames["val"][features].to_numpy(dtype=float)
    test = frames["test"][features].to_numpy(dtype=float)

    assert np.allclose(train.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(train.std(axis=0, ddof=0), 1.0, atol=1e-6)

    # If the scaler had been refitted on validation, its mean would also
    # be exactly zero.
    assert np.abs(val.mean(axis=0)).max() > 1e-6
    assert np.abs(test.mean(axis=0)).max() > 1e-6


def test_scaler_artifact_matches_metadata(metadata):
    scaler = joblib.load(PROCESSED / metadata["scaler_file"])

    assert scaler.n_features_in_ == len(metadata["selected_features"])
    assert scaler.mean_.shape == (len(metadata["selected_features"]),)


def test_rul_labels_recomputed_correctly(frames):
    for split in ["train", "val"]:
        df = frames[split]
        recomputed = (
            df.groupby("engine_id")["cycle"].transform("max")
            - df["cycle"]
        )

        assert (df["rul"].to_numpy() == recomputed.to_numpy()).all()
        assert df["rul"].min() == 0


def test_official_test_labels_match_file(frames):
    labels = pd.read_csv(PROCESSED / "fd001_test_rul.csv")

    raw_values = [
        int(float(token))
        for token in (ROOT / "data/raw/RUL_FD001.txt").read_text().split()
    ]

    assert len(labels) == 100
    assert labels["engine_id"].tolist() == list(range(1, 101))
    assert labels["rul"].tolist() == raw_values


def test_test_frame_has_no_target_column(frames):
    assert "rul" not in frames["test"].columns


def test_no_missing_or_duplicate_rows(frames, metadata):
    features = metadata["selected_features"]

    for name, df in frames.items():
        assert not df[["engine_id", "cycle"] + features].isna().any().any()
        assert not df.duplicated(["engine_id", "cycle"]).any()

        assert np.isfinite(
            df[features].to_numpy(dtype=float)
        ).all()


def test_raw_row_counts(metadata):
    raw_train = pd.read_csv(
        ROOT / "data/raw/train_FD001.txt",
        sep=r"\s+", header=None,
    )
    raw_test = pd.read_csv(
        ROOT / "data/raw/test_FD001.txt",
        sep=r"\s+", header=None,
    )

    assert metadata["train_rows"] + metadata["validation_rows"] == len(
        raw_train
    )
    assert metadata["test_rows"] == len(raw_test)
