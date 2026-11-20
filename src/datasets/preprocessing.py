"""FD001 preprocessing: train / validation / official test preparation.

Protocol
--------
* Training and validation use complete run-to-failure trajectories.
* The split is engine-wise (GroupShuffleSplit, seed 42, 80/20) and is
  stored in ``configs/data/fd001_split.json``. It is never re-generated
  silently.
* Feature selection drops constant channels, using the training split
  only.
* StandardScaler is fitted on the training split only and reused for
  validation and test.
* RUL for training/validation rows is
  ``max_cycle(engine) - cycle`` (uncapped; capping is a training-time
  decision, see ``prepare_training_targets``).
* The official test trajectories are truncated, so their labels come
  from ``RUL_FD001.txt`` and must never be recomputed from
  ``max_cycle - cycle``.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PATHS & CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw"
TRAIN_PATH = DATA_DIR / "train_FD001.txt"
TEST_PATH = DATA_DIR / "test_FD001.txt"
TEST_RUL_PATH = DATA_DIR / "RUL_FD001.txt"

SPLIT_PATH = ROOT / "configs" / "data" / "fd001_split.json"

OUTPUT_DIR = ROOT / "data" / "processed"

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

CANDIDATE_FEATURES = (
    [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


# ============================================================
# 2. DATA VALIDATION
# ============================================================

def load_raw(path: Path) -> pd.DataFrame:
    """Read one whitespace-separated C-MAPSS trajectory file."""

    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    return df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the original NASA C-MAPSS dataset."""

    if list(df.columns) != COLUMNS:
        raise ValueError("Invalid dataset schema")

    if df.empty:
        raise ValueError("Dataset is empty")

    if df.isna().any().any():
        raise ValueError("Dataset contains missing values")

    if (df["engine_id"] <= 0).any():
        raise ValueError("Engine IDs must be positive")

    if (df["cycle"] <= 0).any():
        raise ValueError("Cycles must be positive")

    for column in ["engine_id", "cycle"]:
        if (df[column] % 1 != 0).any():
            raise ValueError(f"{column} must contain integers")

    if df.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine-cycle records detected")


def load_test_rul(path: Path = TEST_RUL_PATH) -> pd.DataFrame:
    """Load the official test RUL labels.

    ``RUL_FD001.txt`` contains one integer per test engine, in engine
    order (row 1 -> engine 1). The value is the remaining life at the
    LAST recorded cycle of that truncated test trajectory.
    """

    if not path.is_file():
        raise FileNotFoundError(f"Test RUL labels not found: {path}")

    values = [
        int(float(token))
        for token in path.read_text().split()
    ]

    if len(values) != 100:
        raise ValueError(
            f"Expected 100 test RUL labels, found {len(values)}"
        )

    if any(value < 0 for value in values):
        raise ValueError("Negative test RUL labels are invalid")

    return pd.DataFrame({
        "engine_id": list(range(1, len(values) + 1)),
        "rul": values,
    })


# ============================================================
# 3. RUL TARGET GENERATION
# ============================================================

def add_rul_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate uncapped RUL for complete training trajectories.

    RUL = maximum observed cycle - current cycle.

    IMPORTANT:
    This function must not be used to generate ground-truth
    targets for official truncated test trajectories.
    """

    result = df.copy()

    max_cycles = result.groupby("engine_id")[
        "cycle"
    ].transform("max")

    result["rul"] = (max_cycles - result["cycle"]).astype(int)

    return result


# ============================================================
# 4. MAIN PREPROCESSING PIPELINE
# ============================================================

def preprocess_fd001() -> dict:
    """Preprocess FD001 using a leakage-free train/validation split."""

    print("\n=== NASA C-MAPSS FD001 PREPROCESSING ===")

    # --------------------------------------------------------
    # STEP 1: Load dataset
    # --------------------------------------------------------

    df = load_raw(TRAIN_PATH)

    validate_dataset(df)

    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"Split configuration not found: {SPLIT_PATH}"
        )

    # --------------------------------------------------------
    # STEP 2: Load predefined engine-wise split
    # --------------------------------------------------------

    split = json.loads(
        SPLIT_PATH.read_text(encoding="utf-8")
    )

    train_ids = set(split["train_engine_ids"])
    val_ids = set(split["val_engine_ids"])

    if train_ids & val_ids:
        raise ValueError(
            "Data leakage: train/validation engine overlap"
        )

    all_engine_ids = set(df["engine_id"].unique())

    if train_ids | val_ids != all_engine_ids:
        raise ValueError(
            "Train/validation split does not cover all engines"
        )

    train_df = df[
        df["engine_id"].isin(train_ids)
    ].copy()

    val_df = df[
        df["engine_id"].isin(val_ids)
    ].copy()

    # --------------------------------------------------------
    # STEP 3: Feature selection (training data only)
    # --------------------------------------------------------

    unique_counts = train_df[CANDIDATE_FEATURES].nunique()

    constant_features = unique_counts[
        unique_counts <= 1
    ].index.tolist()

    selected_features = [
        feature
        for feature in CANDIDATE_FEATURES
        if feature not in constant_features
    ]

    if not selected_features:
        raise ValueError("No usable features were found")

    # --------------------------------------------------------
    # STEP 4: Generate RUL targets
    # --------------------------------------------------------

    train_df = add_rul_targets(train_df)
    val_df = add_rul_targets(val_df)

    # --------------------------------------------------------
    # STEP 5: Load and validate the official test set
    # --------------------------------------------------------

    test_df = load_raw(TEST_PATH)
    test_rul = load_test_rul()

    validate_dataset(test_df)

    test_engine_ids = sorted(
        int(x) for x in test_df["engine_id"].unique()
    )

    if test_engine_ids != test_rul["engine_id"].tolist():
        raise ValueError(
            "Test engine IDs do not match RUL_FD001.txt ordering"
        )

    if test_df.groupby("engine_id")["cycle"].min().ne(1).any():
        raise ValueError("Test trajectories must start at cycle 1")

    # --------------------------------------------------------
    # STEP 6: Convert features to float64
    # --------------------------------------------------------

    # Prevent pandas FutureWarning when assigning scaled
    # floating-point values to original integer columns.

    feature_dtypes = {
        feature: "float64"
        for feature in selected_features
    }

    train_df = train_df.astype(feature_dtypes)
    val_df = val_df.astype(feature_dtypes)
    test_df = test_df.astype(feature_dtypes)

    # --------------------------------------------------------
    # STEP 7: Standardization (NO DATA LEAKAGE)
    # --------------------------------------------------------

    scaler = StandardScaler()

    train_df[selected_features] = scaler.fit_transform(
        train_df[selected_features]
    )

    # Apply the fitted training scaler to validation / test.
    # Never call fit() or fit_transform() on them.
    val_df[selected_features] = scaler.transform(
        val_df[selected_features]
    )

    test_df[selected_features] = scaler.transform(
        test_df[selected_features]
    )

    # --------------------------------------------------------
    # STEP 8: Save processed datasets
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = (
        ["engine_id", "cycle"]
        + selected_features
        + ["rul"]
    )

    train_output = OUTPUT_DIR / "train_FD001.csv"
    val_output = OUTPUT_DIR / "val_FD001.csv"
    test_output = OUTPUT_DIR / "test_FD001.csv"
    test_rul_output = OUTPUT_DIR / "fd001_test_rul.csv"

    train_df[output_columns].to_csv(train_output, index=False)
    val_df[output_columns].to_csv(val_output, index=False)
    test_df[["engine_id", "cycle"] + selected_features].to_csv(
        test_output,
        index=False,
    )
    test_rul.to_csv(test_rul_output, index=False)

    # --------------------------------------------------------
    # STEP 9: Save fitted scaler
    # --------------------------------------------------------

    scaler_path = OUTPUT_DIR / "fd001_scaler.joblib"

    joblib.dump(scaler, scaler_path)

    # --------------------------------------------------------
    # STEP 10: Save preprocessing metadata
    # --------------------------------------------------------

    metadata = {
        "dataset": "FD001",
        "split_seed": split["seed"],
        "split_strategy": split["strategy"],
        "train_engines": len(train_ids),
        "validation_engines": len(val_ids),
        "test_engines": len(test_engine_ids),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "test_rows": len(test_df),
        "candidate_features": CANDIDATE_FEATURES,
        "selected_features": selected_features,
        "constant_features": constant_features,
        "normalization": "StandardScaler",
        "scaler_fit_on": "train_split_only",
        "target": "uncapped_rul",
        "test_target": "RUL_FD001.txt",
        "scaler_file": scaler_path.name,
        "raw_files": {
            "train": TRAIN_PATH.name,
            "test": TEST_PATH.name,
            "test_rul": TEST_RUL_PATH.name,
        },
    }

    metadata_path = OUTPUT_DIR / "fd001_metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # STEP 11: Final sanity checks
    # --------------------------------------------------------

    assert set(train_df["engine_id"]).isdisjoint(
        set(val_df["engine_id"])
    )

    # Engine IDs restart at 1 in the official test file, so the same
    # identifier can appear in train and test. They are different
    # physical units; the guarantee we need is that no test row was used
    # to fit the scaler or to train a model, which is enforced above by
    # transforming test with the training scaler only.
    assert "rul" not in test_df.columns

    assert train_df["rul"].min() == 0
    assert val_df["rul"].min() == 0

    assert not train_df[output_columns].isna().any().any()
    assert not val_df[output_columns].isna().any().any()

    assert len(selected_features) == scaler.n_features_in_

    # The training split must be standardized after fitting.
    scaled_values = (
        train_df[selected_features].to_numpy(dtype=float)
    )

    assert np.allclose(
        scaled_values.mean(axis=0), 0.0, atol=1e-9
    )
    # StandardScaler standardizes with the population standard
    # deviation (ddof=0).
    assert np.allclose(
        scaled_values.std(axis=0, ddof=0), 1.0, atol=1e-6
    )
    assert np.isfinite(scaler.scale_).all()
    assert (scaler.scale_ > 0).all()

    # --------------------------------------------------------
    # STEP 12: Summary
    # --------------------------------------------------------

    print("\n=== PREPROCESSING COMPLETED ===")

    print("Train engines:", train_df["engine_id"].nunique())
    print("Validation engines:", val_df["engine_id"].nunique())
    print("Test engines:", test_df["engine_id"].nunique())

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))
    print("Test rows:", len(test_df))

    print("\nConstant features:")
    print(constant_features)

    print("\nSelected features:", len(selected_features))
    print(selected_features)

    print("\nTraining feature mean (max absolute):")
    print(train_df[selected_features].mean().abs().max())

    print("\nTraining feature standard deviation:")
    print(
        train_df[selected_features]
        .std(ddof=0)
        .round(4)
    )

    print("\nOfficial test RUL label statistics:")
    print(test_rul["rul"].describe().round(2).to_string())

    print("\nSaved files:")
    for path in [
        train_output,
        val_output,
        test_output,
        test_rul_output,
        scaler_path,
        metadata_path,
    ]:
        print("-", path)

    print("\nAll preprocessing sanity checks passed!")

    return metadata


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    preprocess_fd001()
