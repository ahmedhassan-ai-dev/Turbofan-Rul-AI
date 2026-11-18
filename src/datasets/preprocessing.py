
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. PATHS & CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "raw" / "train_FD001.txt"
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
        raise ValueError(
            "Duplicate engine-cycle records detected"
        )


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

def preprocess_fd001() -> None:
    """Preprocess FD001 using a leakage-free train/validation split."""

    print("\n=== NASA C-MAPSS FD001 PREPROCESSING ===")

    # --------------------------------------------------------
    # STEP 1: Load dataset
    # --------------------------------------------------------

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Training dataset not found: {DATA_PATH}"
        )

    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"Split configuration not found: {SPLIT_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    validate_dataset(df)

    df["engine_id"] = df["engine_id"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    df = df.sort_values(
        ["engine_id", "cycle"]
    ).reset_index(drop=True)

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
    # STEP 5: Convert features to float64
    # --------------------------------------------------------

    # Prevent pandas FutureWarning when assigning scaled
    # floating-point values to original integer columns.

    feature_dtypes = {
        feature: "float64"
        for feature in selected_features
    }

    train_df = train_df.astype(feature_dtypes)
    val_df = val_df.astype(feature_dtypes)

    # --------------------------------------------------------
    # STEP 6: Standardization (NO DATA LEAKAGE)
    # --------------------------------------------------------

    scaler = StandardScaler()

    train_df[selected_features] = scaler.fit_transform(
        train_df[selected_features]
    )

    # Apply the fitted training scaler to validation.
    # Never call fit() or fit_transform() on validation.
    val_df[selected_features] = scaler.transform(
        val_df[selected_features]
    )

    # --------------------------------------------------------
    # STEP 7: Save processed datasets
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_columns = (
        ["engine_id", "cycle"]
        + selected_features
        + ["rul"]
    )

    train_output = OUTPUT_DIR / "train_FD001.csv"
    val_output = OUTPUT_DIR / "val_FD001.csv"

    train_df[output_columns].to_csv(
        train_output,
        index=False,
    )

    val_df[output_columns].to_csv(
        val_output,
        index=False,
    )

    # --------------------------------------------------------
    # STEP 8: Save fitted scaler
    # --------------------------------------------------------

    scaler_path = OUTPUT_DIR / "fd001_scaler.joblib"

    joblib.dump(scaler, scaler_path)

    # --------------------------------------------------------
    # STEP 9: Save preprocessing metadata
    # --------------------------------------------------------

    metadata = {
        "dataset": "FD001",
        "split_seed": split["seed"],
        "split_strategy": split["strategy"],
        "train_engines": len(train_ids),
        "validation_engines": len(val_ids),
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "candidate_features": CANDIDATE_FEATURES,
        "selected_features": selected_features,
        "constant_features": constant_features,
        "normalization": "StandardScaler",
        "target": "uncapped_rul",
        "scaler_file": scaler_path.name,
    }

    metadata_path = OUTPUT_DIR / "fd001_metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # STEP 10: Final sanity checks
    # --------------------------------------------------------

    assert set(train_df["engine_id"]).isdisjoint(
        set(val_df["engine_id"])
    )

    assert train_df["rul"].min() == 0
    assert val_df["rul"].min() == 0

    assert not train_df[output_columns].isna().any().any()
    assert not val_df[output_columns].isna().any().any()

    assert len(selected_features) == scaler.n_features_in_

    # --------------------------------------------------------
    # STEP 11: Summary
    # --------------------------------------------------------

    print("\n=== PREPROCESSING COMPLETED ===")

    print("Train engines:", train_df["engine_id"].nunique())
    print("Validation engines:", val_df["engine_id"].nunique())

    print("Train rows:", len(train_df))
    print("Validation rows:", len(val_df))

    print("\nConstant features:")
    print(constant_features)

    print("\nSelected features:", len(selected_features))
    print(selected_features)

    print("\nTraining feature mean (max absolute):")
    print(
        train_df[selected_features].mean().abs().max()
    )

    print("\nTraining feature standard deviation:")
    print(
        train_df[selected_features]
        .std(ddof=0)
        .round(4)
    )

    print("\nSaved files:")
    print("-", train_output)
    print("-", val_output)
    print("-", scaler_path)
    print("-", metadata_path)

    print("\nAll preprocessing sanity checks passed!")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    preprocess_fd001()
