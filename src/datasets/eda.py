
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "raw" / "train_FD001.txt"

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Raw FD001 training data not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    print("\n=== ENGINE LIFETIME STATISTICS ===")

    lifetimes = df.groupby("engine_id")["cycle"].max()

    print(lifetimes.describe().round(2))

    features = [
        col for col in df.columns
        if col.startswith(("setting_", "sensor_"))
    ]

    statistics = pd.DataFrame({
        "unique_values": df[features].nunique(),
        "std": df[features].std(),
        "min": df[features].min(),
        "max": df[features].max(),
    })

    print("\n=== SENSOR STATISTICS ===")
    print(statistics.round(5).to_string())

    constant_features = statistics[
        statistics["unique_values"] == 1
    ].index.tolist()

    print("\n=== CONSTANT FEATURES ===")
    print(constant_features)

    print("\n=== DATA QUALITY ===")
    print("Missing values:", df.isna().sum().sum())
    print(
        "Duplicate engine-cycle records:",
        df.duplicated(["engine_id", "cycle"]).sum(),
    )


if __name__ == "__main__":
    main()
