
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data/raw/train_FD001.txt"
SPLIT_PATH = ROOT / "configs/data/fd001_split.json"

SEED = 42
VALIDATION_SIZE = 0.20

COLUMNS = (
    ["engine_id", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)


def create_split():
    df = pd.read_csv(
        DATA_PATH,
        sep=r"\s+",
        header=None,
        names=COLUMNS,
    )

    if df.shape[1] != len(COLUMNS):
        raise ValueError("Invalid C-MAPSS column count")

    if df.isna().any().any():
        raise ValueError("Missing values detected")

    if df.duplicated(["engine_id", "cycle"]).any():
        raise ValueError("Duplicate engine-cycle records detected")

    engine_ids = sorted(df["engine_id"].unique().tolist())

    if len(engine_ids) != 100:
        raise ValueError("Expected 100 FD001 training engines")

    if SPLIT_PATH.exists():
        raise FileExistsError(
            f"Split already exists: {SPLIT_PATH}. "
            "Do not overwrite an established experiment split."
        )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
    )

    train_idx, val_idx = next(
        splitter.split(df, groups=df["engine_id"])
    )

    train_ids = sorted(
        int(x) for x in df.iloc[train_idx]["engine_id"].unique()
    )

    val_ids = sorted(
        int(x) for x in df.iloc[val_idx]["engine_id"].unique()
    )

    assert len(train_ids) == 80
    assert len(val_ids) == 20
    assert not set(train_ids).intersection(val_ids)
    assert set(train_ids) | set(val_ids) == set(engine_ids)

    split_info = {
        "dataset": "FD001",
        "seed": SEED,
        "strategy": "engine_wise_group_shuffle",
        "train_engine_ids": train_ids,
        "val_engine_ids": val_ids,
    }

    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    SPLIT_PATH.write_text(
        json.dumps(split_info, indent=2),
        encoding="utf-8",
    )

    print("=== FD001 SPLIT CREATED ===")
    print("Train engines:", len(train_ids))
    print("Validation engines:", len(val_ids))
    print("Train rows:", len(train_idx))
    print("Validation rows:", len(val_idx))
    print("Engine overlap:", len(set(train_ids) & set(val_ids)))
    print("Saved to:", SPLIT_PATH)


if __name__ == "__main__":
    create_split()
