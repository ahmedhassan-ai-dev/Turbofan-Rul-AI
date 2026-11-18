
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from src.datasets.windowing import create_windows


ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = ROOT / "data" / "processed"

METADATA_PATH = (
    PROCESSED_DIR / "fd001_metadata.json"
)

ENDPOINTS_PATH = (
    ROOT / "configs/data/fd001_validation_endpoints.csv"
)


class CMapssDataset(Dataset):
    """
    PyTorch dataset for C-MAPSS RUL regression.

    Modes:
        train: all training windows
        val: all validation windows
        val_endpoints: predefined validation cutoffs
    """

    def __init__(
        self,
        split: str,
        window_size: int = 30,
    ):
        if split not in {
            "train", "val", "val_endpoints"
        }:
            raise ValueError(f"Unsupported split: {split}")

        if window_size < 1:
            raise ValueError("window_size must be positive")

        self.split = split
        self.window_size = window_size

        metadata = json.loads(
            METADATA_PATH.read_text(encoding="utf-8")
        )

        self.features = metadata["selected_features"]

        csv_name = (
            "train_FD001.csv"
            if split == "train"
            else "val_FD001.csv"
        )

        df = pd.read_csv(PROCESSED_DIR / csv_name)

        arrays = create_windows(
            df=df,
            features=self.features,
            window_size=window_size,
        )

        if split == "val_endpoints":
            endpoints = pd.read_csv(ENDPOINTS_PATH)

            keys = list(zip(
                arrays["engines"].tolist(),
                arrays["cycles"].tolist(),
            ))

            lookup = {
                (int(engine), int(cycle)): index
                for index, (engine, cycle)
                in enumerate(keys)
            }

            indices = []

            for row in endpoints.itertuples(index=False):
                key = (int(row.engine_id), int(row.cycle))

                if key not in lookup:
                    raise ValueError(
                        f"Validation endpoint not found: {key}"
                    )

                idx = lookup[key]

                if not np.isclose(
                    arrays["y"][idx],
                    float(row.rul),
                ):
                    raise ValueError(
                        f"Endpoint target mismatch: {key}"
                    )

                indices.append(idx)

            if len(indices) != len(set(indices)):
                raise ValueError(
                    "Duplicate validation endpoints"
                )

            arrays = {
                key: values[indices]
                for key, values in arrays.items()
            }

        self.x = torch.from_numpy(
            arrays["X"].copy()
        ).float()

        self.y = torch.from_numpy(
            arrays["y"].copy()
        ).float().unsqueeze(-1)

        self.mask = torch.from_numpy(
            arrays["mask"].copy()
        ).bool()

        self.engine_ids = torch.from_numpy(
            arrays["engines"].copy()
        ).long()

        self.cycles = torch.from_numpy(
            arrays["cycles"].copy()
        ).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return {
            "x": self.x[index],
            "y": self.y[index],
            "mask": self.mask[index],
            "engine_id": self.engine_ids[index],
            "cycle": self.cycles[index],
        }


def create_dataloader(
    split: str,
    batch_size: int = 64,
    window_size: int = 30,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Create a reproducible PyTorch DataLoader."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    dataset = CMapssDataset(
        split=split,
        window_size=window_size,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator,
    )


def main():
    for split in ["train", "val", "val_endpoints"]:
        loader = create_dataloader(
            split=split,
            batch_size=64,
            window_size=30,
        )

        batch = next(iter(loader))

        print(f"\n=== {split.upper()} ===")

        print("Samples:", len(loader.dataset))
        print("Batches:", len(loader))
        print("X:", tuple(batch["x"].shape))
        print("Y:", tuple(batch["y"].shape))
        print("Mask:", tuple(batch["mask"].shape))
        print("X dtype:", batch["x"].dtype)
        print("Y dtype:", batch["y"].dtype)

        assert torch.isfinite(batch["x"]).all()
        assert torch.isfinite(batch["y"]).all()
        assert batch["x"].shape[-1] == 17

        if split == "val_endpoints":
            assert len(loader.dataset) == 60

    print("\nAll DataLoader checks passed!")


if __name__ == "__main__":
    main()
