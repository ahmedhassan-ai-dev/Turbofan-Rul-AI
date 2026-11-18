
import math

import pytest
import torch

from src.datasets.datamodule import (
    CMapssDataset,
    create_dataloader,
)


@pytest.fixture(scope="module")
def datasets():
    return {
        "train": CMapssDataset("train"),
        "val": CMapssDataset("val"),
        "val_endpoints": CMapssDataset("val_endpoints"),
    }


def test_dataset_sizes(datasets):
    assert len(datasets["train"]) == 16561
    assert len(datasets["val"]) == 4070
    assert len(datasets["val_endpoints"]) == 60


def test_dataset_tensor_contract(datasets):
    for dataset in datasets.values():
        sample = dataset[0]

        assert sample["x"].shape == (30, 17)
        assert sample["y"].shape == (1,)
        assert sample["mask"].shape == (30,)

        assert sample["x"].dtype == torch.float32
        assert sample["y"].dtype == torch.float32
        assert sample["mask"].dtype == torch.bool

        assert torch.isfinite(sample["x"]).all()
        assert torch.isfinite(sample["y"]).all()


def test_no_engine_overlap(datasets):
    train_ids = set(
        datasets["train"].engine_ids.tolist()
    )
    val_ids = set(
        datasets["val"].engine_ids.tolist()
    )

    assert train_ids.isdisjoint(val_ids)


def test_endpoint_targets(datasets):
    endpoints = datasets["val_endpoints"]

    actual = sorted(endpoints.y.squeeze(-1).tolist())

    assert actual == (
        [30.0] * 20
        + [60.0] * 20
        + [100.0] * 20
    )


def test_padding_mask(datasets):
    dataset = datasets["train"]

    for index in range(len(dataset)):
        mask = dataset.mask[index]

        assert mask.any()

        # All padding appears before valid observations.
        assert not torch.any(
            mask[:-1] & ~mask[1:]
        )

        assert mask[-1].item() is True


def test_batch_count():
    loader = create_dataloader(
        "train",
        batch_size=64,
    )

    assert len(loader) == math.ceil(16561 / 64)


def test_all_samples_preserved():
    loader = create_dataloader(
        "val",
        batch_size=64,
    )

    total = sum(
        len(batch["y"])
        for batch in loader
    )

    assert total == 4070


def test_training_shuffle_reproducible():
    loader_a = create_dataloader(
        "train", batch_size=64, seed=42
    )
    loader_b = create_dataloader(
        "train", batch_size=64, seed=42
    )

    def first_batch_keys(loader):
        batch = next(iter(loader))
        return torch.stack(
            [batch["engine_id"], batch["cycle"]],
            dim=1,
        )

    assert torch.equal(
        first_batch_keys(loader_a),
        first_batch_keys(loader_b),
    )


def test_validation_is_not_shuffled(datasets):
    loader = create_dataloader(
        "val_endpoints",
        batch_size=16,
    )

    observed = torch.cat([
        torch.stack(
            [batch["engine_id"], batch["cycle"]],
            dim=1,
        )
        for batch in loader
    ])

    expected = torch.stack([
        datasets["val_endpoints"].engine_ids,
        datasets["val_endpoints"].cycles,
    ], dim=1)

    assert torch.equal(observed, expected)


def test_invalid_split():
    with pytest.raises(ValueError):
        CMapssDataset("test")


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        create_dataloader(
            "train", batch_size=0
        )
