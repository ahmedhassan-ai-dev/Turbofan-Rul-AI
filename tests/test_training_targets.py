
import pytest
import torch

from src.training.trainer import prepare_training_targets


def test_uncapped_targets_unchanged():
    y = torch.tensor([[30.0], [200.0]])

    output = prepare_training_targets(y, None)

    torch.testing.assert_close(output, y)


def test_rul_cap_125():
    y = torch.tensor([
        [0.0],
        [30.0],
        [100.0],
        [125.0],
        [180.0],
        [361.0],
    ])

    expected = torch.tensor([
        [0.0],
        [30.0],
        [100.0],
        [125.0],
        [125.0],
        [125.0],
    ])

    output = prepare_training_targets(y, 125)

    torch.testing.assert_close(output, expected)


def test_original_targets_preserved():
    y = torch.tensor([[30.0], [200.0]])
    original = y.clone()

    _ = prepare_training_targets(y, 125)

    torch.testing.assert_close(y, original)


def test_invalid_cap_rejected():
    y = torch.tensor([[100.0]])

    with pytest.raises(ValueError):
        prepare_training_targets(y, 0)


def test_training_gradient_preserved():
    predictions = torch.tensor(
        [[20.0], [150.0]],
        requires_grad=True,
    )

    targets = torch.tensor([[30.0], [200.0]])
    capped = prepare_training_targets(targets, 125)

    loss = torch.nn.functional.mse_loss(
        predictions, capped
    )
    loss.backward()

    assert predictions.grad is not None
    assert torch.isfinite(predictions.grad).all()
