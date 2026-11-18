
import pytest
import torch

from src.models.lstm import RULLSTM


@pytest.fixture
def model():
    torch.manual_seed(42)

    net = RULLSTM(
        input_size=17,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )

    net.eval()
    return net


def test_output_shape(model):
    x = torch.randn(8, 30, 17)
    mask = torch.ones(8, 30, dtype=torch.bool)

    with torch.no_grad():
        output = model(x, mask)

    assert output.shape == (8, 1)
    assert torch.isfinite(output).all()


def test_mixed_sequence_lengths(model):
    x = torch.randn(3, 30, 17)

    mask = torch.zeros(3, 30, dtype=torch.bool)
    mask[0, -30:] = True
    mask[1, -10:] = True
    mask[2, -1:] = True

    with torch.no_grad():
        output = model(x, mask)

    assert output.shape == (3, 1)
    assert torch.isfinite(output).all()


def test_padding_invariance(model):
    x = torch.randn(2, 30, 17)

    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[0, -5:] = True
    mask[1, -12:] = True

    modified = x.clone()

    # Change only invalid/padded sensor readings.
    modified[~mask] = 9999.0

    with torch.no_grad():
        original_pred = model(x, mask)
        modified_pred = model(modified, mask)

    torch.testing.assert_close(
        original_pred,
        modified_pred,
        rtol=0,
        atol=1e-6,
    )


def test_correct_temporal_order():
    x = torch.tensor([
        [[0.0], [0.0], [1.0], [2.0], [3.0]]
    ])

    mask = torch.tensor([
        [False, False, True, True, True]
    ])

    aligned, lengths = RULLSTM.prepare_sequences(
        x, mask
    )

    assert lengths.tolist() == [3]

    torch.testing.assert_close(
        aligned[0, :3, 0],
        torch.tensor([1.0, 2.0, 3.0]),
    )


def test_gradient_flow():
    torch.manual_seed(42)

    model = RULLSTM(
        input_size=17,
        hidden_size=32,
        num_layers=2,
    )

    model.train()

    x = torch.randn(4, 30, 17)
    mask = torch.ones(4, 30, dtype=torch.bool)

    targets = torch.tensor([
        [30.0], [60.0], [90.0], [120.0]
    ])

    predictions = model(x, mask)

    loss = torch.nn.functional.mse_loss(
        predictions, targets
    )

    loss.backward()

    assert torch.isfinite(loss)

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(grad is not None for grad in gradients)

    assert all(
        torch.isfinite(grad).all()
        for grad in gradients
    )


def test_invalid_mask(model):
    x = torch.randn(2, 30, 17)
    mask = torch.zeros(2, 30, dtype=torch.bool)

    with pytest.raises(ValueError):
        model(x, mask)


def test_invalid_feature_count(model):
    x = torch.randn(2, 30, 10)
    mask = torch.ones(2, 30, dtype=torch.bool)

    with pytest.raises(ValueError):
        model(x, mask)


def test_invalid_padding_pattern(model):
    x = torch.randn(1, 5, 17)

    mask = torch.tensor([
        [True, False, True, True, True]
    ])

    with pytest.raises(ValueError):
        model(x, mask)
