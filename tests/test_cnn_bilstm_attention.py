
import pytest
import torch

from src.models.components.additive_attention import (
    MaskedAdditiveAttention,
)
from src.models.cnn_bilstm_attention import CNNBiLSTMAttention


def test_attention_weights_sum_to_one():
    attention = MaskedAdditiveAttention(16, 8)

    x = torch.randn(3, 10, 16)
    mask = torch.ones(3, 10, dtype=torch.bool)

    context, weights = attention(x, mask)

    assert context.shape == (3, 16)
    assert weights.shape == (3, 10)

    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones(3),
    )


def test_padding_gets_zero_attention():
    attention = MaskedAdditiveAttention(8, 4)

    x = torch.randn(2, 5, 8)

    mask = torch.tensor([
        [True, True, True, False, False],
        [True, False, False, False, False],
    ])

    _, weights = attention(x, mask)

    assert torch.all(weights[~mask] == 0)


def test_attention_padding_invariance():
    torch.manual_seed(42)
    attention = MaskedAdditiveAttention(8, 4)
    attention.eval()

    x = torch.randn(2, 5, 8)
    mask = torch.tensor([
        [True, True, True, False, False],
        [True, True, False, False, False],
    ])

    changed = x.clone()
    changed[~mask] = 9999.0

    with torch.no_grad():
        context_a, _ = attention(x, mask)
        context_b, _ = attention(changed, mask)

    torch.testing.assert_close(
        context_a, context_b, atol=1e-6, rtol=0
    )


def test_attention_rejects_empty_mask():
    attention = MaskedAdditiveAttention(8)
    x = torch.randn(1, 5, 8)
    mask = torch.zeros(1, 5, dtype=torch.bool)

    with pytest.raises(ValueError):
        attention(x, mask)


def test_hybrid_output_shape():
    model = CNNBiLSTMAttention()
    model.eval()

    x = torch.randn(4, 30, 17)
    mask = torch.ones(4, 30, dtype=torch.bool)

    with torch.no_grad():
        output = model(x, mask)

    assert output.shape == (4, 1)
    assert torch.isfinite(output).all()


def test_hybrid_mixed_sequence_lengths():
    model = CNNBiLSTMAttention()
    model.eval()

    x = torch.randn(3, 30, 17)

    mask = torch.zeros(3, 30, dtype=torch.bool)
    mask[0, :] = True
    mask[1, -10:] = True
    mask[2, -1:] = True

    with torch.no_grad():
        output, weights = model(
            x, mask, return_attention=True
        )

    assert output.shape == (3, 1)
    assert weights.shape == (3, 30)

    lengths = mask.sum(dim=1)

    for i, length in enumerate(lengths):
        assert torch.all(weights[i, length:] == 0)

    torch.testing.assert_close(
        weights.sum(dim=1),
        torch.ones(3),
    )


def test_hybrid_without_attention_uses_uniform_valid_weights():
    torch.manual_seed(42)
    attention_model = CNNBiLSTMAttention(
        cnn_channels=8,
        hidden_size=8,
        num_layers=1,
    )

    torch.manual_seed(42)
    model = CNNBiLSTMAttention(
        cnn_channels=8,
        hidden_size=8,
        num_layers=1,
        use_attention=False,
    )
    model.eval()

    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value,
            attention_model.state_dict()[key],
        )

    x = torch.randn(2, 6, 17)
    mask = torch.tensor([
        [True, True, True, True, True, True],
        [False, False, False, True, True, True],
    ])

    with torch.no_grad():
        predictions, weights = model(
            x, mask, return_attention=True
        )

    assert predictions.shape == (2, 1)
    assert weights.shape == (2, 6)
    torch.testing.assert_close(
        weights,
        torch.tensor([
            [1 / 6] * 6,
            [1 / 3, 1 / 3, 1 / 3, 0, 0, 0],
        ]),
    )
    assert model.attention is None
    assert not any(
        key.startswith("attention.")
        for key in model.state_dict()
    )


def test_hybrid_padding_invariance():
    torch.manual_seed(42)
    model = CNNBiLSTMAttention()
    model.eval()

    x = torch.randn(2, 30, 17)

    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[0, -5:] = True
    mask[1, -12:] = True

    changed = x.clone()
    changed[~mask] = 9999.0

    with torch.no_grad():
        a = model(x, mask)
        b = model(changed, mask)

    torch.testing.assert_close(
        a, b, atol=1e-6, rtol=0
    )


def test_hybrid_gradient_flow():
    model = CNNBiLSTMAttention(
        cnn_channels=16,
        hidden_size=16,
        attention_dim=16,
        num_layers=1,
    )

    x = torch.randn(4, 30, 17)
    mask = torch.ones(4, 30, dtype=torch.bool)
    target = torch.randn(4, 1)

    prediction = model(x, mask)
    loss = (prediction - target).square().mean()
    loss.backward()

    assert torch.isfinite(loss)

    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()


def test_no_attention_uniform_weights():
    model = CNNBiLSTMAttention(use_attention=False)
    model.eval()

    x = torch.randn(2, 30, 17)

    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[0, -5:] = True
    mask[1, -10:] = True

    with torch.no_grad():
        predictions, weights = model(
            x, mask, return_attention=True
        )

    expected = torch.zeros_like(weights)
    expected[0, :5] = 1.0 / 5
    expected[1, :10] = 1.0 / 10

    assert predictions.shape == (2, 1)
    assert model.attention is None

    torch.testing.assert_close(weights, expected)


def test_no_attention_gradient_flow():
    model = CNNBiLSTMAttention(
        cnn_channels=16,
        hidden_size=16,
        num_layers=1,
        use_attention=False,
    )

    x = torch.randn(4, 30, 17)
    mask = torch.ones(4, 30, dtype=torch.bool)

    output = model(x, mask)
    loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(loss)

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_shared_initialization_matches():
    torch.manual_seed(42)
    full = CNNBiLSTMAttention(use_attention=True)

    torch.manual_seed(42)
    ablated = CNNBiLSTMAttention(use_attention=False)

    full_state = full.state_dict()
    ablated_state = ablated.state_dict()

    for name, value in ablated_state.items():
        torch.testing.assert_close(
            value,
            full_state[name],
            rtol=0,
            atol=0,
        )


def test_attention_ablation_reduces_parameters():
    full = CNNBiLSTMAttention()
    ablated = CNNBiLSTMAttention(use_attention=False)

    full_params = sum(p.numel() for p in full.parameters())
    ablated_params = sum(p.numel() for p in ablated.parameters())

    assert full_params == 198209
    assert ablated_params == 189889



def test_hybrid_invalid_feature_count():
    model = CNNBiLSTMAttention()

    x = torch.randn(2, 30, 10)
    mask = torch.ones(2, 30, dtype=torch.bool)

    with pytest.raises(ValueError):
        model(x, mask)
