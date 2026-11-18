
import pytest
import torch

from src.models.transformer import RULTransformer


@pytest.fixture
def model():
    torch.manual_seed(42)

    net = RULTransformer(
        input_size=17,
        d_model=64,
        num_heads=4,
        num_layers=2,
    )
    net.eval()
    return net


def test_output_shape(model):
    x = torch.randn(8, 30, 17)
    mask = torch.ones(8, 30, dtype=torch.bool)

    with torch.no_grad():
        y = model(x, mask)

    assert y.shape == (8, 1)
    assert torch.isfinite(y).all()


def test_mixed_sequence_lengths(model):
    x = torch.randn(3, 30, 17)
    mask = torch.zeros(3, 30, dtype=torch.bool)

    mask[0, :] = True
    mask[1, -10:] = True
    mask[2, -1:] = True

    with torch.no_grad():
        y = model(x, mask)

    assert y.shape == (3, 1)
    assert torch.isfinite(y).all()


def test_padding_invariance(model):
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
        a, b, rtol=0, atol=1e-5
    )


def test_temporal_order_matters(model):
    x = torch.randn(2, 30, 17)
    mask = torch.ones(2, 30, dtype=torch.bool)

    reversed_x = torch.flip(x, dims=[1])

    with torch.no_grad():
        a = model(x, mask)
        b = model(reversed_x, mask)

    assert not torch.allclose(a, b, atol=1e-7)


def test_gradient_flow():
    model = RULTransformer(
        input_size=17,
        d_model=32,
        num_heads=4,
        num_layers=1,
        dim_feedforward=64,
    )

    model.train()

    x = torch.randn(4, 30, 17)
    mask = torch.ones(4, 30, dtype=torch.bool)

    output = model(x, mask)
    loss = output.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
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


def test_invalid_attention_heads():
    with pytest.raises(ValueError):
        RULTransformer(
            d_model=64,
            num_heads=3,
        )
