"""Attention ablation tests: same hybrid, mean pooling instead of attention."""

import torch

import torch.nn.utils.rnn as rnn_utils

from src.models.cnn_bilstm_attention import CNNBiLSTMAttention
from src.models.factory import build_model
from src.models.lstm import RULLSTM


def test_ablation_has_no_attention_module():
    model = CNNBiLSTMAttention(use_attention=False)

    assert model.attention is None
    assert model.use_attention is False


def test_ablation_output_shape_and_padding_invariance():
    torch.manual_seed(0)

    model = CNNBiLSTMAttention(use_attention=False)
    model.eval()

    x = torch.randn(2, 30, 17)
    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[0, -8:] = True
    mask[1, -20:] = True

    changed = x.clone()
    changed[~mask] = 1234.0

    with torch.no_grad():
        a = model(x, mask)
        b = model(changed, mask)

    assert a.shape == (2, 1)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=0)


def test_mean_pooling_matches_manual_computation():
    """Masked mean pooling must equal the masked average of BiLSTM rows."""

    torch.manual_seed(3)

    model = CNNBiLSTMAttention(
        cnn_channels=8,
        hidden_size=6,
        attention_dim=6,
        num_layers=1,
        use_attention=False,
    )
    model.eval()

    x = torch.randn(1, 12, 17)
    mask = torch.zeros(1, 12, dtype=torch.bool)
    mask[0, -5:] = True

    captured = {}

    def hook(module, inputs, output):
        # The regression head receives the pooled context vector.
        captured["context"] = inputs[0]

    handle = model.regression_head.register_forward_hook(hook)

    with torch.no_grad():
        prediction = model(x, mask)

    handle.remove()

    # Recompute the expected context from the BiLSTM sequence output.
    aligned, lengths = RULLSTM.prepare_sequences(x, mask)

    time = torch.arange(12).unsqueeze(0)
    valid = time < lengths.unsqueeze(1)

    features = model.cnn(
        aligned.masked_fill(~valid.unsqueeze(-1), 0.0)
        .transpose(1, 2)
    ).transpose(1, 2)

    features = features.masked_fill(~valid.unsqueeze(-1), 0.0)

    packed = rnn_utils.pack_padded_sequence(
        features,
        lengths=lengths.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    with torch.no_grad():
        packed_output, _ = model.bilstm(packed)
        sequence, _ = rnn_utils.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=12,
        )

    weights = valid.unsqueeze(-1).float()
    expected = (sequence * weights).sum(dim=1) / weights.sum(dim=1)

    torch.testing.assert_close(
        captured["context"], expected, atol=1e-5, rtol=1e-5
    )

    with torch.no_grad():
        expected_prediction = model.regression_head(expected)

    torch.testing.assert_close(prediction, expected_prediction)


def test_ablation_alias_builds_same_backbone():
    hybrid = build_model("cnn_bilstm_attention", input_size=17)
    ablated = build_model("cnn_bilstm_mean", input_size=17)

    assert isinstance(ablated, CNNBiLSTMAttention)
    assert hybrid.attention is not None
    assert ablated.attention is None

    hybrid_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in hybrid.state_dict().items()
        if not name.startswith("attention.")
    }
    ablated_shapes = {
        name: tuple(parameter.shape)
        for name, parameter in ablated.state_dict().items()
    }

    assert hybrid_shapes == ablated_shapes

    # The ablation removes exactly the attention parameters.
    assert len(hybrid.state_dict()) > len(ablated.state_dict())


def test_ablation_gradients_flow():
    model = CNNBiLSTMAttention(
        cnn_channels=8,
        hidden_size=6,
        attention_dim=6,
        num_layers=1,
        use_attention=False,
    )

    x = torch.randn(3, 20, 17)
    mask = torch.ones(3, 20, dtype=torch.bool)
    target = torch.randn(3, 1)

    loss = (model(x, mask) - target).square().mean()
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
