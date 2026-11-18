
import pytest

from src.train import build_model
from src.models.cnn_bilstm_attention import CNNBiLSTMAttention
from src.models.lstm import RULLSTM
from src.models.transformer import RULTransformer


def test_build_lstm():
    model = build_model("lstm", input_size=17)
    assert isinstance(model, RULLSTM)


def test_build_transformer():
    model = build_model("transformer", input_size=17)
    assert isinstance(model, RULTransformer)


def test_build_cnn_bilstm_attention():
    model = build_model(
        "cnn_bilstm_attention",
        input_size=17,
    )
    assert isinstance(model, CNNBiLSTMAttention)
    assert model.input_size == 17


def test_build_cnn_bilstm_no_attention():
    model = build_model(
        "cnn_bilstm_no_attention",
        input_size=17,
    )

    assert isinstance(model, CNNBiLSTMAttention)
    assert model.input_size == 17
    assert model.use_attention is False
    assert model.attention is None


def test_invalid_model():
    with pytest.raises(ValueError):
        build_model("unknown", input_size=17)



def test_build_hybrid_without_attention():
    model = build_model(
        "cnn_bilstm_no_attention",
        input_size=17,
    )

    assert isinstance(model, CNNBiLSTMAttention)
    assert model.use_attention is False
    assert model.attention is None