
import pytest

from src.train import build_model
from src.models.lstm import RULLSTM
from src.models.transformer import RULTransformer


def test_build_lstm():
    model = build_model("lstm", input_size=17)
    assert isinstance(model, RULLSTM)


def test_build_transformer():
    model = build_model("transformer", input_size=17)
    assert isinstance(model, RULTransformer)


def test_invalid_model():
    with pytest.raises(ValueError):
        build_model("unknown", input_size=17)
