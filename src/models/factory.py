"""Shared model factory.

Every training entry point (single runs, experiment sweep, demo,
evaluation) builds architectures through this module so that a model
name always maps to exactly one architecture configuration.
"""

import json
from pathlib import Path

import torch
from torch import nn

from src.models.cnn_bilstm_attention import CNNBiLSTMAttention
from src.models.lstm import RULLSTM
from src.models.transformer import RULTransformer


ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_DIR = ROOT / "configs" / "models"

DEFAULT_CONFIGS: dict[str, dict] = {
    "lstm": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.2,
    },
    "transformer": {
        "d_model": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
    },
    "cnn_bilstm_attention": {
        "cnn_channels": 64,
        "hidden_size": 64,
        "num_layers": 2,
        "attention_dim": 64,
        "dropout": 0.2,
        "use_attention": True,
    },
}

MODEL_CLASSES = {
    "lstm": RULLSTM,
    "transformer": RULTransformer,
    "cnn_bilstm_attention": CNNBiLSTMAttention,
}

# Architecture aliases used by the attention ablation study. The alias
# keeps the CNN, BiLSTM and regression head identical and only swaps the
# temporal pooling operator.
MODEL_ALIASES: dict[str, tuple[str, dict]] = {
    "cnn_bilstm_mean": (
        "cnn_bilstm_attention",
        {"use_attention": False},
    ),
    "cnn_bilstm_no_attention": (
        "cnn_bilstm_attention",
        {"use_attention": False},
    ),
}


def resolve_model(model_name: str) -> tuple[str, dict]:
    """Map a model name or alias to (base class name, overrides)."""

    if model_name in MODEL_ALIASES:
        base, overrides = MODEL_ALIASES[model_name]
        return base, dict(overrides)

    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unsupported model: {model_name}")

    return model_name, {}


def load_model_config(model_name: str) -> dict:
    """Return the architecture config for ``model_name``."""

    base, overrides = resolve_model(model_name)

    config = dict(DEFAULT_CONFIGS[base])
    config.update(overrides)

    path = MODEL_CONFIG_DIR / f"{model_name}.json"

    if path.is_file():
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored.pop("model", None)
        stored.pop("input_size", None)
        config.update(stored)

    return config


def build_model(
    model_name: str,
    input_size: int = 17,
    **overrides,
) -> nn.Module:
    """Build a model with the registered architecture configuration."""

    base, alias_overrides = resolve_model(model_name)

    if input_size < 1:
        raise ValueError("input_size must be positive")

    config = load_model_config(model_name)
    config.update(alias_overrides)
    config.update(overrides)

    if "max_seq_length" in config and base != "transformer":
        config.pop("max_seq_length")

    return MODEL_CLASSES[base](
        input_size=input_size,
        **config,
    )


def resolve_architecture(model_name: str, **overrides) -> tuple[str, dict]:
    """Return (class name, full kwargs) for checkpoint metadata."""

    base, alias_overrides = resolve_model(model_name)

    kwargs = load_model_config(model_name)
    kwargs.update(alias_overrides)
    kwargs.update(overrides)

    if "max_seq_length" in kwargs and base != "transformer":
        kwargs.pop("max_seq_length")

    return base, kwargs


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: nn.Module) -> float:
    """Approximate in-memory size of the parameters in MiB."""

    total = 0
    for parameter in model.parameters():
        total += parameter.numel() * parameter.element_size()

    for buffer in model.buffers():
        total += buffer.numel() * buffer.element_size()

    return float(total) / (1024 * 1024)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    model_name: str,
    model_kwargs: dict,
    metadata: dict,
) -> None:
    """Save a reloadable checkpoint with its architecture config."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_name": model_name,
            "model_kwargs": model_kwargs,
            "model_state_dict": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_checkpoint(path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Rebuild a model from a checkpoint written by ``save_checkpoint``."""

    payload = torch.load(
        Path(path),
        map_location=device,
        weights_only=False,
    )

    model = build_model(
        payload["model_name"],
        **payload["model_kwargs"],
    )

    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()

    return model, payload.get("metadata", {})