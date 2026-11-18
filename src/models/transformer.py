
import torch
from torch import nn


class RULTransformer(nn.Module):
    """
    Padding-aware Transformer Encoder for RUL prediction.

    x:    [B, T, F]
    mask: [B, T], True for valid readings
    output: [B, 1]
    """

    def __init__(
        self,
        input_size: int = 17,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()

        if input_size <= 0 or d_model <= 0:
            raise ValueError("Model dimensions must be positive")

        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        if num_layers <= 0 or dim_feedforward <= 0:
            raise ValueError("Layer dimensions must be positive")

        if max_seq_length <= 0:
            raise ValueError("max_seq_length must be positive")

        if not 0 <= dropout < 1:
            raise ValueError("Invalid dropout")

        self.input_size = input_size
        self.max_seq_length = max_seq_length

        self.input_projection = nn.Linear(input_size, d_model)

        self.position_embedding = nn.Embedding(
            max_seq_length, d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.final_norm = nn.LayerNorm(d_model)

        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError("Expected input [B, T, F]")

        batch_size, sequence_length, features = x.shape

        if features != self.input_size:
            raise ValueError("Incorrect input feature count")

        if sequence_length > self.max_seq_length:
            raise ValueError("Sequence exceeds maximum length")

        if mask.shape != (batch_size, sequence_length):
            raise ValueError("Incorrect mask shape")

        if mask.dtype != torch.bool:
            raise TypeError("Mask must be boolean")

        if mask.device != x.device:
            raise ValueError("Input and mask must share device")

        if not torch.isfinite(x).all():
            raise ValueError("Non-finite input")

        if not mask.any(dim=1).all():
            raise ValueError("Each window needs valid readings")

        # Position numbering begins with the first REAL reading.
        # This prevents left padding from shifting time positions.
        positions = (
            mask.long().cumsum(dim=1) - 1
        ).clamp(min=0)

        # Remove arbitrary sensor values in padded positions.
        valid_x = x.masked_fill(~mask.unsqueeze(-1), 0.0)

        hidden = (
            self.input_projection(valid_x)
            + self.position_embedding(positions)
        )

        hidden = self.encoder(
            hidden,
            src_key_padding_mask=~mask,
        )

        hidden = self.final_norm(hidden)

        # Pool over actual readings only.
        weights = mask.unsqueeze(-1).to(hidden.dtype)

        representation = (
            (hidden * weights).sum(dim=1)
            / weights.sum(dim=1).clamp(min=1)
        )

        return self.regression_head(representation)
