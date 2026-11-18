
import torch
from torch import nn


class MaskedAdditiveAttention(nn.Module):
    """
    Custom additive temporal attention.

    Input:
        x:    [B, T, D]
        mask: [B, T], True = valid timestep

    Returns:
        context: [B, D]
        weights: [B, T]
    """

    def __init__(
        self,
        input_dim: int,
        attention_dim: int = 64,
    ):
        super().__init__()

        if input_dim <= 0 or attention_dim <= 0:
            raise ValueError("Attention dimensions must be positive")

        self.projection = nn.Linear(
            input_dim,
            attention_dim,
        )

        self.score = nn.Linear(
            attention_dim,
            1,
            bias=False,
        )

    def forward(self, x, mask):
        if x.ndim != 3:
            raise ValueError("Expected input [B, T, D]")

        if mask.shape != x.shape[:2]:
            raise ValueError("Mask shape mismatch")

        if mask.dtype != torch.bool:
            raise TypeError("Mask must be boolean")

        if mask.device != x.device:
            raise ValueError("Mask and input must share device")

        if not mask.any(dim=1).all():
            raise ValueError("Every sequence needs valid observations")

        # Learn one importance score for each timestep.
        energy = torch.tanh(self.projection(x))
        scores = self.score(energy).squeeze(-1)

        # Padding receives no attention probability.
        scores = scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        weights = torch.softmax(scores, dim=1)

        # Weighted combination of temporal representations.
        context = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        return context, weights
