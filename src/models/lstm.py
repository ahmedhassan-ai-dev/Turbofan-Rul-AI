
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class RULLSTM(nn.Module):
    """
    Padding-aware stacked LSTM for turbofan RUL prediction.

    Input:
        x:    [batch, sequence_length, input_size]
        mask: [batch, sequence_length]
              True = real observation
              False = left padding

    Output:
        RUL predictions: [batch, 1]
    """

    def __init__(
        self,
        input_size: int = 17,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        if input_size < 1 or hidden_size < 1:
            raise ValueError("Feature and hidden sizes must be positive")

        if num_layers < 1:
            raise ValueError("num_layers must be positive")

        if not 0 <= dropout < 1:
            raise ValueError("dropout must be between 0 and 1")

        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    @staticmethod
    def prepare_sequences(
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert left-padded windows into right-padded windows.

        Example:
            [P, P, A, B, C] -> [A, B, C, C, C]

        Repeated trailing values are ignored by packing.
        """

        if x.ndim != 3:
            raise ValueError("Expected x with shape [B, T, F]")

        if mask.shape != x.shape[:2]:
            raise ValueError("Mask shape mismatch")

        if mask.dtype != torch.bool:
            raise TypeError("Mask must be a boolean tensor")

        if mask.device != x.device:
            raise ValueError("Input and mask must use the same device")

        if not torch.isfinite(x).all():
            raise ValueError("Input contains non-finite values")

        lengths = mask.sum(dim=1).long()

        if torch.any(lengths == 0):
            raise ValueError("Every sequence needs a valid observation")

        # Validate left-padding structure:
        # False ... False, True ... True.
        if not mask[:, -1].all():
            raise ValueError("Last time step must be valid")

        if torch.any(mask[:, :-1] & ~mask[:, 1:]):
            raise ValueError("Invalid left-padding mask")

        batch_size, sequence_length, num_features = x.shape

        # Index of the first real observation in each sequence.
        offsets = sequence_length - lengths

        time_indices = torch.arange(
            sequence_length,
            device=x.device,
        ).unsqueeze(0)

        # Move valid observations to the beginning.
        indices = time_indices + offsets.unsqueeze(1)

        # Repeat the final real observation in trailing positions.
        # The packed LSTM will never process those positions.
        indices = indices.clamp(max=sequence_length - 1)

        indices = indices.unsqueeze(-1).expand(
            batch_size,
            sequence_length,
            num_features,
        )

        aligned = torch.gather(
            x,
            dim=1,
            index=indices,
        )

        return aligned, lengths

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} features, "
                f"received {x.shape[-1]}"
            )

        aligned, lengths = self.prepare_sequences(x, mask)

        packed = pack_padded_sequence(
            aligned,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        _, (hidden_state, _) = self.lstm(packed)

        # Final hidden state of the last LSTM layer.
        representation = hidden_state[-1]

        predictions = self.regression_head(representation)

        return predictions
