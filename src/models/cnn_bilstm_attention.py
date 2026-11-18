
import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from src.models.lstm import RULLSTM
from src.models.components.additive_attention import (
    MaskedAdditiveAttention,
)


class CNNBiLSTMAttention(nn.Module):
    """
    CNN + BiLSTM + custom attention for RUL regression.

    Input:
        x:    [B, T, F]
        mask: [B, T], True for real observations

    Output:
        predictions: [B, 1]
    """

    def __init__(
        self,
        input_size: int = 17,
        cnn_channels: int = 64,
        hidden_size: int = 64,
        num_layers: int = 2,
        attention_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        if input_size <= 0 or cnn_channels <= 0:
            raise ValueError("Input and CNN dimensions must be positive")

        if hidden_size <= 0 or num_layers <= 0:
            raise ValueError("LSTM dimensions must be positive")

        if not 0 <= dropout < 1:
            raise ValueError("Invalid dropout")

        self.input_size = input_size

        self.cnn = nn.Sequential(
            nn.Conv1d(
                input_size,
                cnn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.Conv1d(
                cnn_channels,
                cnn_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
        )

        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        self.attention = MaskedAdditiveAttention(
            input_dim=hidden_size * 2,
            attention_dim=attention_dim,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        return_attention: bool = False,
    ):
        if x.ndim != 3:
            raise ValueError("Expected input [B, T, F]")

        if x.shape[-1] != self.input_size:
            raise ValueError("Incorrect feature count")

        # Reuse the validated left-padding conversion.
        aligned, lengths = RULLSTM.prepare_sequences(
            x, mask
        )

        batch_size, seq_len, _ = aligned.shape

        # After alignment, valid readings start at position zero.
        time = torch.arange(
            seq_len,
            device=x.device,
        ).unsqueeze(0)

        valid_mask = time < lengths.unsqueeze(1)

        # Exclude repeated trailing padding values from CNN input.
        aligned = aligned.masked_fill(
            ~valid_mask.unsqueeze(-1),
            0.0,
        )

        # Conv1d requires [B, Channels, T].
        features = self.cnn(
            aligned.transpose(1, 2)
        ).transpose(1, 2)

        # Exclude invalid CNN output positions.
        features = features.masked_fill(
            ~valid_mask.unsqueeze(-1),
            0.0,
        )

        packed = pack_padded_sequence(
            features,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_output, _ = self.bilstm(packed)

        sequence_output, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=seq_len,
        )

        context, attention_weights = self.attention(
            sequence_output,
            valid_mask,
        )

        predictions = self.regression_head(context)

        if return_attention:
            return predictions, attention_weights

        return predictions

