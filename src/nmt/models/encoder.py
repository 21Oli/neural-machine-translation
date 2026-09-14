"""RNN Encoder for sequence-to-sequence models."""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class Encoder(nn.Module):
    """Bidirectional GRU encoder.

    Args:
        vocab_size: Source vocabulary size.
        embed_dim: Embedding dimensionality.
        hidden_dim: GRU hidden state dimensionality.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability (applied between layers).
        bidirectional: Whether to use a bidirectional GRU.
        padding_idx: Index of the padding token in the embedding layer.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

        # Project bidirectional hidden state down to hidden_dim for the decoder
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode a batch of source sequences.

        Args:
            src: Source token IDs of shape (batch_size, src_len).

        Returns:
            outputs: All hidden states, shape (batch_size, src_len, hidden_dim * num_directions).
            hidden:  Final hidden state passed to the decoder,
                     shape (num_layers, batch_size, hidden_dim).
        """
        # src → (batch, src_len, embed_dim)
        embedded = self.dropout(self.embedding(src))

        # outputs → (batch, src_len, hidden_dim * num_directions)
        # hidden  → (num_layers * num_directions, batch, hidden_dim)
        outputs, hidden = self.rnn(embedded)

        if self.bidirectional:
            # Merge forward and backward final hidden states for each layer
            # hidden shape: (num_layers*2, batch, hidden_dim)
            # Reshape to (num_layers, 2, batch, hidden_dim), concat on last dim
            hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
            # (num_layers, batch, hidden_dim*2)
            hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
            # Project to hidden_dim → (num_layers, batch, hidden_dim)
            hidden = torch.tanh(self.fc(hidden))

        return outputs, hidden
