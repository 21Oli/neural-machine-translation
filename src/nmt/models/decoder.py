"""RNN Decoder for the baseline seq2seq model (no attention)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class Decoder(nn.Module):
    """GRU decoder without attention.

    At each step the decoder receives the previous target token embedding
    concatenated with the context vector (final encoder hidden state) as input.

    Args:
        vocab_size: Target vocabulary size.
        embed_dim: Embedding dimensionality.
        hidden_dim: GRU hidden state dimensionality.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability.
        padding_idx: Padding token index.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # Input: embedding + context (= encoder final hidden of last layer)
        self.rnn = nn.GRU(
            input_size=embed_dim + hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim * 2 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_token: Tensor,
        hidden: Tensor,
        context: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Decode one step.

        Args:
            tgt_token: Previous target token IDs, shape (batch_size,).
            hidden: Decoder hidden state, shape (num_layers, batch_size, hidden_dim).
            context: Encoder context vector, shape (num_layers, batch_size, hidden_dim).
                     Typically the encoder's final hidden state.

        Returns:
            prediction: Log-softmax scores over vocabulary, shape (batch_size, vocab_size).
            hidden: Updated hidden state, shape (num_layers, batch_size, hidden_dim).
        """
        # (batch,) → (batch, 1) → (batch, 1, embed_dim)
        tgt_token = tgt_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(tgt_token))

        # Use top layer of context as input context vector
        # context[-1]: (batch, hidden_dim) → (batch, 1, hidden_dim)
        context_input = context[-1].unsqueeze(1)

        # rnn_input: (batch, 1, embed_dim + hidden_dim)
        rnn_input = torch.cat([embedded, context_input], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        # output: (batch, 1, hidden_dim)
        # Prediction combines output, context, and embedding
        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context[-1], embedded.squeeze(1)], dim=1)
        )
        return prediction, hidden
