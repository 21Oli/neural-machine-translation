"""RNN Decoder for the baseline seq2seq model (no attention)."""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        embedding_dim,
        hidden_dim,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            output_dim,
            embedding_dim,
            padding_idx=0,
        )

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.fc_out = nn.Linear(
            hidden_dim,
            output_dim,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, cell):
        input_token = input_token.unsqueeze(1)

        embedded = self.dropout(
            self.embedding(input_token)
        )

        output, (hidden, cell) = self.rnn(
            embedded,
            (hidden, cell),
        )

        prediction = self.fc_out(
            output.squeeze(1)
        )

        return prediction, hidden, cell