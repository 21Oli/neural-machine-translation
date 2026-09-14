"""Attention-based Seq2Seq model (Bahdanau / Luong attention)."""

import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .encoder import Encoder


class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention mechanism.

    Args:
        hidden_dim: Decoder hidden state dimensionality.
        encoder_dim: Encoder output dimensionality (hidden_dim * num_directions).
        attention_dim: Dimensionality of the attention energy projection.
    """

    def __init__(self, hidden_dim: int, encoder_dim: int, attention_dim: int) -> None:
        super().__init__()
        self.attn_hidden = nn.Linear(hidden_dim, attention_dim)
        self.attn_encoder = nn.Linear(encoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden: Tensor, encoder_outputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute attention weights and context vector.

        Args:
            hidden: Decoder hidden state (top layer),
                    shape (batch_size, hidden_dim).
            encoder_outputs: All encoder outputs,
                             shape (batch_size, src_len, encoder_dim).

        Returns:
            context: Weighted sum of encoder outputs, shape (batch_size, encoder_dim).
            attention_weights: Normalized attention scores, shape (batch_size, src_len).
        """
        # hidden: (batch, hidden_dim) → (batch, 1, attention_dim)
        hidden_proj = self.attn_hidden(hidden).unsqueeze(1)
        # encoder_outputs: (batch, src_len, encoder_dim) → (batch, src_len, attention_dim)
        encoder_proj = self.attn_encoder(encoder_outputs)

        # energy: (batch, src_len, 1) → (batch, src_len)
        energy = self.v(torch.tanh(hidden_proj + encoder_proj)).squeeze(2)
        attention_weights = F.softmax(energy, dim=1)

        # context: (batch, 1, src_len) × (batch, src_len, encoder_dim) → (batch, encoder_dim)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attention_weights


class AttentionDecoder(nn.Module):
    """GRU decoder with Bahdanau attention.

    Args:
        vocab_size: Target vocabulary size.
        embed_dim: Embedding dimensionality.
        hidden_dim: GRU hidden state dimensionality.
        encoder_dim: Encoder output dimensionality.
        attention_dim: Attention projection dimensionality.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout probability.
        padding_idx: Padding token index.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        encoder_dim: int,
        attention_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.attention = BahdanauAttention(hidden_dim, encoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(
            input_size=embed_dim + encoder_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hidden_dim + encoder_dim + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt_token: Tensor,
        hidden: Tensor,
        encoder_outputs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Decode one step with attention.

        Args:
            tgt_token: Previous target token IDs, shape (batch_size,).
            hidden: Decoder hidden state, shape (num_layers, batch_size, hidden_dim).
            encoder_outputs: All encoder outputs,
                             shape (batch_size, src_len, encoder_dim).

        Returns:
            prediction: Logits over vocabulary, shape (batch_size, vocab_size).
            hidden: Updated hidden state, shape (num_layers, batch_size, hidden_dim).
            attention_weights: shape (batch_size, src_len).
        """
        tgt_token = tgt_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(tgt_token))  # (batch, 1, embed_dim)

        # Attend using the top-layer hidden state
        context, attention_weights = self.attention(hidden[-1], encoder_outputs)
        # context: (batch, encoder_dim) → (batch, 1, encoder_dim)
        context_input = context.unsqueeze(1)

        rnn_input = torch.cat([embedded, context_input], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        # output: (batch, 1, hidden_dim)

        prediction = self.fc_out(
            torch.cat([output.squeeze(1), context, embedded.squeeze(1)], dim=1)
        )
        return prediction, hidden, attention_weights


class AttentionSeq2Seq(nn.Module):
    """End-to-end attention-based sequence-to-sequence model.

    Args:
        encoder: Encoder module.
        decoder: AttentionDecoder module.
        src_pad_idx: Source padding index.
        tgt_pad_idx: Target padding index.
        device: Torch device.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: AttentionDecoder,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device or torch.device("cpu")

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """Run forward pass.

        Args:
            src: Source token IDs, shape (batch_size, src_len).
            tgt: Target token IDs, shape (batch_size, tgt_len).
            teacher_forcing_ratio: Teacher forcing probability.

        Returns:
            outputs: Logits, shape (batch_size, tgt_len - 1, tgt_vocab_size).
            attentions: Attention weights, shape (batch_size, tgt_len - 1, src_len).
        """
        batch_size, tgt_len = tgt.shape
        src_len = src.shape[1]
        tgt_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, tgt_len - 1, tgt_vocab_size, device=self.device)
        attentions = torch.zeros(batch_size, tgt_len - 1, src_len, device=self.device)

        encoder_outputs, hidden = self.encoder(src)
        dec_input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden, attn_weights = self.decoder(dec_input, hidden, encoder_outputs)
            outputs[:, t - 1] = output
            attentions[:, t - 1] = attn_weights

            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            dec_input = tgt[:, t] if use_teacher_forcing else top1

        return outputs, attentions

    @torch.no_grad()
    def translate(
        self,
        src: Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 100,
    ) -> Tuple[Tensor, Tensor]:
        """Greedy decode a single source sequence.

        Args:
            src: Source token IDs, shape (1, src_len).
            bos_idx: Beginning-of-sequence token index.
            eos_idx: End-of-sequence token index.
            max_len: Maximum decoding steps.

        Returns:
            tokens: Predicted token IDs, shape (output_len,).
            attentions: Attention weights, shape (output_len, src_len).
        """
        self.eval()
        encoder_outputs, hidden = self.encoder(src)
        dec_input = torch.tensor([bos_idx], device=self.device)
        tokens, attentions = [], []

        for _ in range(max_len):
            output, hidden, attn_weights = self.decoder(dec_input, hidden, encoder_outputs)
            top1 = output.argmax(dim=1)
            tokens.append(top1.item())
            attentions.append(attn_weights.squeeze(0))
            if top1.item() == eos_idx:
                break
            dec_input = top1

        return (
            torch.tensor(tokens, device=self.device),
            torch.stack(attentions) if attentions else torch.empty(0),
        )
