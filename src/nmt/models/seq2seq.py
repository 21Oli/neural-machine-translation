"""Baseline Seq2Seq model (encoder + decoder, no attention)."""

import random
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .decoder import Decoder
from .encoder import Encoder


class Seq2Seq(nn.Module):
    """Sequence-to-sequence model without attention.

    Args:
        encoder: Encoder module.
        decoder: Decoder module.
        src_pad_idx: Source padding index (used for masking, reserved).
        tgt_pad_idx: Target padding index.
        device: Torch device.
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
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
    ) -> Tensor:
        """Run the full seq2seq forward pass.

        Args:
            src: Source token IDs, shape (batch_size, src_len).
            tgt: Target token IDs, shape (batch_size, tgt_len).
            teacher_forcing_ratio: Probability of using ground-truth target as
                next decoder input (1.0 = always use ground truth).

        Returns:
            outputs: Logits for each target position,
                     shape (batch_size, tgt_len - 1, tgt_vocab_size).
        """
        batch_size, tgt_len = tgt.shape
        tgt_vocab_size = self.decoder.vocab_size

        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, tgt_len - 1, tgt_vocab_size, device=self.device)

        # Encode the source sequence
        encoder_outputs, hidden = self.encoder(src)

        # First decoder input: <bos> token (first column of tgt)
        dec_input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(dec_input, hidden, hidden)
            outputs[:, t - 1] = output

            # Teacher forcing
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            top1 = output.argmax(dim=1)
            dec_input = tgt[:, t] if use_teacher_forcing else top1

        return outputs

    @torch.no_grad()
    def translate(
        self,
        src: Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 100,
    ) -> Tensor:
        """Greedy decode a single source sequence.

        Args:
            src: Source token IDs, shape (1, src_len).
            bos_idx: Beginning-of-sequence token index.
            eos_idx: End-of-sequence token index.
            max_len: Maximum number of decoding steps.

        Returns:
            Tensor of predicted token IDs (excluding BOS), shape (output_len,).
        """
        self.eval()
        _, hidden = self.encoder(src)
        dec_input = torch.tensor([bos_idx], device=self.device)
        tokens = []

        for _ in range(max_len):
            output, hidden = self.decoder(dec_input, hidden, hidden)
            top1 = output.argmax(dim=1)
            tokens.append(top1.item())
            if top1.item() == eos_idx:
                break
            dec_input = top1

        return torch.tensor(tokens, device=self.device)
