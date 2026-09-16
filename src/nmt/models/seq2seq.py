"""Baseline Seq2Seq model (encoder + decoder, no attention)."""

import torch
import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        device,
    ):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        src,
        trg,
        teacher_forcing_ratio=0.5,
    ):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(
            batch_size,
            trg_len,
            trg_vocab_size,
            device=self.device,
        )

        hidden, cell = self.encoder(src)

        input_token = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(
                input_token,
                hidden,
                cell,
            )

            outputs[:, t, :] = output

            teacher_force = (
                torch.rand(1).item()
                < teacher_forcing_ratio
            )

            top1 = output.argmax(1)

            input_token = (
                trg[:, t]
                if teacher_force
                else top1
            )

        return outputs