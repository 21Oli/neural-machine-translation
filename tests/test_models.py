"""Unit tests for Encoder, Decoder, Seq2Seq, and AttentionSeq2Seq."""

import pytest
import torch

from nmt.models.encoder import Encoder
from nmt.models.decoder import Decoder
from nmt.models.seq2seq import Seq2Seq
from nmt.models.attention_seq2seq import (
    AttentionDecoder,
    AttentionSeq2Seq,
    BahdanauAttention,
)

# Small dimensions for fast CPU tests
VOCAB_SIZE = 50
EMBED_DIM = 16
HIDDEN_DIM = 32
NUM_LAYERS = 2
BATCH = 4
SRC_LEN = 10
TGT_LEN = 8
DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class TestEncoder:
    def _make_encoder(self, bidirectional=True):
        return Encoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=0.0,
            bidirectional=bidirectional,
        )

    def test_output_shape_bidirectional(self):
        enc = self._make_encoder(bidirectional=True)
        src = torch.randint(0, VOCAB_SIZE, (BATCH, SRC_LEN))
        outputs, hidden = enc(src)
        # Bidirectional: outputs shape (batch, src_len, hidden_dim * 2)
        assert outputs.shape == (BATCH, SRC_LEN, HIDDEN_DIM * 2)
        # Hidden is projected back to hidden_dim
        assert hidden.shape == (NUM_LAYERS, BATCH, HIDDEN_DIM)

    def test_output_shape_unidirectional(self):
        enc = self._make_encoder(bidirectional=False)
        src = torch.randint(0, VOCAB_SIZE, (BATCH, SRC_LEN))
        outputs, hidden = enc(src)
        assert outputs.shape == (BATCH, SRC_LEN, HIDDEN_DIM)
        assert hidden.shape == (NUM_LAYERS, BATCH, HIDDEN_DIM)

    def test_no_nan_in_outputs(self):
        enc = self._make_encoder()
        src = torch.randint(0, VOCAB_SIZE, (BATCH, SRC_LEN))
        outputs, hidden = enc(src)
        assert not torch.isnan(outputs).any()
        assert not torch.isnan(hidden).any()


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class TestDecoder:
    def _make_decoder(self):
        return Decoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            dropout=0.0,
        )

    def test_output_shape(self):
        dec = self._make_decoder()
        token = torch.randint(0, VOCAB_SIZE, (BATCH,))
        hidden = torch.zeros(NUM_LAYERS, BATCH, HIDDEN_DIM)
        pred, new_hidden = dec(token, hidden, hidden)
        assert pred.shape == (BATCH, VOCAB_SIZE)
        assert new_hidden.shape == (NUM_LAYERS, BATCH, HIDDEN_DIM)

    def test_no_nan_in_outputs(self):
        dec = self._make_decoder()
        token = torch.randint(0, VOCAB_SIZE, (BATCH,))
        hidden = torch.zeros(NUM_LAYERS, BATCH, HIDDEN_DIM)
        pred, _ = dec(token, hidden, hidden)
        assert not torch.isnan(pred).any()


# ---------------------------------------------------------------------------
# Seq2Seq
# ---------------------------------------------------------------------------


class TestSeq2Seq:
    def _make_model(self):
        enc = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=0.0)
        dec = Decoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=0.0)
        return Seq2Seq(enc, dec, device=DEVICE)

    def test_forward_output_shape(self):
        model = self._make_model()
        src = torch.randint(0, VOCAB_SIZE, (BATCH, SRC_LEN))
        tgt = torch.randint(0, VOCAB_SIZE, (BATCH, TGT_LEN))
        output = model(src, tgt, teacher_forcing_ratio=0.5)
        assert output.shape == (BATCH, TGT_LEN - 1, VOCAB_SIZE)

    def test_forward_no_teacher_forcing(self):
        model = self._make_model()
        src = torch.randint(0, VOCAB_SIZE, (BATCH, SRC_LEN))
        tgt = torch.randint(0, VOCAB_SIZE, (BATCH, TGT_LEN))
        output = model(src, tgt, teacher_forcing_ratio=0.0)
        assert output.shape == (BATCH, TGT_LEN - 1, VOCAB_SIZE)

    def test_translate_returns_tensor(self):
        model = self._make_model()
        src = torch.randint(0, VOCAB_SIZE, (1, SRC_LEN))
        result = model.translate(src, bos_idx=2, eos_idx=3, max_len=20)
        assert isinstance(result, torch.Tensor)
        assert result.ndim == 1


# ---------------------------------------------------------------------------
# BahdanauAttention
# ---------------------------------------------------------------------------


class TestBahdanauAttention:
    def test_attention_weights_sum_to_one(self):
        attn = BahdanauAttention(
            hidden_dim=HIDDEN_DIM,
            encoder_dim=HIDDEN_DIM * 2,
            attention_dim=16,
        )
        hidden = torch.randn(BATCH, HIDDEN_DIM)
        enc_outputs = torch.randn(BATCH, SRC_LEN, HIDDEN_DIM * 2)
        context, weights = attn(hidden, enc_outputs)
        weight_sums = weights.sum(dim=1)
        assert torch.allclose(weight_sums, torch.ones(BATCH), atol=1e-5)

    def test_context_shape(self):
        attn = BahdanauAttention(HIDDEN_DIM, HIDDEN_DIM * 2, 16)
        hidden = torch.randn(BATCH, HIDDEN_DIM)
        enc_outputs = torch.randn(BATCH, SRC_LEN, HIDDEN_DIM * 2)
        context, _ = attn(hidden, enc_outputs)
        assert context.shape == (BATCH, HIDDEN_DIM * 2)


# ---------------------------------------------------------------------------
# AttentionSeq2Seq
# ---------------------------------------------------------------------------


class TestAttentionSeq2Seq:
    def _make_model(self):
        encoder_dim = HIDDEN_DIM * 2  # bidirectional
        enc = Encoder(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, dropout=0.0, bidirectional=True)
        dec = AttentionDecoder(
            vocab_size=VOCAB_SIZE,
            embed_dim=EMBED_DIM,
            hidden_dim=HIDDEN_DIM,
            encoder_dim=encoder_dim,
            attention_dim=16,
            num_layers=NUM_LAYERS,
            dropout=0.0,
        )
        return AttentionSeq2Seq(enc, dec, device=DEVICE)

    def test_forward_output_shapes(self):
        model = self._make_model()
        src = torch.randint(0, VOCAB_SIZE, (BATCH, SRC_LEN))
        tgt = torch.randint(0, VOCAB_SIZE, (BATCH, TGT_LEN))
        outputs, attentions = model(src, tgt)
        assert outputs.shape == (BATCH, TGT_LEN - 1, VOCAB_SIZE)
        assert attentions.shape == (BATCH, TGT_LEN - 1, SRC_LEN)

    def test_attention_weights_sum_to_one(self):
        model = self._make_model()
        src = torch.randint(0, VOCAB_SIZE, (BATCH, SRC_LEN))
        tgt = torch.randint(0, VOCAB_SIZE, (BATCH, TGT_LEN))
        _, attentions = model(src, tgt)
        weight_sums = attentions.sum(dim=2)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)

    def test_translate_returns_tokens_and_attention(self):
        model = self._make_model()
        src = torch.randint(0, VOCAB_SIZE, (1, SRC_LEN))
        tokens, attn = model.translate(src, bos_idx=2, eos_idx=3, max_len=10)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.ndim == 1
