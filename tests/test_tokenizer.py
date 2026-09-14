"""Unit tests for nmt.tokenization.vocabulary and nmt.tokenization.tokenizer."""

import json
import os
import tempfile

import pytest

from nmt.tokenization.vocabulary import Vocabulary


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


class TestVocabularySpecialTokens:
    def setup_method(self):
        self.vocab = Vocabulary()

    def test_pad_is_zero(self):
        assert self.vocab.token2idx("<pad>") == Vocabulary.PAD_IDX

    def test_unk_is_one(self):
        assert self.vocab.token2idx("<unk>") == Vocabulary.UNK_IDX

    def test_bos_is_two(self):
        assert self.vocab.token2idx("<bos>") == Vocabulary.BOS_IDX

    def test_eos_is_three(self):
        assert self.vocab.token2idx("<eos>") == Vocabulary.EOS_IDX

    def test_initial_length_is_four(self):
        assert len(self.vocab) == 4


class TestVocabularyBuild:
    def test_build_from_tokens_adds_new_tokens(self):
        vocab = Vocabulary()
        vocab.build_from_tokens([["hello", "world"], ["hello", "foo"]])
        assert "hello" in vocab
        assert "world" in vocab
        assert "foo" in vocab

    def test_build_does_not_override_special_tokens(self):
        vocab = Vocabulary()
        vocab.build_from_tokens([["<pad>", "<unk>"]])
        assert vocab.token2idx("<pad>") == Vocabulary.PAD_IDX
        assert vocab.token2idx("<unk>") == Vocabulary.UNK_IDX

    def test_unknown_token_returns_unk_idx(self):
        vocab = Vocabulary()
        assert vocab.token2idx("nonexistent") == Vocabulary.UNK_IDX


class TestVocabularyEncodeDecode:
    def setup_method(self):
        self.vocab = Vocabulary()
        self.vocab.build_from_tokens([["hello", "world", "foo"]])

    def test_encode_adds_bos_eos(self):
        ids = self.vocab.encode(["hello", "world"], add_special_tokens=True)
        assert ids[0] == Vocabulary.BOS_IDX
        assert ids[-1] == Vocabulary.EOS_IDX

    def test_encode_without_special_tokens(self):
        ids = self.vocab.encode(["hello", "world"], add_special_tokens=False)
        assert ids[0] != Vocabulary.BOS_IDX
        assert ids[-1] != Vocabulary.EOS_IDX

    def test_decode_removes_special_tokens(self):
        ids = self.vocab.encode(["hello", "world"], add_special_tokens=True)
        tokens = self.vocab.decode(ids, remove_special_tokens=True)
        assert "<bos>" not in tokens
        assert "<eos>" not in tokens
        assert "hello" in tokens

    def test_roundtrip(self):
        original = ["hello", "world", "foo"]
        ids = self.vocab.encode(original, add_special_tokens=False)
        decoded = self.vocab.decode(ids, remove_special_tokens=False)
        assert decoded == original


class TestVocabularySerialization:
    def test_save_and_load(self):
        vocab = Vocabulary()
        vocab.build_from_tokens([["hello", "world", "foo"]])

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            path = f.name

        try:
            vocab.save(path)
            loaded = Vocabulary.load(path)
            assert len(loaded) == len(vocab)
            assert loaded.token2idx("hello") == vocab.token2idx("hello")
            assert loaded.token2idx("<pad>") == Vocabulary.PAD_IDX
        finally:
            os.unlink(path)

    def test_to_dict_from_dict_roundtrip(self):
        vocab = Vocabulary()
        vocab.build_from_tokens([["a", "b", "c"]])
        d = vocab.to_dict()
        restored = Vocabulary.from_dict(d)
        assert len(restored) == len(vocab)
        assert restored.token2idx("a") == vocab.token2idx("a")


# ---------------------------------------------------------------------------
# NMTTokenizer — only tests that do NOT require SentencePiece model on disk
# ---------------------------------------------------------------------------


class TestNMTTokenizerUnloaded:
    def test_raises_if_not_loaded(self):
        from nmt.tokenization.tokenizer import NMTTokenizer

        tok = NMTTokenizer()
        with pytest.raises(RuntimeError, match="not loaded"):
            tok.encode("hello")

    def test_raises_vocab_size_if_not_loaded(self):
        from nmt.tokenization.tokenizer import NMTTokenizer

        tok = NMTTokenizer()
        with pytest.raises(RuntimeError):
            _ = tok.vocab_size
