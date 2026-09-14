"""Unit tests for nmt.inference.preprocessing and nmt.inference.translator."""

from unittest.mock import MagicMock, patch

import pytest
import torch

from nmt.inference.preprocessing import InferencePreprocessor


# ---------------------------------------------------------------------------
# InferencePreprocessor
# ---------------------------------------------------------------------------


class TestInferencePreprocessor:
    def test_strips_html(self):
        pp = InferencePreprocessor(remove_html=True, remove_url=False)
        result = pp.process("<b>Hello</b> world")
        assert "<b>" not in result
        assert "Hello" in result

    def test_removes_url(self):
        pp = InferencePreprocessor(remove_html=False, remove_url=True)
        result = pp.process("Visit https://example.com today")
        assert "https" not in result

    def test_normalizes_whitespace(self):
        pp = InferencePreprocessor(remove_html=False, remove_url=False)
        result = pp.process("hello   world  ")
        assert result == "hello world"

    def test_process_batch(self):
        pp = InferencePreprocessor()
        texts = ["  hello  ", "<p>world</p>"]
        results = pp.process_batch(texts)
        assert len(results) == 2
        assert results[0] == "hello"
        assert "<p>" not in results[1]

    def test_empty_string(self):
        pp = InferencePreprocessor()
        assert pp.process("") == ""

    def test_no_modifications_when_flags_off(self):
        pp = InferencePreprocessor(remove_html=False, remove_url=False, lowercase=False)
        text = "Hello World"
        assert pp.process(text) == "Hello World"

    def test_lowercase_flag(self):
        pp = InferencePreprocessor(lowercase=True)
        result = pp.process("HELLO WORLD")
        assert result == "hello world"


# ---------------------------------------------------------------------------
# Translator (using mocked tokenizer and model)
# ---------------------------------------------------------------------------


class TestTranslator:
    def _make_translator(self, model_returns_tuple=False):
        """Build a Translator with fully mocked dependencies."""
        from nmt.inference.translator import Translator

        src_tok = MagicMock()
        src_tok.encode.return_value = [2, 10, 11, 12, 3]  # BOS, tokens, EOS
        src_tok.bos_id = 2
        src_tok.eos_id = 3

        tgt_tok = MagicMock()
        tgt_tok.decode.return_value = "translated output"
        tgt_tok.bos_id = 2
        tgt_tok.eos_id = 3

        model = MagicMock()
        if model_returns_tuple:
            attention = torch.rand(5, 5)
            model.translate.return_value = (torch.tensor([20, 21, 22, 3]), attention)
        else:
            model.translate.return_value = torch.tensor([20, 21, 22, 3])

        translator = Translator(
            model=model,
            src_tokenizer=src_tok,
            tgt_tokenizer=tgt_tok,
            device=torch.device("cpu"),
            max_len=50,
        )
        return translator, model, src_tok, tgt_tok

    def test_translate_returns_string(self):
        translator, _, _, _ = self._make_translator()
        result = translator.translate("Hello world")
        assert isinstance(result, str)

    def test_translate_calls_model(self):
        translator, model, _, _ = self._make_translator()
        translator.translate("Hello world")
        model.translate.assert_called_once()

    def test_translate_strips_eos(self):
        translator, _, _, tgt_tok = self._make_translator()
        translator.translate("Hello world")
        # Decoder call should NOT include the EOS token (index 3) in decoded ids
        called_ids = tgt_tok.decode.call_args[0][0]
        assert 3 not in called_ids

    def test_translate_with_attention_returns_tuple(self):
        translator, _, _, _ = self._make_translator(model_returns_tuple=True)
        translation, attn = translator.translate_with_attention("Hello world")
        assert isinstance(translation, str)
        assert attn is not None

    def test_translate_batch(self):
        translator, model, _, _ = self._make_translator()
        results = translator.translate_batch(["Hello", "World", "Foo"])
        assert len(results) == 3
        assert model.translate.call_count == 3

    def test_preprocessor_is_called(self):
        from nmt.inference.translator import Translator

        src_tok = MagicMock()
        src_tok.encode.return_value = [2, 10, 3]
        src_tok.bos_id = 2
        src_tok.eos_id = 3

        tgt_tok = MagicMock()
        tgt_tok.decode.return_value = "output"
        tgt_tok.bos_id = 2
        tgt_tok.eos_id = 3

        model = MagicMock()
        model.translate.return_value = torch.tensor([20, 3])

        preprocessor = MagicMock()
        preprocessor.process.return_value = "cleaned input"

        translator = Translator(
            model=model,
            src_tokenizer=src_tok,
            tgt_tokenizer=tgt_tok,
            preprocessor=preprocessor,
            device=torch.device("cpu"),
        )
        translator.translate("<b>raw input</b>")
        preprocessor.process.assert_called_once_with("<b>raw input</b>")
