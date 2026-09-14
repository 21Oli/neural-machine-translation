"""High-level translation interface."""

import logging
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..tokenization.tokenizer import NMTTokenizer
from .preprocessing import InferencePreprocessor

logger = logging.getLogger(__name__)


class Translator:
    """End-to-end translation pipeline.

    Wraps a trained model with tokenizer and preprocessor for easy inference.

    Args:
        model: Trained Seq2Seq or AttentionSeq2Seq model.
        src_tokenizer: Source language tokenizer.
        tgt_tokenizer: Target language tokenizer.
        device: Torch device to run inference on.
        preprocessor: Optional InferencePreprocessor for input cleaning.
        max_len: Maximum number of output tokens.
    """

    def __init__(
        self,
        model: nn.Module,
        src_tokenizer: NMTTokenizer,
        tgt_tokenizer: NMTTokenizer,
        device: Optional[torch.device] = None,
        preprocessor: Optional[InferencePreprocessor] = None,
        max_len: int = 100,
    ) -> None:
        self.model = model
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.device = device or torch.device("cpu")
        self.preprocessor = preprocessor
        self.max_len = max_len
        self.model.eval()

    def translate(self, text: str) -> str:
        """Translate a single input string.

        Args:
            text: Raw source language text.

        Returns:
            Translated target language string.
        """
        if self.preprocessor:
            text = self.preprocessor.process(text)

        src_ids = self.src_tokenizer.encode(text, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        bos_id = self.tgt_tokenizer.bos_id
        eos_id = self.tgt_tokenizer.eos_id

        result = self.model.translate(src_tensor, bos_id, eos_id, self.max_len)
        token_ids = result[0] if isinstance(result, tuple) else result

        # Strip EOS if present
        ids = token_ids.tolist()
        if ids and ids[-1] == eos_id:
            ids = ids[:-1]

        return self.tgt_tokenizer.decode(ids)

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate a list of strings one by one.

        Args:
            texts: List of source language strings.

        Returns:
            List of translated strings.
        """
        return [self.translate(t) for t in texts]

    def translate_with_attention(
        self, text: str
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """Translate and return attention weights (for attention models).

        Args:
            text: Raw source language text.

        Returns:
            Tuple of (translation string, attention tensor or None).
            Attention tensor has shape (output_len, src_len).
        """
        if self.preprocessor:
            text = self.preprocessor.process(text)

        src_ids = self.src_tokenizer.encode(text, add_special_tokens=True)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)

        bos_id = self.tgt_tokenizer.bos_id
        eos_id = self.tgt_tokenizer.eos_id

        result = self.model.translate(src_tensor, bos_id, eos_id, self.max_len)

        if isinstance(result, tuple):
            token_ids, attentions = result
        else:
            token_ids = result
            attentions = None

        ids = token_ids.tolist()
        if ids and ids[-1] == eos_id:
            ids = ids[:-1]

        translation = self.tgt_tokenizer.decode(ids)
        return translation, attentions
