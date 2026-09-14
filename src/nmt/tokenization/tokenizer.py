"""NMT tokenizer wrapper around SentencePiece."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class NMTTokenizer:
    """SentencePiece-based tokenizer for NMT.

    Supports training a new model from a corpus or loading a pre-trained one.

    Args:
        model_path: Path to a trained SentencePiece `.model` file.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._sp = None
        if model_path:
            self.load(model_path)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        corpus_files: List[str],
        model_prefix: str,
        vocab_size: int = 8000,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",
        pad_id: int = 0,
        unk_id: int = 1,
        bos_id: int = 2,
        eos_id: int = 3,
    ) -> None:
        """Train a SentencePiece model from a list of corpus files.

        Args:
            corpus_files: Paths to plain-text training files (one sentence per line).
            model_prefix: Output prefix; writes `<prefix>.model` and `<prefix>.vocab`.
            vocab_size: Target vocabulary size.
            character_coverage: Coverage for character-rich scripts (use ~0.9995 for
                Ethiopic scripts).
            model_type: SentencePiece model type: 'bpe' | 'unigram' | 'char' | 'word'.
            pad_id: Index reserved for <pad>.
            unk_id: Index reserved for <unk>.
            bos_id: Index reserved for <bos>.
            eos_id: Index reserved for <eos>.
        """
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError("Install sentencepiece: pip install sentencepiece") from e

        input_str = ",".join(corpus_files)
        logger.info(
            "Training SentencePiece model (vocab_size=%d, type=%s)...",
            vocab_size,
            model_type,
        )
        spm.SentencePieceTrainer.train(
            input=input_str,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            character_coverage=character_coverage,
            model_type=model_type,
            pad_id=pad_id,
            unk_id=unk_id,
            bos_id=bos_id,
            eos_id=eos_id,
        )
        self.load(f"{model_prefix}.model")
        logger.info("SentencePiece model saved to %s.model", model_prefix)

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """Encode a string to a list of subword IDs.

        Args:
            text: Input text.
            add_special_tokens: Whether to prepend BOS and append EOS.

        Returns:
            List of integer IDs.
        """
        self._check_loaded()
        ids: List[int] = self._sp.encode(text, out_type=int)
        if add_special_tokens:
            ids = [self._sp.bos_id()] + ids + [self._sp.eos_id()]
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decode a list of subword IDs back to a string.

        Args:
            ids: List of integer IDs.

        Returns:
            Decoded string.
        """
        self._check_loaded()
        return self._sp.decode(ids)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subword pieces (strings).

        Args:
            text: Input text.

        Returns:
            List of subword piece strings.
        """
        self._check_loaded()
        return self._sp.encode(text, out_type=str)

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def load(self, model_path: str) -> None:
        """Load a SentencePiece model from disk."""
        try:
            import sentencepiece as spm
        except ImportError as e:
            raise ImportError("Install sentencepiece: pip install sentencepiece") from e

        self._sp = spm.SentencePieceProcessor()
        self._sp.load(model_path)
        logger.info("Loaded SentencePiece model from %s", model_path)

    @property
    def vocab_size(self) -> int:
        self._check_loaded()
        return self._sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        self._check_loaded()
        return self._sp.pad_id()

    @property
    def unk_id(self) -> int:
        self._check_loaded()
        return self._sp.unk_id()

    @property
    def bos_id(self) -> int:
        self._check_loaded()
        return self._sp.bos_id()

    @property
    def eos_id(self) -> int:
        self._check_loaded()
        return self._sp.eos_id()

    def _check_loaded(self) -> None:
        if self._sp is None:
            raise RuntimeError("Tokenizer model not loaded. Call .train() or .load() first.")
