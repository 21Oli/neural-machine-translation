"""Tokenizer and vocabulary serialization helpers."""

import json
import logging
from pathlib import Path

from .tokenizer import WordTokenizer
from .vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def save_tokenizer(tokenizer: WordTokenizer, path: str) -> None:
    """Save a WordTokenizer config to a JSON file.

    Args:
        tokenizer: WordTokenizer instance to save.
        path: Destination file path (e.g. 'artifacts/tokenizers/config.json').
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {"lowercase": tokenizer.lowercase}
    out.write_text(json.dumps(config, indent=2), encoding="utf-8")
    logger.info("Tokenizer config saved to %s", out)


def load_tokenizer(path: str) -> WordTokenizer:
    """Load a WordTokenizer from a JSON config file.

    Args:
        path: Path to the JSON config file.

    Returns:
        Reconstructed WordTokenizer instance.
    """
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    tokenizer = WordTokenizer(lowercase=config.get("lowercase", False))
    logger.info("Tokenizer loaded from %s", path)
    return tokenizer


def save_vocabulary(vocab: Vocabulary, path: str) -> None:
    """Persist a Vocabulary to disk as JSON.

    Args:
        vocab: Vocabulary instance.
        path: File path (e.g. 'artifacts/vocabularies/src_vocab.json').
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    vocab.save(path)
    logger.info("Vocabulary (%d tokens) saved to %s", len(vocab), path)


def load_vocabulary(path: str) -> Vocabulary:
    """Load a Vocabulary from a JSON file.

    Args:
        path: Path to the vocabulary JSON file.

    Returns:
        Loaded Vocabulary instance.
    """
    vocab = Vocabulary.load(path)
    logger.info("Vocabulary (%d tokens) loaded from %s", len(vocab), path)
    return vocab
