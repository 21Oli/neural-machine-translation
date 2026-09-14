"""Tokenizer and vocabulary serialization helpers."""

import json
import logging
from pathlib import Path
from typing import Tuple

from .tokenizer import NMTTokenizer
from .vocabulary import Vocabulary

logger = logging.getLogger(__name__)


def save_tokenizer(tokenizer: NMTTokenizer, output_dir: str, name: str = "tokenizer") -> None:
    """Save a tokenizer's metadata to a directory.

    The underlying SentencePiece model file is expected to already exist
    (created by NMTTokenizer.train()).  This function saves a JSON manifest
    pointing to it so it can be re-loaded without extra arguments.

    Args:
        tokenizer: Trained NMTTokenizer instance.
        output_dir: Directory to write the manifest into.
        name: Basename for the manifest file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "vocab_size": tokenizer.vocab_size,
        "pad_id": tokenizer.pad_id,
        "unk_id": tokenizer.unk_id,
        "bos_id": tokenizer.bos_id,
        "eos_id": tokenizer.eos_id,
    }
    manifest_path = out / f"{name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Tokenizer manifest saved to %s", manifest_path)


def load_tokenizer(model_path: str) -> NMTTokenizer:
    """Load a SentencePiece-based tokenizer from a .model file.

    Args:
        model_path: Path to the .model file.

    Returns:
        Loaded NMTTokenizer instance.
    """
    tokenizer = NMTTokenizer(model_path=model_path)
    logger.info("Tokenizer loaded from %s", model_path)
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
