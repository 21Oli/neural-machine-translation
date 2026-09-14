"""Train / validation / test splitting utilities."""

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def split_dataset(
    src_sentences: List[str],
    tgt_sentences: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Split a parallel corpus into train / val / test sets.

    Args:
        src_sentences: Source language sentences.
        tgt_sentences: Target language sentences.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for testing.
        shuffle: Whether to shuffle before splitting.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys 'train', 'val', 'test', each mapping to
        a (src_list, tgt_list) tuple.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )
    assert len(src_sentences) == len(tgt_sentences)

    pairs = list(zip(src_sentences, tgt_sentences))

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(pairs)

    n = len(pairs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }

    result = {}
    for name, split_pairs in splits.items():
        src, tgt = zip(*split_pairs) if split_pairs else ([], [])
        result[name] = (list(src), list(tgt))
        logger.info("Split '%s': %d pairs", name, len(split_pairs))

    return result


def save_splits(
    splits: Dict[str, Tuple[List[str], List[str]]],
    output_dir: str,
    src_lang: str = "src",
    tgt_lang: str = "tgt",
) -> None:
    """Write split files to disk in plain-text format.

    Args:
        splits: Output of split_dataset().
        output_dir: Directory to write files into.
        src_lang: Source language code (used in filenames).
        tgt_lang: Target language code (used in filenames).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name, (src_sents, tgt_sents) in splits.items():
        src_file = out / f"{split_name}.{src_lang}"
        tgt_file = out / f"{split_name}.{tgt_lang}"

        src_file.write_text("\n".join(src_sents), encoding="utf-8")
        tgt_file.write_text("\n".join(tgt_sents), encoding="utf-8")
        logger.info("Saved %s → %s, %s", split_name, src_file, tgt_file)
