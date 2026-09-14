"""Corpus cleaning utilities — removes noise, duplicates, and malformed pairs."""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def clean_corpus(
    src_sentences: List[str],
    tgt_sentences: List[str],
    min_len: int = 2,
    max_len: int = 100,
    remove_duplicates: bool = True,
) -> Tuple[List[str], List[str]]:
    """Clean a parallel corpus by filtering bad sentence pairs.

    Args:
        src_sentences: Source language sentences.
        tgt_sentences: Target language sentences.
        min_len: Minimum word count (inclusive).
        max_len: Maximum word count (inclusive).
        remove_duplicates: Whether to deduplicate sentence pairs.

    Returns:
        Tuple of (cleaned_src, cleaned_tgt).
    """
    assert len(src_sentences) == len(tgt_sentences)

    original_count = len(src_sentences)
    pairs = list(zip(src_sentences, tgt_sentences))

    # Remove empty lines
    pairs = [(s, t) for s, t in pairs if s.strip() and t.strip()]

    # Filter by length
    pairs = [
        (s, t)
        for s, t in pairs
        if min_len <= len(s.split()) <= max_len
        and min_len <= len(t.split()) <= max_len
    ]

    # Remove duplicates
    if remove_duplicates:
        seen = set()
        deduped = []
        for pair in pairs:
            if pair not in seen:
                seen.add(pair)
                deduped.append(pair)
        pairs = deduped

    src_clean, tgt_clean = zip(*pairs) if pairs else ([], [])
    src_clean, tgt_clean = list(src_clean), list(tgt_clean)

    logger.info(
        "Cleaning: %d → %d pairs (removed %d)",
        original_count,
        len(src_clean),
        original_count - len(src_clean),
    )
    return src_clean, tgt_clean


def remove_html_tags(text: str) -> str:
    """Strip HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text)


def remove_urls(text: str) -> str:
    """Remove URLs from a string."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into a single space."""
    return re.sub(r"\s+", " ", text).strip()
