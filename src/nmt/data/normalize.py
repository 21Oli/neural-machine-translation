"""
Text normalization utilities for the English-Amharic NMT corpus.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: object) -> str:
    """Apply conservative Unicode, markup, and whitespace normalization."""

    if text is None:
        return ""

    text = str(text)

    # Unicode canonical normalization.
    text = unicodedata.normalize("NFC", text)

    # Normalize common escaped formatting artifacts.
    text = text.replace("\\_", "_")
    text = text.replace("\\-", "-")

    # Remove obvious duplicated angle-bracket wrappers.
    # Example:
    # << text >>  -> text
    text = re.sub(r"^\s*<\s*<\s*(.*?)\s*>\s*>\s*$", r"\1", text)

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def is_empty(text: str) -> bool:
    """Return True when text is empty after normalization."""
    return not text.strip()