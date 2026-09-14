"""Text normalization — Unicode, punctuation, and script-specific fixes."""

import logging
import unicodedata
from typing import Optional

logger = logging.getLogger(__name__)


def normalize_text(
    text: str,
    lang: Optional[str] = None,
    unicode_form: str = "NFC",
    lowercase: bool = False,
) -> str:
    """Normalize a text string.

    Args:
        text: Input text.
        lang: Optional language code for language-specific normalization.
        unicode_form: Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD').
        lowercase: Whether to lowercase the text.

    Returns:
        Normalized text string.
    """
    # Unicode normalization
    text = unicodedata.normalize(unicode_form, text)

    # Lowercase (typically applied only to source/English side)
    if lowercase:
        text = text.lower()

    # Language-specific normalization
    if lang == "am":
        text = normalize_amharic(text)
    elif lang == "en":
        text = normalize_english(text)

    return text.strip()


def normalize_amharic(text: str) -> str:
    """Apply Amharic-specific normalization rules.

    Handles variant Ethiopic characters that are phonetically identical
    but encoded differently.

    Args:
        text: Raw Amharic text.

    Returns:
        Normalized Amharic text.
    """
    # Map visually/phonetically identical Ethiopic characters to a canonical form
    # ሃ → ሀ  (ha variants)
    replacements = {
        "\u1203": "\u1200",  # ሃ → ሀ
        "\u1233": "\u1230",  # ሳ variant
        "\u1273": "\u1270",  # ቃ variant
        "\u12D3": "\u12D0",  # ዓ → ዐ
    }
    for variant, canonical in replacements.items():
        text = text.replace(variant, canonical)
    return text


def normalize_english(text: str) -> str:
    """Apply English-specific normalization rules.

    Args:
        text: Raw English text.

    Returns:
        Normalized English text.
    """
    # Normalize curly quotes to straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201C", '"').replace("\u201D", '"')
    # Normalize dashes
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text
