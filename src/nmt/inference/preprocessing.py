"""Preprocessing pipeline for inference-time inputs."""

import logging
from typing import List, Optional

from ..data.clean import normalize_whitespace, remove_html_tags, remove_urls
from ..data.normalize import normalize_text

logger = logging.getLogger(__name__)


class InferencePreprocessor:
    """Cleans and normalizes raw text before translation.

    Args:
        src_lang: Source language code (e.g. 'en', 'am').
        lowercase: Whether to lowercase the input.
        remove_html: Whether to strip HTML tags.
        remove_url: Whether to strip URLs.
        unicode_form: Unicode normalization form.
    """

    def __init__(
        self,
        src_lang: Optional[str] = None,
        lowercase: bool = False,
        remove_html: bool = True,
        remove_url: bool = True,
        unicode_form: str = "NFC",
    ) -> None:
        self.src_lang = src_lang
        self.lowercase = lowercase
        self.remove_html = remove_html
        self.remove_url = remove_url
        self.unicode_form = unicode_form

    def process(self, text: str) -> str:
        """Apply the full preprocessing pipeline to a single text string.

        Args:
            text: Raw input text.

        Returns:
            Preprocessed text ready for tokenization.
        """
        if self.remove_html:
            text = remove_html_tags(text)
        if self.remove_url:
            text = remove_urls(text)
        text = normalize_whitespace(text)
        text = normalize_text(
            text,
            lang=self.src_lang,
            unicode_form=self.unicode_form,
            lowercase=self.lowercase,
        )
        return text

    def process_batch(self, texts: List[str]) -> List[str]:
        """Apply preprocessing to a list of texts.

        Args:
            texts: List of raw input strings.

        Returns:
            List of preprocessed strings.
        """
        return [self.process(t) for t in texts]
