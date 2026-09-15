
from __future__ import annotations

import re
from typing import Iterable, List


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"


SPECIAL_TOKENS = (
    PAD_TOKEN,
    UNK_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
)


WHITESPACE_RE = re.compile(r"\s+")


class WordTokenizer:
    """
    Simple whitespace-based tokenizer for parallel text data.

    This tokenizer intentionally preserves punctuation as part of the
    surrounding token. For example:

        "Hello, world!"

    becomes:

        ["Hello,", "world!"]
    """

    def __init__(
        self,
        lowercase: bool = False,
    ) -> None:
        self.lowercase = lowercase

    def normalize_text(self, text: str) -> str:
        """Normalize whitespace and optionally lowercase text."""

        if not isinstance(text, str):
            raise TypeError(
                f"Expected text to be str, got {type(text).__name__}."
            )

        normalized = WHITESPACE_RE.sub(" ", text).strip()

        if self.lowercase:
            normalized = normalized.lower()

        return normalized

    def tokenize(self, text: str) -> List[str]:
        """Convert a sentence into a list of word-level tokens."""

        normalized = self.normalize_text(text)

        if not normalized:
            return []

        return normalized.split(" ")

    def add_special_tokens(
        self,
        tokens: Iterable[str],
    ) -> List[str]:
        """Add sentence boundary tokens."""

        return [
            SOS_TOKEN,
            *list(tokens),
            EOS_TOKEN,
        ]

    def tokenize_with_special_tokens(
        self,
        text: str,
    ) -> List[str]:
        """Tokenize text and add SOS/EOS tokens."""

        tokens = self.tokenize(text)
        return self.add_special_tokens(tokens)

    def detokenize(
        self,
        tokens: Iterable[str],
    ) -> str:
        """Convert tokens back into a whitespace-separated sentence."""

        return " ".join(tokens).strip()

    def remove_special_tokens(
        self,
        tokens: Iterable[str],
    ) -> List[str]:
        """Remove SOS, EOS, and PAD tokens from a token sequence."""

        removable_tokens = {
            PAD_TOKEN,
            SOS_TOKEN,
            EOS_TOKEN,
        }

        return [
            token
            for token in tokens
            if token not in removable_tokens
        ]

    def detokenize_without_special_tokens(
        self,
        tokens: Iterable[str],
    ) -> str:
        """Detokenize after removing special tokens."""

        cleaned_tokens = self.remove_special_tokens(tokens)
        return self.detokenize(cleaned_tokens)

    def tokenize_batch(
        self,
        texts: Iterable[str],
    ) -> List[List[str]]:
        """Tokenize multiple sentences."""

        return [
            self.tokenize(text)
            for text in texts
        ]

    def tokenize_batch_with_special_tokens(
        self,
        texts: Iterable[str],
    ) -> List[List[str]]:
        """Tokenize multiple sentences and add SOS/EOS."""

        return [
            self.tokenize_with_special_tokens(text)
            for text in texts
        ]


def tokenize_text(
    text: str,
    lowercase: bool = False,
) -> List[str]:
    """Convenience function for tokenizing one sentence."""

    tokenizer = WordTokenizer(lowercase=lowercase)
    return tokenizer.tokenize(text)


if __name__ == "__main__":
    tokenizer = WordTokenizer()

    english_example = "I love learning AI."
    amharic_example = "እኔ አርቴፊሻል ኢንተለጀንስን መማር እወዳለሁ።"

    print("English tokens:")
    print(tokenizer.tokenize(english_example))

    print("\nAmharic tokens:")
    print(tokenizer.tokenize(amharic_example))

    print("\nEnglish with special tokens:")
    print(tokenizer.tokenize_with_special_tokens(english_example))

    print("\nAmharic with special tokens:")
    print(tokenizer.tokenize_with_special_tokens(amharic_example))