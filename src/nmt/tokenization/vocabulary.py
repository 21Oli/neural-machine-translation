from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence

from .tokenizer import (
    PAD_TOKEN,
    UNK_TOKEN,
    SOS_TOKEN,
    EOS_TOKEN,
    SPECIAL_TOKENS,
)


class Vocabulary:
    """
    Token-to-integer vocabulary.

    Special-token IDs are fixed:

        <PAD> -> 0
        <UNK> -> 1
        <SOS> -> 2
        <EOS> -> 3

    Additional tokens are ordered deterministically by:
        1. Frequency, descending
        2. Token, ascending
    """

    def __init__(
        self,
        min_freq: int = 2,
        max_size: int | None = None,
    ) -> None:
        if min_freq < 1:
            raise ValueError("min_freq must be at least 1.")

        if max_size is not None and max_size < 1:
            raise ValueError("max_size must be at least 1 or None.")

        self.min_freq = min_freq
        self.max_size = max_size

        self.itos: List[str] = list(SPECIAL_TOKENS)
        self.stoi = {
            token: index
            for index, token in enumerate(self.itos)
        }

        self.token_counts: Counter[str] = Counter()

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self.itos)

    def __contains__(self, token: str) -> bool:
        """Return whether a token exists in the vocabulary."""
        return token in self.stoi

    def __repr__(self) -> str:
        return (
            f"Vocabulary(size={len(self)}, "
            f"min_freq={self.min_freq}, "
            f"max_size={self.max_size})"
        )

    def build(self, token_sequences: Iterable[Sequence[str]]) -> None:
        """
        Build the vocabulary from token sequences.

        This method should receive training-set tokens only.
        Special tokens are excluded from frequency counting because
        their IDs are predefined.
        """
        counter: Counter[str] = Counter()

        for sequence in token_sequences:
            for token in sequence:
                if token not in SPECIAL_TOKENS:
                    counter[token] += 1

        self.token_counts = counter

        candidate_tokens = [
            token
            for token, count in counter.items()
            if count >= self.min_freq
        ]

        candidate_tokens.sort(
            key=lambda token: (-counter[token], token)
        )

        if self.max_size is not None:
            available_token_slots = self.max_size - len(SPECIAL_TOKENS)

            if available_token_slots < 0:
                raise ValueError(
                    "max_size must be at least the number of special tokens."
                )

            candidate_tokens = candidate_tokens[:available_token_slots]

        self.itos = list(SPECIAL_TOKENS) + candidate_tokens
        self.stoi = {
            token: index
            for index, token in enumerate(self.itos)
        }

    def token_to_id(self, token: str) -> int:
        """Convert a token into an integer ID."""
        return self.stoi.get(token, self.stoi[UNK_TOKEN])

    def id_to_token(self, index: int) -> str:
        """Convert an integer ID into a token."""
        if index < 0 or index >= len(self.itos):
            raise IndexError(
                f"Token ID {index} is outside the vocabulary range."
            )

        return self.itos[index]

    def numericalize(
        self,
        tokens: Sequence[str],
        add_special_tokens: bool = False,
    ) -> List[int]:
        """
        Convert a token sequence into integer IDs.

        Unknown tokens receive the <UNK> ID.
        """
        token_list = list(tokens)

        if add_special_tokens:
            token_list = [
                SOS_TOKEN,
                *token_list,
                EOS_TOKEN,
            ]

        return [
            self.token_to_id(token)
            for token in token_list
        ]

    def denumericalize(
        self,
        ids: Sequence[int],
        remove_special_tokens: bool = False,
    ) -> List[str]:
        """Convert integer IDs back into tokens."""
        tokens = [
            self.id_to_token(index)
            for index in ids
        ]

        if remove_special_tokens:
            tokens = [
                token
                for token in tokens
                if token not in SPECIAL_TOKENS
            ]

        return tokens

    def token_frequency(self, token: str) -> int:
        """Return the training frequency of a token."""
        return self.token_counts.get(token, 0)

    def coverage(self, token_sequences: Iterable[Sequence[str]]) -> dict:
        """
        Calculate vocabulary coverage for a collection of token sequences.

        Returns:
            total_tokens
            known_tokens
            unknown_tokens
            coverage_percent
            unknown_percent
        """
        total_tokens = 0
        known_tokens = 0
        unknown_tokens = 0

        for sequence in token_sequences:
            for token in sequence:
                if token in SPECIAL_TOKENS:
                    continue

                total_tokens += 1

                if token in self.stoi:
                    known_tokens += 1
                else:
                    unknown_tokens += 1

        if total_tokens == 0:
            coverage_percent = 0.0
            unknown_percent = 0.0
        else:
            coverage_percent = (
                known_tokens / total_tokens
            ) * 100

            unknown_percent = (
                unknown_tokens / total_tokens
            ) * 100

        return {
            "total_tokens": total_tokens,
            "known_tokens": known_tokens,
            "unknown_tokens": unknown_tokens,
            "coverage_percent": coverage_percent,
            "unknown_percent": unknown_percent,
        }

    def save(self, path: str | Path) -> None:
        """Save the vocabulary as JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "min_freq": self.min_freq,
            "max_size": self.max_size,
            "itos": self.itos,
            "stoi": self.stoi,
            "token_counts": dict(self.token_counts),
        }

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        """Load a vocabulary from a JSON file."""
        input_path = Path(path)

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            payload = json.load(file)

        vocabulary = cls(
            min_freq=payload["min_freq"],
            max_size=payload["max_size"],
        )

        vocabulary.itos = payload["itos"]
        vocabulary.stoi = {
            token: int(index)
            for token, index in payload["stoi"].items()
        }
        vocabulary.token_counts = Counter(
            payload.get("token_counts", {})
        )

        return vocabulary


def build_vocabulary(
    token_sequences: Iterable[Sequence[str]],
    min_freq: int = 2,
    max_size: int | None = None,
) -> Vocabulary:
    """Convenience function for constructing a vocabulary."""
    vocabulary = Vocabulary(
        min_freq=min_freq,
        max_size=max_size,
    )
    vocabulary.build(token_sequences)
    return vocabulary


if __name__ == "__main__":
    example_sequences = [
        ["I", "love", "learning", "AI"],
        ["I", "love", "Python"],
        ["I", "love", "AI"],
    ]

    vocabulary = build_vocabulary(
        example_sequences,
        min_freq=1,
    )

    print(vocabulary)
    print(vocabulary.stoi)

    example_tokens = ["I", "love", "unknown_token"]
    example_ids = vocabulary.numericalize(
        example_tokens,
        add_special_tokens=True,
    )

    print("Tokens:", example_tokens)
    print("IDs:", example_ids)
    print(
        "Decoded:",
        vocabulary.denumericalize(
            example_ids,
            remove_special_tokens=True,
        ),
    )