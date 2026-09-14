"""Vocabulary — maps tokens to integer indices and back."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class Vocabulary:
    """Token-to-index and index-to-token mapping.

    Special tokens are always placed at fixed indices:
        0 → <pad>
        1 → <unk>
        2 → <bos>
        3 → <eos>

    Args:
        pad_token: Padding token string.
        unk_token: Unknown token string.
        bos_token: Beginning-of-sequence token string.
        eos_token: End-of-sequence token string.
    """

    PAD_IDX = 0
    UNK_IDX = 1
    BOS_IDX = 2
    EOS_IDX = 3

    def __init__(
        self,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
    ) -> None:
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token

        self._token2idx: Dict[str, int] = {}
        self._idx2token: Dict[int, str] = {}

        # Reserve special token slots
        for idx, tok in enumerate([pad_token, unk_token, bos_token, eos_token]):
            self._token2idx[tok] = idx
            self._idx2token[idx] = tok

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build_from_tokens(self, token_lists: Iterable[Iterable[str]]) -> None:
        """Populate vocabulary from an iterable of token sequences.

        Args:
            token_lists: E.g. [[tok1, tok2, ...], [tok3, ...], ...]
        """
        from collections import Counter

        counter: Counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)

        for token, _ in counter.most_common():
            if token not in self._token2idx:
                idx = len(self._token2idx)
                self._token2idx[token] = idx
                self._idx2token[idx] = token

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def token2idx(self, token: str) -> int:
        return self._token2idx.get(token, self.UNK_IDX)

    def idx2token(self, idx: int) -> str:
        return self._idx2token.get(idx, self.unk_token)

    def encode(self, tokens: List[str], add_special_tokens: bool = True) -> List[int]:
        """Convert a token list to a list of indices.

        Args:
            tokens: List of string tokens.
            add_special_tokens: Wrap with <bos> and <eos>.

        Returns:
            List of integer indices.
        """
        ids = [self.token2idx(t) for t in tokens]
        if add_special_tokens:
            ids = [self.BOS_IDX] + ids + [self.EOS_IDX]
        return ids

    def decode(self, ids: List[int], remove_special_tokens: bool = True) -> List[str]:
        """Convert a list of indices back to tokens.

        Args:
            ids: List of integer indices.
            remove_special_tokens: Strip <bos>, <eos>, and <pad> tokens.

        Returns:
            List of string tokens.
        """
        special = {self.PAD_IDX, self.BOS_IDX, self.EOS_IDX}
        tokens = [self.idx2token(i) for i in ids]
        if remove_special_tokens:
            tokens = [t for i, t in zip(ids, tokens) if i not in special]
        return tokens

    def __len__(self) -> int:
        return len(self._token2idx)

    def __contains__(self, token: str) -> bool:
        return token in self._token2idx

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "token2idx": self._token2idx,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Vocabulary":
        vocab = cls(
            pad_token=data["pad_token"],
            unk_token=data["unk_token"],
            bos_token=data["bos_token"],
            eos_token=data["eos_token"],
        )
        for token, idx in data["token2idx"].items():
            vocab._token2idx[token] = idx
            vocab._idx2token[idx] = token
        return vocab

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)
