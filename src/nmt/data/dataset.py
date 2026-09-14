"""PyTorch Dataset for parallel NMT corpora."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class NMTDataset(Dataset):
    """Parallel corpus dataset that returns (src_ids, tgt_ids) tensors.

    Args:
        src_sentences: Tokenized and encoded source sentences (list of id lists).
        tgt_sentences: Tokenized and encoded target sentences (list of id lists).
        src_pad_idx: Padding index for source vocabulary.
        tgt_pad_idx: Padding index for target vocabulary.
        max_src_len: Truncate source sequences to this length.
        max_tgt_len: Truncate target sequences to this length.
    """

    def __init__(
        self,
        src_sentences: List[List[int]],
        tgt_sentences: List[List[int]],
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
        max_src_len: Optional[int] = None,
        max_tgt_len: Optional[int] = None,
    ) -> None:
        assert len(src_sentences) == len(tgt_sentences)
        self.src = src_sentences
        self.tgt = tgt_sentences
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        src = self.src[idx]
        tgt = self.tgt[idx]
        if self.max_src_len:
            src = src[: self.max_src_len]
        if self.max_tgt_len:
            tgt = tgt[: self.max_tgt_len]
        return src, tgt

    def collate_fn(
        self, batch: List[Tuple[List[int], List[int]]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad a batch of (src, tgt) pairs to the same length.

        Args:
            batch: List of (src_ids, tgt_ids) tuples.

        Returns:
            Padded (src_tensor, tgt_tensor) of shape (batch_size, max_len).
        """
        src_batch, tgt_batch = zip(*batch)

        src_padded = _pad_sequences(src_batch, self.src_pad_idx)
        tgt_padded = _pad_sequences(tgt_batch, self.tgt_pad_idx)

        return src_padded, tgt_padded

    def get_dataloader(
        self,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """Convenience method to create a DataLoader for this dataset."""
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )


def _pad_sequences(sequences: Tuple[List[int], ...], pad_idx: int) -> torch.Tensor:
    """Pad sequences to the same length and return as a tensor."""
    max_len = max(len(s) for s in sequences)
    padded = [s + [pad_idx] * (max_len - len(s)) for s in sequences]
    return torch.tensor(padded, dtype=torch.long)
