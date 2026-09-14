"""Unit tests for nmt.data.dataset (NMTDataset) and nmt.data.split."""

import pytest
import torch

from nmt.data.dataset import NMTDataset, _pad_sequences
from nmt.data.split import split_dataset


# ---------------------------------------------------------------------------
# _pad_sequences helper
# ---------------------------------------------------------------------------


class TestPadSequences:
    def test_pads_to_max_length(self):
        seqs = ([1, 2, 3], [4, 5], [6])
        result = _pad_sequences(seqs, pad_idx=0)
        assert result.shape == (3, 3)

    def test_uses_correct_pad_value(self):
        seqs = ([1, 2], [3])
        result = _pad_sequences(seqs, pad_idx=99)
        assert result[1, 1].item() == 99

    def test_no_padding_needed(self):
        seqs = ([1, 2, 3], [4, 5, 6])
        result = _pad_sequences(seqs, pad_idx=0)
        assert result.shape == (2, 3)
        assert (result != 0).all()


# ---------------------------------------------------------------------------
# NMTDataset
# ---------------------------------------------------------------------------


class TestNMTDataset:
    def _make_dataset(self):
        src = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        tgt = [[10, 11], [12, 13, 14], [15]]
        return NMTDataset(src, tgt, src_pad_idx=0, tgt_pad_idx=0)

    def test_len(self):
        ds = self._make_dataset()
        assert len(ds) == 3

    def test_getitem_returns_pair(self):
        ds = self._make_dataset()
        src, tgt = ds[0]
        assert src == [1, 2, 3]
        assert tgt == [10, 11]

    def test_max_src_len_truncates(self):
        ds = NMTDataset([[1, 2, 3, 4, 5]], [[10, 11]], max_src_len=3)
        src, _ = ds[0]
        assert len(src) == 3

    def test_max_tgt_len_truncates(self):
        ds = NMTDataset([[1, 2]], [[10, 11, 12, 13]], max_tgt_len=2)
        _, tgt = ds[0]
        assert len(tgt) == 2

    def test_collate_fn_pads(self):
        ds = self._make_dataset()
        batch = [ds[0], ds[1]]
        src_tensor, tgt_tensor = ds.collate_fn(batch)
        assert src_tensor.shape[0] == 2
        assert tgt_tensor.shape[0] == 2
        # Both rows should be padded to the same width
        assert src_tensor.shape[1] == max(len(ds[0][0]), len(ds[1][0]))

    def test_collate_fn_returns_long_tensors(self):
        ds = self._make_dataset()
        src_tensor, tgt_tensor = ds.collate_fn([ds[0], ds[1]])
        assert src_tensor.dtype == torch.long
        assert tgt_tensor.dtype == torch.long

    def test_raises_on_mismatched_lengths(self):
        with pytest.raises(AssertionError):
            NMTDataset([[1, 2]], [[10, 11], [12, 13]])


# ---------------------------------------------------------------------------
# split_dataset
# ---------------------------------------------------------------------------


class TestSplitDataset:
    def _make_corpus(self, n=100):
        src = [f"src sentence {i}" for i in range(n)]
        tgt = [f"tgt sentence {i}" for i in range(n)]
        return src, tgt

    def test_split_sizes_sum_to_total(self):
        src, tgt = self._make_corpus(100)
        splits = split_dataset(src, tgt, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        total = sum(len(s) for s, _ in splits.values())
        assert total == 100

    def test_train_is_largest(self):
        src, tgt = self._make_corpus(100)
        splits = split_dataset(src, tgt)
        train_len = len(splits["train"][0])
        val_len = len(splits["val"][0])
        test_len = len(splits["test"][0])
        assert train_len > val_len
        assert train_len > test_len

    def test_no_overlap_between_splits(self):
        src, tgt = self._make_corpus(100)
        splits = split_dataset(src, tgt, shuffle=False)
        train_set = set(splits["train"][0])
        val_set = set(splits["val"][0])
        test_set = set(splits["test"][0])
        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_shuffle_reproducible_with_seed(self):
        src, tgt = self._make_corpus(100)
        splits1 = split_dataset(src, tgt, shuffle=True, seed=42)
        splits2 = split_dataset(src, tgt, shuffle=True, seed=42)
        assert splits1["train"][0] == splits2["train"][0]

    def test_different_seeds_give_different_order(self):
        src, tgt = self._make_corpus(100)
        splits1 = split_dataset(src, tgt, shuffle=True, seed=1)
        splits2 = split_dataset(src, tgt, shuffle=True, seed=99)
        assert splits1["train"][0] != splits2["train"][0]

    def test_ratios_must_sum_to_one(self):
        src, tgt = self._make_corpus(10)
        with pytest.raises(AssertionError):
            split_dataset(src, tgt, train_ratio=0.6, val_ratio=0.2, test_ratio=0.1)
