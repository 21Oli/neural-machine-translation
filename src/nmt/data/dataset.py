from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


PAD_ID = 0
UNK_ID = 1
SOS_ID = 2
EOS_ID = 3


class TranslationDataset(Dataset):
    def __init__(
        self,
        dataframe,
        tokenizer,
        en_vocab,
        am_vocab,
        max_src_len=64,
        max_tgt_len=64,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.en_vocab = en_vocab
        self.am_vocab = am_vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        src_tokens = self.tokenizer.tokenize(row["eng"])
        tgt_tokens = self.tokenizer.tokenize(row["amh"])

        src_ids = self.en_vocab.numericalize(
            src_tokens[: self.max_src_len - 2]
        )
        tgt_ids = self.am_vocab.numericalize(
            tgt_tokens[: self.max_tgt_len - 2]
        )

        src_ids = [SOS_ID] + src_ids + [EOS_ID]
        tgt_ids = [SOS_ID] + tgt_ids + [EOS_ID]

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=PAD_ID,
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=PAD_ID,
    )

    return src_batch, tgt_batch


def create_dataloader(
    dataset,
    batch_size=64,
    shuffle=False,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )