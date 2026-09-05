"""Memory-mapped, little-endian uint32 tokens for packed causal language modeling."""

import mmap
from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset


class TokenBlocks(Dataset):
    def __init__(self, path, sequence_length):
        if sys.byteorder != 'little' or type(sequence_length) is not int or sequence_length < 1:
            raise ValueError('use a little-endian host and a positive sequence length')
        self.path = Path(path)
        size = self.path.stat().st_size
        if size % 4 or size // 4 <= sequence_length:
            raise ValueError('token file must contain complete uint32 tokens and at least one target block')
        self.sequence_length = sequence_length
        self.blocks = (size // 4 - 1) // sequence_length
        self.file = self.path.open('rb')
        self.mapping = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_COPY)

    def __len__(self):
        return self.blocks

    def __getitem__(self, index):
        if not 0 <= index < self.blocks:
            raise IndexError(index)
        tokens = torch.frombuffer(self.mapping, dtype=torch.int32, count=self.sequence_length + 1,
                                  offset=index * self.sequence_length * 4).to(torch.long)
        return tokens[:-1], tokens[1:]

    def close(self):
        self.mapping.close()
        self.file.close()
