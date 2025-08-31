from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CharDataset(Dataset):
    def __init__(self, tokens: np.ndarray, block_size: int):
        assert tokens.ndim == 1
        self.tokens = tokens.astype(np.int64)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.tokens[idx:idx + self.block_size])
        y = torch.from_numpy(self.tokens[idx + 1:idx + 1 + self.block_size])
        return x, y

def create_dataloaders(tokens: np.ndarray, block_size: int, batch_size: int, split: float = 0.9) -> Tuple[DataLoader, DataLoader]:
    n = len(tokens)
    n_train = int(n * split)
    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train:]
    train_ds = CharDataset(train_tokens, block_size)
    val_ds = CharDataset(val_tokens, block_size)
    common = dict(num_workers=4, pin_memory=True, persistent_workers=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, **common)
    return train_loader, val_loader
