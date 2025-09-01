from __future__ import annotations
from typing import Iterator, Tuple
import numpy as np


class CharDataset:
    def __init__(self, tokens: np.ndarray, block_size: int):
        assert tokens.ndim == 1
        self.tokens = tokens.astype(np.int64)
        self.block_size = int(block_size)

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        x = self.tokens[idx:idx + self.block_size]
        y = self.tokens[idx + 1:idx + 1 + self.block_size]
        return x, y


class NumpyDataLoader:
    def __init__(self, dataset: CharDataset, batch_size: int, shuffle: bool, drop_last: bool):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        idxs = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idxs)
        B = self.batch_size
        for start in range(0, len(idxs), B):
            end = start + B
            if end > len(idxs) and self.drop_last:
                break
            batch_idx = idxs[start:end]
            if len(batch_idx) == 0:
                continue
            xs, ys = zip(*(self.dataset[i] for i in batch_idx))
            yield np.stack(xs, axis=0), np.stack(ys, axis=0)

    def __len__(self):
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def create_dataloaders(tokens: np.ndarray, block_size: int, batch_size: int, split: float = 0.9) -> Tuple[NumpyDataLoader, NumpyDataLoader]:
    n = len(tokens)
    n_train = int(n * split)
    train_tokens = tokens[:n_train]
    val_tokens = tokens[n_train:]
    train_ds = CharDataset(train_tokens, block_size)
    val_ds = CharDataset(val_tokens, block_size)
    train_loader = NumpyDataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = NumpyDataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader
