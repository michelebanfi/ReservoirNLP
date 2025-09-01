from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from typing import Iterable, Tuple


def _batch_to_onehot_lists(batch_x, batch_y, vocab_size: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert a batch of token ids [B,T] to lists of one-hot sequences [(T,V)]."""
    # Accept torch.Tensor or numpy arrays
    x_np = batch_x.detach().cpu().numpy() if hasattr(batch_x, "detach") else np.asarray(batch_x)
    y_np = batch_y.detach().cpu().numpy() if hasattr(batch_y, "detach") else np.asarray(batch_y)
    B, T = x_np.shape
    eye = np.eye(vocab_size, dtype=np.float32)
    X_list, Y_list = [], []
    for b in range(B):
        X_list.append(eye[x_np[b]])  # (T,V)
        Y_list.append(eye[y_np[b]])  # (T,V)
    return X_list, Y_list


def _dataset_to_lists(loader: Iterable[Tuple[np.ndarray, np.ndarray]], vocab_size: int, max_batches: int | None = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    X_all: List[np.ndarray] = []
    Y_all: List[np.ndarray] = []
    for i, (xb, yb) in enumerate(loader):
        Xb, Yb = _batch_to_onehot_lists(xb, yb, vocab_size)
        X_all.extend(Xb)
        Y_all.extend(Yb)
        if max_batches is not None and (i + 1) >= max_batches:
            break
    return X_all, Y_all


def train(model, train_loader: Iterable[Tuple[np.ndarray, np.ndarray]], val_loader: Iterable[Tuple[np.ndarray, np.ndarray]], vocab_size: int,
          epochs: int = 1, warmup: int = 0, eval_batches: int = 10) -> Dict:
    """Offline training using ReservoirPy's fit().

    - model: ReservoirPy node/model supporting fit and run
    - train_loader/val_loader: yield batches of token ids [B,T]
    - vocab_size: size of one-hot vectors
    """
    history = {"train_mse": [], "val_mse": []}
    for ep in range(epochs):
        X_train, Y_train = _dataset_to_lists(train_loader, vocab_size)
        model.fit(X_train, Y_train, warmup=warmup)
        # quick eval on a subset
        val = evaluate(model, val_loader, vocab_size, max_batches=eval_batches)
        history["val_mse"].append(val)
        history["train_mse"].append(float('nan'))
    return history


def evaluate(model, loader: Iterable[Tuple[np.ndarray, np.ndarray]], vocab_size: int, max_batches: int | None = 10) -> float:
    X_val, Y_val = _dataset_to_lists(loader, vocab_size, max_batches=max_batches)
    Y_pred = model.run(X_val)
    # model.run returns a list of arrays aligned with X_val
    se = 0.0
    count = 0
    for yt, yp in zip(Y_val, Y_pred):
        # ensure same length in case of warmup effects
        T = min(len(yt), len(yp))
        if T == 0:
            continue
        diff = yt[-T:] - yp[-T:]
        se += float(np.mean(diff * diff))
        count += 1
    return se / max(1, count)
