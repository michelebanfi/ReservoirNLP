from __future__ import annotations
from typing import Dict, List, Tuple, Any
import numpy as np

# Training with reservoirpy uses NumPy arrays. We precompute embeddings and
# convert PyTorch dataset batches to NumPy on the fly to minimize memory.


def batch_to_embeddings(
    x_batch: Any,
    y_batch: Any,
    embeddings: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert integer token batches (B,T) to lists of NumPy sequences.

    - Input sequence X_t: embedding vectors for tokens x[0..T-2]
    - Target sequence Y_t: next-token one-hot vectors for tokens x[1..T-1]
    Returns lists with length B. Each element is 2D array (T-1, D) and (T-1, V).
    """
    # Accept NumPy arrays directly (preferred) or Torch tensors (convert lazily)
    if hasattr(x_batch, "cpu") and hasattr(x_batch, "numpy"):
        x_np = x_batch.cpu().numpy()
    else:
        x_np = np.asarray(x_batch)
    if hasattr(y_batch, "cpu") and hasattr(y_batch, "numpy"):
        y_np = y_batch.cpu().numpy()
    else:
        y_np = np.asarray(y_batch)

    assert x_np.ndim == 2 and y_np.ndim == 2
    B, T = x_np.shape
    D = embeddings.shape[1]
    V = embeddings.shape[0]
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    eye = np.identity(V, dtype=np.float32)
    for b in range(B):
        # Use T-1 pairs
        xt = x_np[b, :-1]
        yt = y_np[b, :-1]
        X = embeddings[xt]  # (T-1, D)
    # One-hot targets via advanced indexing
    Y = eye[yt]  # (T-1, V)
    X_list.append(X.astype(np.float32))
    Y_list.append(Y.astype(np.float32))
    return X_list, Y_list


def fit_offline(model, train_loader, embeddings: np.ndarray, max_batches: int | None = None) -> Dict:
    """Offline training with model.fit on a subset of batches.

    To limit memory usage on Kaggle, we fit incrementally by concatenating batches
    in small chunks rather than loading all sequences at once.
    """
    history: Dict[str, float] = {}
    X_all: List[np.ndarray] = []
    Y_all: List[np.ndarray] = []
    seen = 0
    for i, (x, y) in enumerate(train_loader):
        Xb, Yb = batch_to_embeddings(x, y, embeddings)
        X_all.extend(Xb)
        Y_all.extend(Yb)
        seen += 1
        # Periodically fit to avoid holding too much in memory
        if seen % 8 == 0:
            model.fit(X_all, Y_all)
            X_all.clear(); Y_all.clear()
        if max_batches is not None and i + 1 >= max_batches:
            break
    if X_all:
        model.fit(X_all, Y_all)
        X_all.clear(); Y_all.clear()
    return history


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean((a - b) ** 2))


def evaluate_mse(model, val_loader, embeddings: np.ndarray, max_batches: int = 10) -> float:
    losses: List[float] = []
    for i, (x, y) in enumerate(val_loader):
        Xb, Yb = batch_to_embeddings(x, y, embeddings)
        # Compute predictions and accumulate MSE per sequence
        for X_seq, Y_seq in zip(Xb, Yb):
            Y_pred = model.run(X_seq)
            losses.append(_mse(Y_seq, Y_pred))
        if i + 1 >= max_batches:
            break
    return float(np.mean(losses) if losses else 0.0)
