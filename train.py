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


def _cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute categorical cross-entropy loss between one-hot targets and predictions.
    
    Args:
        y_true: one-hot encoded targets (T, V)
        y_pred: predicted probabilities (T, V)
    Returns:
        Average cross-entropy loss
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    
    # Apply softmax to predictions to get probabilities
    y_pred = y_pred - np.max(y_pred, axis=-1, keepdims=True)  # numerical stability
    exp_pred = np.exp(y_pred)
    y_pred_softmax = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
    
    # Clip to avoid log(0)
    y_pred_softmax = np.clip(y_pred_softmax, 1e-12, 1.0)
    
    # Cross-entropy: -sum(y_true * log(y_pred))
    ce_loss = -np.sum(y_true * np.log(y_pred_softmax), axis=-1)
    return float(np.mean(ce_loss))


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy between one-hot targets and predictions.
    
    Args:
        y_true: one-hot encoded targets (T, V)
        y_pred: predicted logits (T, V)
    Returns:
        Accuracy as fraction of correct predictions
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    
    # Get predicted classes (argmax)
    pred_classes = np.argmax(y_pred, axis=-1)
    true_classes = np.argmax(y_true, axis=-1)
    
    return float(np.mean(pred_classes == true_classes))


def evaluate_classification(model, val_loader, embeddings: np.ndarray, max_batches: int = 10) -> Dict[str, float]:
    """Evaluate model using proper classification metrics (cross-entropy and accuracy)."""
    losses: List[float] = []
    accuracies: List[float] = []
    
    for i, (x, y) in enumerate(val_loader):
        Xb, Yb = batch_to_embeddings(x, y, embeddings)
        # Compute predictions and accumulate metrics per sequence
        for X_seq, Y_seq in zip(Xb, Yb):
            Y_pred = model.run(X_seq)
            losses.append(_cross_entropy(Y_seq, Y_pred))
            accuracies.append(_accuracy(Y_seq, Y_pred))
        if i + 1 >= max_batches:
            break
    
    return {
        "cross_entropy": float(np.mean(losses) if losses else 0.0),
        "accuracy": float(np.mean(accuracies) if accuracies else 0.0)
    }


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    """Legacy MSE function - kept for compatibility but shouldn't be used for classification."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.mean((a - b) ** 2))


def evaluate_mse(model, val_loader, embeddings: np.ndarray, max_batches: int = 10) -> float:
    """Legacy MSE evaluation - kept for compatibility but shouldn't be used for classification."""
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
