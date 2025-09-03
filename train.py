from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Training with reservoirpy uses NumPy arrays. We precompute embeddings and
# convert PyTorch dataset batches to NumPy on the fly to minimize memory.


def batch_to_embeddings(
    x_batch: Any,
    y_batch: Any,
    embeddings: np.ndarray,
    pos_encoding: Optional[np.ndarray] = None,
    pos_scale: float = 0.1,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Convert integer token batches (B,T) to lists of NumPy sequences with optional positional encoding.

    - Input sequence X_t: embedding vectors for tokens x[0..T-2] + positional encoding
    - Target sequence Y_t: embedding vectors for tokens x[1..T-1] (NOT one-hot!)
    Returns lists with length B. Each element is 2D array (T-1, D) for both inputs and targets.
    This approach uses embeddings for both inputs and targets, avoiding large one-hot matrices.
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
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    
    for b in range(B):
        # Use T-1 pairs
        xt = x_np[b, :-1]  # input tokens
        yt = y_np[b, :-1]  # target tokens
        X = embeddings[xt]  # (T-1, D) - embed input tokens  
        Y = embeddings[yt]  # (T-1, D) - embed target tokens (NOT one-hot!)
        
        # Apply positional encoding to inputs if provided
        if pos_encoding is not None:
            from model import apply_positional_encoding
            X = apply_positional_encoding(X, pos_encoding, scale_factor=pos_scale)
        
        X_list.append(X.astype(np.float32))
        Y_list.append(Y.astype(np.float32))
    return X_list, Y_list


def fit_offline(model, train_loader, embeddings: np.ndarray, pos_encoding: Optional[np.ndarray] = None, pos_scale: float = 0.1, max_batches: int | None = None) -> Dict:
    """Offline training with model.fit on a subset of batches.

    To limit memory usage on Kaggle, we fit incrementally by concatenating batches
    in small chunks rather than loading all sequences at once.
    Now includes optional positional encoding support.
    """
    history: Dict[str, float] = {}
    X_all: List[np.ndarray] = []
    Y_all: List[np.ndarray] = []
    seen = 0
    for i, (x, y) in enumerate(train_loader):
        Xb, Yb = batch_to_embeddings(x, y, embeddings, pos_encoding, pos_scale)
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


def compute_vocab_logits(
    pred_embeddings: np.ndarray,
    embeddings: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """Convert predicted embeddings to vocabulary logits via similarity.
    
    Args:
        pred_embeddings: (T, D) predicted embedding vectors
        embeddings: (V, D) vocabulary embedding matrix  
        temperature: softmax temperature for calibration
        
    Returns:
        logits: (T, V) unnormalized log probabilities
    """
    # Compute similarities: (T, D) @ (D, V) -> (T, V)
    similarities = pred_embeddings @ embeddings.T  # (T, V)
    return similarities / temperature


def _sparse_cross_entropy_from_embeddings(
    y_pred_embeddings: np.ndarray,
    y_true_indices: np.ndarray,
    embeddings: np.ndarray
) -> float:
    """Compute cross-entropy loss between predicted embeddings and target tokens.
    
    Args:
        y_pred_embeddings: (T, D) predicted embedding vectors
        y_true_indices: (T,) target token indices
        embeddings: (V, D) vocabulary embeddings for computing logits
        
    Returns:
        Average cross-entropy loss over sequence
    """
    # Convert embeddings to logits via similarity
    logits = compute_vocab_logits(y_pred_embeddings, embeddings)  # (T, V)
    
    # Use sparse cross-entropy (more memory efficient than one-hot)
    T = logits.shape[0]
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))  # (T, V)
    
    # Gather log probabilities for true classes
    true_log_probs = log_probs[np.arange(T), y_true_indices]  # (T,)
    return -np.mean(true_log_probs)
    """Compute sparse categorical cross-entropy loss from one-hot targets.
    
    Args:
        y_true_onehot: one-hot encoded targets (T, V)
        y_pred_logits: predicted logits (T, V)
    Returns:
        Average cross-entropy loss
    """
    y_true_onehot = np.asarray(y_true_onehot, dtype=np.float32)
    y_pred_logits = np.asarray(y_pred_logits, dtype=np.float32)
    
    # Convert one-hot to sparse indices
    y_true_indices = np.argmax(y_true_onehot, axis=-1).astype(np.int32)
    
    # Apply softmax to predictions to get probabilities
    y_pred_logits = y_pred_logits - np.max(y_pred_logits, axis=-1, keepdims=True)  # numerical stability
    exp_pred = np.exp(y_pred_logits)
    y_pred_softmax = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
    
    # Clip to avoid log(0)
    y_pred_softmax = np.clip(y_pred_softmax, 1e-12, 1.0)
    
    # Sparse cross-entropy: -log(pred[true_class]) for each timestep
    T = y_true_indices.shape[0]
    ce_losses = []
    for t in range(T):
        true_class = y_true_indices[t]
        if 0 <= true_class < y_pred_softmax.shape[1]:  # valid class
            ce_losses.append(-np.log(y_pred_softmax[t, true_class]))
        else:  # invalid class (shouldn't happen but safety)
            ce_losses.append(10.0)  # high penalty
    
    return float(np.mean(ce_losses))


def _sparse_accuracy_from_onehot(y_true_onehot: np.ndarray, y_pred_logits: np.ndarray) -> float:
    """Compute accuracy from one-hot targets.
    
    Args:
        y_true_onehot: one-hot encoded targets (T, V)
        y_pred_logits: predicted logits (T, V)
    Returns:
        Accuracy as fraction of correct predictions
    """
    y_true_onehot = np.asarray(y_true_onehot, dtype=np.float32)
    y_pred_logits = np.asarray(y_pred_logits, dtype=np.float32)
    
    # Convert one-hot to sparse indices
    y_true_indices = np.argmax(y_true_onehot, axis=-1)
    
    # Get predicted classes (argmax)
    pred_classes = np.argmax(y_pred_logits, axis=-1)
    
    return float(np.mean(pred_classes == y_true_indices))


def evaluate_classification(model, val_loader, embeddings: np.ndarray, pos_encoding: Optional[np.ndarray] = None, pos_scale: float = 0.1, max_batches: int = 10) -> Dict[str, float]:
    """Evaluate model using proper classification metrics (cross-entropy and accuracy).
    
    Now works with embedding-to-embedding approach, converting predictions to logits for metrics.
    Includes optional positional encoding support.
    """
    losses: List[float] = []
    accuracies: List[float] = []
    
    for i, (x, y) in enumerate(val_loader):
        Xb, Yb = batch_to_embeddings(x, y, embeddings, pos_encoding, pos_scale)
        # Compute predictions and accumulate metrics per sequence
        for X_seq, Y_embed_seq in zip(Xb, Yb):
            # Get embedding predictions from reservoir
            Y_pred_embed = model.run(X_seq)  # (T, D)
            
            # Convert target embeddings back to token indices for loss computation
            T = Y_embed_seq.shape[0]
            y_true_indices = []
            for t in range(T):
                # Find which embedding this target corresponds to
                distances = np.sum((embeddings - Y_embed_seq[t]) ** 2, axis=1)
                y_true_indices.append(np.argmin(distances))
            y_true_indices = np.array(y_true_indices)
            
            # Compute loss and accuracy using embedding-based approach
            loss = _sparse_cross_entropy_from_embeddings(Y_pred_embed, y_true_indices, embeddings)
            
            # Compute accuracy
            logits = compute_vocab_logits(Y_pred_embed, embeddings)
            pred_tokens = np.argmax(logits, axis=1)
            accuracy = np.mean(pred_tokens == y_true_indices)
            
            losses.append(loss)
            accuracies.append(accuracy)
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
