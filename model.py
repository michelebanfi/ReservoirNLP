from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np


def create_sinusoidal_positional_encoding(
    max_len: int, 
    embed_dim: int
) -> np.ndarray:
    """Create sinusoidal positional encoding matrix.
    
    Args:
        max_len: Maximum sequence length
        embed_dim: Embedding dimension
        
    Returns:
        pos_encoding: (max_len, embed_dim) positional encoding matrix
    """
    pos_encoding = np.zeros((max_len, embed_dim), dtype=np.float32)
    
    position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
    div_term = np.exp(np.arange(0, embed_dim, 2, dtype=np.float32) * 
                     -(np.log(10000.0) / embed_dim))
    
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    if embed_dim > 1:
        pos_encoding[:, 1::2] = np.cos(position * div_term[:pos_encoding[:, 1::2].shape[1]])
    
    return pos_encoding


def create_learned_positional_encoding(
    max_len: int, 
    embed_dim: int,
    seed: Optional[int] = None
) -> np.ndarray:
    """Create learned (random) positional encoding matrix.
    
    Args:
        max_len: Maximum sequence length
        embed_dim: Embedding dimension
        seed: Random seed for reproducibility
        
    Returns:
        pos_encoding: (max_len, embed_dim) positional encoding matrix
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize with small random values (similar to transformer practice)
    pos_encoding = np.random.normal(0, 0.02, (max_len, embed_dim)).astype(np.float32)
    return pos_encoding


def apply_positional_encoding(
    embeddings: np.ndarray,
    pos_encoding: np.ndarray,
    positions: Optional[np.ndarray] = None,
    scale_factor: float = 0.1
) -> np.ndarray:
    """Apply positional encoding to embeddings.
    
    Args:
        embeddings: (seq_len, embed_dim) token embeddings
        pos_encoding: (max_len, embed_dim) positional encoding matrix
        positions: (seq_len,) position indices. If None, use 0, 1, 2, ...
        scale_factor: scaling factor for positional encoding (to avoid overwhelming embeddings)
        
    Returns:
        encoded_embeddings: (seq_len, embed_dim) embeddings + scaled positional encoding
    """
    seq_len, embed_dim = embeddings.shape
    
    if positions is None:
        positions = np.arange(seq_len)
    
    # Ensure positions are within bounds
    positions = np.clip(positions, 0, pos_encoding.shape[0] - 1)
    
    # Add scaled positional encoding
    pos_emb = pos_encoding[positions] * scale_factor  # (seq_len, embed_dim)
    return embeddings + pos_emb


@dataclass
class ReservoirConfig:
    """Configuration for a reservoirpy ESN-like model.

    - input_dim: dimension of input features (e.g., embedding size)
    - output_dim: dimension of outputs (e.g., vocab size)
    - reservoir_size: number of reservoir units
    - spectral_radius: echo state spectral radius
    - density: connection density in (0,1]; use 1 - sparsity from old config
    - leak_rate: leaky integration rate
    - input_scaling: scaling for inputs
    - readout_type: 'ridge' (default) or 'rls'
    - ridge_alpha: ridge regularization strength for Ridge readout
    - rls_alpha: diagonal initialization of P matrix for RLS
    - rls_forgetting: forgetting factor for RLS (<=1.0)
    - rls_fit_bias: whether to learn a bias term in RLS
    - use_positional_encoding: whether to add positional encoding
    - pos_encoding_type: type of positional encoding ("sinusoidal", "learned", "none")
    - max_sequence_length: maximum sequence length for positional encoding
    """
    input_dim: int
    output_dim: int
    reservoir_size: int = 256
    spectral_radius: float = 0.9
    density: float = 0.1
    leak_rate: float = 0.3
    input_scaling: float = 1.0
    # Readout / learning rule
    readout_type: str = "ridge"  # "ridge" or "rls"
    ridge_alpha: float = 1e-6
    rls_alpha: float = 1e-1
    rls_forgetting: float = 1.0
    rls_fit_bias: bool = True
    seed: Optional[int] = 42
    n_reservoirs: int = 1
    use_positional_encoding: bool = True
    pos_encoding_type: str = "sinusoidal"  # "sinusoidal", "learned", "none"
    pos_encoding_scale: float = 0.1  # scaling factor for positional encoding
    max_sequence_length: int = 2048


def build_reservoir_model(
    config: ReservoirConfig,
    embeddings: np.ndarray, 
) -> tuple[Any, Optional[np.ndarray]]:
    """Build a reservoirpy model for next-token prediction with optional positional encoding.
    
    Args:
        config: Model configuration
        embeddings: Fixed embedding matrix (V, D)
        
    Returns:
        model: reservoirpy model (Reservoir >> Ridge) 
        pos_encoding: positional encoding matrix (max_len, D) or None if disabled
    """
    try:
        from reservoirpy.nodes import Reservoir, Ridge, RLS
    except ImportError:
        raise ImportError("Please install reservoirpy: pip install reservoirpy")

    embed_dim = embeddings.shape[1]  # D
    vocab_size = embeddings.shape[0]  # V
    
    # Create positional encoding if enabled
    pos_encoding = None
    if config.use_positional_encoding and config.pos_encoding_type != "none":
        if config.pos_encoding_type == "sinusoidal":
            pos_encoding = create_sinusoidal_positional_encoding(
                config.max_sequence_length, embed_dim
            )
        elif config.pos_encoding_type == "learned":
            pos_encoding = create_learned_positional_encoding(
                config.max_sequence_length, embed_dim, config.seed
            )
        else:
            raise ValueError(f"Unknown positional encoding type: {config.pos_encoding_type}")
    
    reservoirs = []
    prev_output_size = embed_dim
    
    # Build reservoir chain
    for i in range(config.n_reservoirs):
        # Parameter introspection for cross-version compatibility
        reservoir_kwargs = {}
        import inspect
        reservoir_sig = inspect.signature(Reservoir)
        reservoir_params = set(reservoir_sig.parameters.keys())
        
        if "units" in reservoir_params:
            reservoir_kwargs["units"] = config.reservoir_size
        elif "N" in reservoir_params:
            reservoir_kwargs["N"] = config.reservoir_size
            
        if "lr" in reservoir_params:
            reservoir_kwargs["lr"] = config.leak_rate
        elif "leaking_rate" in reservoir_params:
            reservoir_kwargs["leaking_rate"] = config.leak_rate
            
        if "sr" in reservoir_params:
            reservoir_kwargs["sr"] = config.spectral_radius
        elif "spectral_radius" in reservoir_params:
            reservoir_kwargs["spectral_radius"] = config.spectral_radius
            
        reservoir_kwargs.update({
            "input_scaling": config.input_scaling,
            "seed": config.seed + i
        })
        
        reservoir = Reservoir(**reservoir_kwargs)
        reservoirs.append(reservoir)
        prev_output_size = config.reservoir_size
    
    # Output layer: Map reservoir states (H_dim) to embedding space (D)
    # Choose learning rule: Ridge (offline) or RLS (online-capable)
    if config.readout_type.lower() == "ridge":
        ridge_kwargs = {}
        ridge_sig = inspect.signature(Ridge)
        ridge_params = set(ridge_sig.parameters.keys())
        if "alpha" in ridge_params:
            ridge_kwargs["alpha"] = config.ridge_alpha
        elif "ridge" in ridge_params:
            ridge_kwargs["ridge"] = config.ridge_alpha
        readout = Ridge(output_dim=embed_dim, **ridge_kwargs)
    elif config.readout_type.lower() == "rls":
        rls_kwargs = {"output_dim": embed_dim}
        rls_sig = inspect.signature(RLS)
        rls_params = set(rls_sig.parameters.keys())
        if "alpha" in rls_params:
            rls_kwargs["alpha"] = config.rls_alpha
        if "forgetting" in rls_params:
            rls_kwargs["forgetting"] = config.rls_forgetting
        if "fit_bias" in rls_params:
            rls_kwargs["fit_bias"] = config.rls_fit_bias
        readout = RLS(**rls_kwargs)
    else:
        raise ValueError(f"Unknown readout_type: {config.readout_type}. Use 'ridge' or 'rls'.")
    
    # Chain all components
    if len(reservoirs) == 1:
        model = reservoirs[0] >> readout
    else:
        chain = reservoirs[0]
        for reservoir in reservoirs[1:]:
            chain = chain >> reservoir
        model = chain >> readout
    
    return model, pos_encoding


def make_random_embeddings(vocab_size: int, embed_dim: int, seed: int = 42) -> np.ndarray:
    """Create fixed, precomputed embeddings to feed the reservoir.

    Using small dense random vectors avoids huge one-hot inputs and reduces memory.
    """
    rng = np.random.default_rng(seed)
    # Xavier-like scaling
    scale = np.sqrt(2.0 / (vocab_size + embed_dim))
    E = rng.normal(0.0, scale, size=(vocab_size, embed_dim)).astype(np.float32)
    return E
