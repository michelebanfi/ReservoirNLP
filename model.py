from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
import numpy as np


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
    - ridge_alpha: ridge regularization strength for readout
    """
    input_dim: int
    output_dim: int
    reservoir_size: int = 256
    spectral_radius: float = 0.9
    density: float = 0.1
    leak_rate: float = 0.3
    input_scaling: float = 1.0
    ridge_alpha: float = 1e-6
    seed: Optional[int] = 42
    n_reservoirs: int = 1


def build_reservoir_model(
    config: ReservoirConfig,
    embeddings: np.ndarray, 
) -> Any:
    """Build a reservoirpy model for next-token prediction.
    
    Now uses embeddings for both inputs AND targets, with a custom output layer
    that maps from reservoir states to final vocabulary logits.
    """
    try:
        from reservoirpy.nodes import Reservoir, Ridge
    except ImportError:
        raise ImportError("Please install reservoirpy: pip install reservoirpy")

    embed_dim = embeddings.shape[1]  # D
    vocab_size = embeddings.shape[0]  # V
    
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
    # Then we'll compute similarities with embeddings to get vocabulary logits
    ridge_kwargs = {}
    ridge_sig = inspect.signature(Ridge)
    ridge_params = set(ridge_sig.parameters.keys())
    
    if "alpha" in ridge_params:
        ridge_kwargs["alpha"] = config.ridge_alpha
    elif "ridge" in ridge_params:
        ridge_kwargs["ridge"] = config.ridge_alpha
    
    # Map to embedding space, not directly to vocabulary
    readout = Ridge(output_dim=embed_dim, **ridge_kwargs)
    
    # Chain all components
    if len(reservoirs) == 1:
        model = reservoirs[0] >> readout
    else:
        chain = reservoirs[0]
        for reservoir in reservoirs[1:]:
            chain = chain >> reservoir
        model = chain >> readout
    
    return model


def make_random_embeddings(vocab_size: int, embed_dim: int, seed: int = 42) -> np.ndarray:
    """Create fixed, precomputed embeddings to feed the reservoir.

    Using small dense random vectors avoids huge one-hot inputs and reduces memory.
    """
    rng = np.random.default_rng(seed)
    # Xavier-like scaling
    scale = np.sqrt(2.0 / (vocab_size + embed_dim))
    E = rng.normal(0.0, scale, size=(vocab_size, embed_dim)).astype(np.float32)
    return E
