from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
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


def build_reservoir_model(cfg: ReservoirConfig):
    """Build a simple reservoirpy model: input -> Reservoir -> Ridge.

    Returns a reservoirpy.Model instance ready for fit()/run().
    """
    # Lazy import to avoid hard dependency at module import time
    from reservoirpy.nodes import Reservoir, Ridge
    from reservoirpy import Model
    res = Reservoir(
        units=cfg.reservoir_size,
        sr=cfg.spectral_radius,
        lr=cfg.leak_rate,
        input_scaling=cfg.input_scaling,
        density=cfg.density,
        activation=np.tanh,
        seed=cfg.seed,
    )
    readout = Ridge(ridge=cfg.ridge_alpha, out_dim=cfg.output_dim)
    model = res >> readout
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
