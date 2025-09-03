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
    n_reservoirs: int = 1


def build_reservoir_model(cfg: ReservoirConfig):
    """Build a simple reservoirpy model: input -> Reservoir -> Ridge.

    Returns a reservoirpy.Model instance ready for fit()/run().
    """
    # Lazy import to avoid hard dependency at module import time
    from reservoirpy.nodes import Reservoir, Ridge
    from reservoirpy import Model
    # Introspect supported kwargs for cross-version compatibility
    import inspect

    res_kwargs = dict(
        units=cfg.reservoir_size,
        sr=cfg.spectral_radius,
        lr=cfg.leak_rate,
        input_scaling=cfg.input_scaling,
        activation=np.tanh,
    )
    if cfg.n_reservoirs > 1:
        # Create a chain of reservoirs
        reservoirs = [Reservoir(
            units=cfg.reservoir_size,
            sr=cfg.spectral_radius,
            lr=cfg.leak_rate,
            input_scaling=cfg.input_scaling,
            activation=np.tanh,
            name=f"res{i+1}"
        ) for i in range(cfg.n_reservoirs)]
        
        # Connect reservoirs in a sequence
        res = reservoirs[0]
        for i in range(1, len(reservoirs)):
            res = res >> reservoirs[i]
    else:
        res = Reservoir(
            units=cfg.reservoir_size,
            sr=cfg.spectral_radius,
            lr=cfg.leak_rate,
            input_scaling=cfg.input_scaling,
            activation=np.tanh,
        )

    res_params = set(inspect.signature(Reservoir.__init__).parameters.keys())
    # connectivity argument naming differences
    if 'density' in res_params:
        res_kwargs['density'] = cfg.density
    elif 'connectivity' in res_params:
        res_kwargs['connectivity'] = cfg.density
    elif 'proba' in res_params:
        res_kwargs['proba'] = cfg.density
    elif 'p' in res_params:
        res_kwargs['p'] = cfg.density
    # seed argument naming differences
    if cfg.seed is not None:
        if 'seed' in res_params:
            res_kwargs['seed'] = cfg.seed
        elif 'random_state' in res_params:
            res_kwargs['random_state'] = cfg.seed
    res = Reservoir(**res_kwargs)

    # Ridge readout argument naming differences
    ridge_params = set(inspect.signature(Ridge.__init__).parameters.keys())
    rdg_kwargs = dict(ridge=cfg.ridge_alpha)
    if 'out_dim' in ridge_params:
        rdg_kwargs['out_dim'] = cfg.output_dim
    elif 'outdim' in ridge_params:
        rdg_kwargs['outdim'] = cfg.output_dim
    elif 'output_dim' in ridge_params:
        rdg_kwargs['output_dim'] = cfg.output_dim
    readout = Ridge(**rdg_kwargs)
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
