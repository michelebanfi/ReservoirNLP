from __future__ import annotations
from dataclasses import dataclass

# ReservoirPy imports (NumPy-based)
from reservoirpy.nodes import Reservoir, Ridge


@dataclass
class ESNConfig:
    """Configuration for building a ReservoirPy ESN model.

    Notes:
    - "sparsity" here refers to the proportion of zeros in the recurrent matrix.
      ReservoirPy expects a connectivity (density), so we internally use
      rc_connectivity = 1 - sparsity.
    - readout_dim is unused: the readout directly maps to vocab_size for LM.
    """
    input_size: int
    hidden_size: int = 256
    spectral_radius: float = 0.9
    sparsity: float = 0.9
    leak_rate: float = 0.3
    readout_dim: int = 256
    ridge: float = 1e-5
    input_scaling: float = 1.0
    seed: int | None = None


def build_reservoirpy_esn(vocab_size: int, cfg: ESNConfig):
    """Create a simple ESN: one Reservoir followed by a Ridge readout.

    Inputs/outputs are expected to be one-hot vectors of size ``vocab_size``.
    Returns a ReservoirPy Model (which behaves like a Node) supporting fit/run/step.
    """
    rc_connectivity = max(1e-6, 1.0 - float(cfg.sparsity))
    reservoir = Reservoir(
        units=cfg.hidden_size,
        sr=cfg.spectral_radius,
        lr=cfg.leak_rate,
        input_scaling=cfg.input_scaling,
        input_connectivity=1.0,
        rc_connectivity=rc_connectivity,
        input_dim=vocab_size,
        seed=cfg.seed,
        name="reservoir",
    )
    readout = Ridge(ridge=cfg.ridge, output_dim=vocab_size, name="readout")
    esn = reservoir >> readout
    return esn
