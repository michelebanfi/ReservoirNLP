from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Tokenization
    tokenizer_type: str = "bpe"  # "char" or "bpe"
    bpe_vocab_size: int = 1024
    bpe_train_split: float = 0.9  # fraction of raw text used to train BPE

    # Data
    block_size: int = 128
    batch_size: int = 32
    data_path: str = "data"  # directory containing .txt files

    # Model (ESN)
    hidden_size: int = 1024
    spectral_radius: float = 0.85
    sparsity: float = 0.9
    leak_rate: float = 0.25
    readout_dim: int = 1024
    embed_dim: int = 512  # dimension of input/output embeddings
    use_positional_encoding: bool = True  # add positional encoding to embeddings
    pos_encoding_type: str = "sinusoidal"  # "sinusoidal", "learned", or "none"
    pos_encoding_scale: float = 0.1  # scaling factor for positional encoding
    max_sequence_length: int = 2048  # maximum sequence length for positional encoding
    use_sparse_reservoir: bool = True
    n_reservoirs: int = 3

    # Training
    readout_type: str = "ridge"  # "ridge" or "rls"
    ridge_alpha: float = 1e-5
    rls_alpha: float = 1e-1
    rls_forgetting: float = 1.0
    rls_fit_bias: bool = True
    epochs: int = 2
    lr: float = 3e-4
    weight_decay: float = 5e-3
    label_smoothing: float = 0.05
    eval_interval: int = 200
    ckpt_path: str = "models/small_esn.pt"
    patience: int | None = None  # early stopping eval windows; None to disable
    eval_batches: int = 5  # number of batches to use during eval for speed

    # Runtime
    device: str | None = None  # autodetect if None

    # Sampling
    temperature: float = 0.7
    top_k: int = 80
    top_p: float = 0.93
