from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TrainConfig:
    # Tokenization
    tokenizer_type: str = "bpe"  # "char" or "bpe"
    bpe_vocab_size: int = 1024
    bpe_train_split: float = 0.9  # fraction of raw text used to train BPE

    # Data
    block_size: int = 512
    batch_size: int = 64
    data_path: str = "data/tiny.txt"

    # Model (ESN)
    hidden_size: int = 512
    spectral_radius: float = 0.85
    sparsity: float = 0.9
    leak_rate: float = 0.25
    readout_dim: int = 512
    dropout: float = 0.5

    # Training
    epochs: int = 12
    lr: float = 3e-4
    weight_decay: float = 5e-3
    label_smoothing: float = 0.05
    eval_interval: int = 100
    ckpt_path: str = "models/small_esn.pt"
    patience: int | None = 12  # early stopping eval windows; None to disable

    # Runtime
    device: str | None = None  # autodetect if None

    # Sampling
    temperature: float = 0.7
    top_k: int = 80
    top_p: float = 0.93
