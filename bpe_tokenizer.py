from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import os


@dataclass
class BPETokenizer:
    """Tiny BPE tokenizer wrapper using the `tokenizers` library.

    Falls back gracefully if library missing when imported by runner.
    """
    tok: any

    @staticmethod
    def _require_lib():
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
            from tokenizers.trainers import BpeTrainer
            from tokenizers.pre_tokenizers import ByteLevel
            from tokenizers.decoders import ByteLevel as ByteLevelDecoder
        except Exception as e:
            raise RuntimeError(
                "The `tokenizers` package is required for BPETokenizer. Install with `pip install tokenizers`."
            ) from e
        return Tokenizer, BPE, BpeTrainer, ByteLevel, ByteLevelDecoder

    @classmethod
    def train_from_texts(cls, texts: List[str], vocab_size: int = 512) -> "BPETokenizer":
        Tokenizer, BPE, BpeTrainer, ByteLevel, ByteLevelDecoder = cls._require_lib()
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        # Byte-level pretokenization ensures reversible mapping and proper spacing
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])
        tokenizer.train_from_iterator(texts, trainer)
        return cls(tok=tokenizer)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.tok.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        Tokenizer, *_ = cls._require_lib()
        return cls(tok=Tokenizer.from_file(path))

    def encode(self, s: str) -> List[int]:
        return self.tok.encode(s).ids

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()
