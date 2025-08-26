from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class CharTokenizer:
    """A minimal character-level tokenizer built from a corpus.

    - build_vocab: from text(s)
    - encode/decode: integers <-> text
    """
    stoi: Dict[str, int]
    itos: List[str]

    @classmethod
    def from_texts(cls, texts: List[str]) -> "CharTokenizer":
        vocab = sorted(list({ch for t in texts for ch in t}))
        stoi = {ch: i for i, ch in enumerate(vocab)}
        itos = vocab
        return cls(stoi=stoi, itos=itos)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos[i] for i in ids if 0 <= i < len(self.itos))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)
