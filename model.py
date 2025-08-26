from __future__ import annotations
import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ESNConfig:
    input_size: int
    hidden_size: int = 256
    spectral_radius: float = 0.9
    sparsity: float = 0.9  # fraction of zeros
    leak_rate: float = 0.3
    readout_dim: int = 256

class ESNLanguageModel(nn.Module):
    """Minimal Echo State Network LM:
    - char embedding -> ESN reservoir (fixed) -> linear readout to vocab logits
    Only readout and embedding are trained.
    """
    def __init__(self, vocab_size: int, cfg: ESNConfig):
        super().__init__()
        self.vocab_size = vocab_size
        self.cfg = cfg
        emb_dim = cfg.readout_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)
        # fixed reservoir weights
        self.W_in = nn.Linear(emb_dim, cfg.hidden_size, bias=False)
        self.W = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
        for p in list(self.W_in.parameters()) + list(self.W.parameters()):
            p.requires_grad = False
        self._init_reservoir()
        # Two-layer MLP readout for a touch more capacity
        self.readout = nn.Sequential(
            nn.Linear(cfg.hidden_size, emb_dim),
            nn.GELU(),
            nn.Dropout(0.0),  # set from runner
            nn.Linear(emb_dim, emb_dim),
        )
        self.ln = nn.LayerNorm(emb_dim)
        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)
        # tie weights (optional)
        self.lm_head.weight = self.embed.weight

    def _init_reservoir(self):
        with torch.no_grad():
            self.W_in.weight.uniform_(-0.1, 0.1)
            H = self.cfg.hidden_size
            W = self.W.weight
            W.zero_()
            # create sparse random matrix
            mask = (torch.rand_like(W) > self.cfg.sparsity).float()
            W.uniform_(-1.0, 1.0)
            W.mul_(mask)
            # scale to spectral radius approx via power iteration
            v = torch.randn(H, 1, device=W.device)
            for _ in range(10):
                v = F.normalize(W @ v, dim=0)
            s = torch.sqrt(((W @ v)**2).sum() / (v**2).sum())
            W.mul_(self.cfg.spectral_radius / (s + 1e-8))

    def forward(self, idx: torch.LongTensor):  # [B,T]
        B, T = idx.shape
        x = self.embed(idx)  # [B,T,E]
        h = torch.zeros(B, self.cfg.hidden_size, device=idx.device)
        outs = []
        for t in range(T):
            u = x[:, t, :]
            pre = self.W_in(u) + self.W(h)
            pre = torch.clamp(pre, -20, 20)
            h_new = torch.tanh(pre)
            h = (1 - self.cfg.leak_rate) * h + self.cfg.leak_rate * h_new
            o = self.readout(h)
            outs.append(o.unsqueeze(1))
        y = torch.cat(outs, dim=1)  # [B,T,E]
        y = self.ln(y)
        logits = self.lm_head(y)
        return logits
