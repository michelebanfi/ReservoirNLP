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
    sparsity: float = 0.9
    leak_rate: float = 0.3
    readout_dim: int = 256
    # Performance tweaks
    use_sparse_reservoir: bool = True  # store W as sparse and use sparse-dense mm

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
        # If using sparse reservoir, hold as registered buffer (no grad)
        self.use_sparse = bool(cfg.use_sparse_reservoir)
        if self.use_sparse:
            self.register_buffer("W_sparse", None, persistent=False)
            # keep a dense tensor only during init; not used after
            self.register_buffer("_W_dense_tmp", torch.empty(0), persistent=False)
        else:
            self.W = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)
            for p in self.W.parameters():
                p.requires_grad = False
        for p in self.W_in.parameters():
            p.requires_grad = False
        self._init_reservoir()
        # Two-layer MLP readout for a touch more capacity
        self.readout = nn.Sequential(
            nn.Linear(cfg.hidden_size, emb_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(0.0),
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
            if self.use_sparse:
                # build a masked dense then convert to sparse CSR
                Wd = torch.zeros(H, H)
                mask = (torch.rand_like(Wd) > self.cfg.sparsity)
                Wd.uniform_(-1.0, 1.0)
                Wd = Wd * mask
                # scale to spectral radius via power iteration
                v = torch.randn(H, 1)
                for _ in range(10):
                    v = F.normalize(Wd @ v, dim=0)
                s = torch.sqrt(((Wd @ v) ** 2).sum() / (v ** 2).sum())
                Wd.mul_(self.cfg.spectral_radius / (s + 1e-8))
                # store as sparse (cpu first; it will be moved with the module)
                W_sparse = Wd.to_sparse_csr()
                # assign buffers
                self.W_sparse = W_sparse
                self._W_dense_tmp = torch.empty(0)
            else:
                W = self.W.weight
                W.zero_()
                mask = (torch.rand_like(W) > self.cfg.sparsity).float()
                W.uniform_(-1.0, 1.0)
                W.mul_(mask)
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
            # reservoir update: W_in * u + W * h
            if self.use_sparse:
                # torch.sparse.mm expects (sparse@dense)
                # (H,H) @ (H,B) -> (H,B) then transpose to (B,H)
                # Guard against AMP half precision on cuSPARSE by forcing FP32
                Wsp = self.W_sparse.to(h.device)
                h_in = h
                if torch.is_autocast_enabled():
                    # compute in fp32 to ensure support
                    with torch.cuda.amp.autocast(enabled=False):
                        Wh32 = torch.sparse.mm(Wsp.float(), h_in.T.float()).T
                    Wh = Wh32.to(dtype=h_in.dtype)
                else:
                    Wh = torch.sparse.mm(Wsp, h.T).T
            else:
                Wh = self.W(h)
            pre = self.W_in(u) + Wh
            pre = torch.clamp(pre, -20, 20)
            h_new = torch.tanh(pre)
            h = (1 - self.cfg.leak_rate) * h + self.cfg.leak_rate * h_new
            o = self.readout(h)
            outs.append(o.unsqueeze(1))
        y = torch.cat(outs, dim=1)  # [B,T,E]
        y = self.ln(y)
        logits = self.lm_head(y)
        return logits
