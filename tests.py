from __future__ import annotations
import numpy as np
import torch

from tokenizer import CharTokenizer
from dataset import create_dataloaders
from model import ESNLanguageModel, ESNConfig


def test_forward_backward():
    text = "hello world\nthis is a test\n"
    tok = CharTokenizer.from_texts([text])
    ids = np.array(tok.encode(text), dtype=np.int64)
    block_size = 8
    batch_size = 2
    train_loader, _ = create_dataloaders(ids, block_size=block_size, batch_size=batch_size)
    model = ESNLanguageModel(vocab_size=tok.vocab_size, cfg=ESNConfig(input_size=tok.vocab_size, hidden_size=64))
    model.train()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    logits = model(x)
    B, T, C = logits.shape
    loss = criterion(logits.view(B*T, C), y.view(B*T))
    opt.zero_grad()
    loss.backward()
    opt.step()
    print("test_forward_backward: OK, loss=", float(loss.item()))


if __name__ == "__main__":
    test_forward_backward()
