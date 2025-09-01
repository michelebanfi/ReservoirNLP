from __future__ import annotations
import numpy as np

from tokenizer import CharTokenizer
from dataset import create_dataloaders
from model import ESNConfig, build_reservoirpy_esn


def test_fit_and_run():
    text = "hello world\nthis is a test\n"
    tok = CharTokenizer.from_texts([text])
    ids = np.array(tok.encode(text), dtype=np.int64)
    block_size = 8
    batch_size = 2
    train_loader, val_loader = create_dataloaders(ids, block_size=block_size, batch_size=batch_size)
    cfg = ESNConfig(input_size=tok.vocab_size, hidden_size=64)
    model = build_reservoirpy_esn(tok.vocab_size, cfg)
    # Build small dataset lists
    from train import _dataset_to_lists
    X, Y = _dataset_to_lists(train_loader, tok.vocab_size, max_batches=1)
    model.fit(X, Y)
    Yp = model.run(X)
    assert len(Yp) == len(X)
    print("test_fit_and_run: OK, series=", len(Yp))


if __name__ == "__main__":
    test_fit_and_run()
