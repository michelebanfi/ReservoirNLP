from __future__ import annotations
import numpy as np

from tokenizer import CharTokenizer
from dataset import create_dataloaders
from model import ReservoirConfig, build_reservoir_model, make_random_embeddings
from train import fit_offline, evaluate_mse, batch_to_embeddings


def test_fit_and_run():
    text = "hello world\nthis is a test\n"
    tok = CharTokenizer.from_texts([text])
    ids = np.array(tok.encode(text), dtype=np.int64)
    block_size = 8
    batch_size = 2
    train_loader, val_loader = create_dataloaders(ids, block_size=block_size, batch_size=batch_size)
    # Small model
    embed_dim = 32
    E = make_random_embeddings(tok.vocab_size, embed_dim)
    cfg = ReservoirConfig(input_dim=embed_dim, output_dim=tok.vocab_size, reservoir_size=64, density=0.1)
    model = build_reservoir_model(cfg)
    # Fit on one chunk
    fit_offline(model, train_loader, E, max_batches=1)
    # Run on one batch and compute shape
    x, y = next(iter(train_loader))
    Xb, Yb = batch_to_embeddings(x, y, E)
    Yp = model.run(Xb[0])
    assert Yp.shape == Yb[0].shape
    print("test_fit_and_run: OK, shape=", Yp.shape)


if __name__ == "__main__":
    test_fit_and_run()
