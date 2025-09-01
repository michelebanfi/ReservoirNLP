from __future__ import annotations
import os
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer
from dataset import create_dataloaders
from model import ReservoirConfig, build_reservoir_model, make_random_embeddings
from train import fit_offline, evaluate_mse
from config import TrainConfig


def build_tiny_corpus(data_path: str | None = None):
    # Load all .txt files from the data directory
    if data_path is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
    else:
        # If a specific file path is provided, use its directory
        if os.path.isfile(data_path):
            data_dir = os.path.dirname(data_path)
        else:
            data_dir = data_path
    
    all_texts = []
    txt_files = []
    
    # Find all .txt files in the data directory
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            if filename.endswith('.txt'):
                txt_files.append(os.path.join(data_dir, filename))
    
    # Read all .txt files
    for txt_file in sorted(txt_files):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                all_texts.append(content)
                print(f"Loaded {len(content)} characters from {os.path.basename(txt_file)}")
        except Exception as e:
            print(f"Warning: Could not read {txt_file}: {e}")
    
    # If no .txt files found, use fallback
    if not all_texts:
        print("No .txt files found in data directory, using fallback corpus")
        texts = [
            "hello world\n",
            "how are you\n",
            "once upon a time\n",
            "tiny stories are fun\n",
            "abc abc abc\n",
        ]
        return ''.join(texts)
    
    combined_text = '\n'.join(all_texts)
    print(f"Total corpus size: {len(combined_text)} characters from {len(txt_files)} files")
    return combined_text


def main():
    cfg = TrainConfig()

    # 1) Build tokenizer from tiny corpus
    full_text = build_tiny_corpus(cfg.data_path)
    # split for BPE training to avoid leakage
    split_idx = int(len(full_text) * cfg.bpe_train_split)
    train_text_for_bpe = full_text[:split_idx]
    val_text_for_bpe = full_text[split_idx:]
    text = full_text
    # Choose tokenizer
    if cfg.tokenizer_type == 'bpe':
        try:
            tok = BPETokenizer.train_from_texts([train_text_for_bpe], vocab_size=cfg.bpe_vocab_size)
            print(f"Using BPE tokenizer, vocab size: {tok.vocab_size}")
            # Optionally save tokenizer json
            tok.save(os.path.join(os.path.dirname(__file__), 'models', 'tiny_bpe.json'))
        except Exception as e:
            print(f"BPE tokenizer unavailable ({e}), falling back to char tokenizer.")
            tok = CharTokenizer.from_texts([text])
            print(f"Using Char tokenizer, vocab size: {tok.vocab_size}")
    else:
        tok = CharTokenizer.from_texts([text])
        print(f"Using Char tokenizer, vocab size: {tok.vocab_size}")

    # 2) Tokenize full text and build loaders
    ids = np.array(tok.encode(text), dtype=np.int64)
    # Use config; adjust if too small
    block_size = cfg.block_size
    if len(ids) <= block_size + 1:
        block_size = max(16, len(ids) // 3)
    batch_size = cfg.batch_size
    if batch_size * block_size > max(1, len(ids)):
        batch_size = max(8, len(ids) // max(1, block_size))
    train_loader, val_loader = create_dataloaders(ids, block_size=block_size, batch_size=batch_size)

    # 3) Model (ReservoirPy)
    # Use compact embeddings to reduce memory on Kaggle
    embed_dim = min(128, max(16, cfg.readout_dim // 2))
    embeddings = make_random_embeddings(tok.vocab_size, embed_dim)
    density = max(0.01, 1.0 - float(cfg.sparsity))
    cfg_model = ReservoirConfig(
        input_dim=embed_dim,
        output_dim=tok.vocab_size,
        reservoir_size=cfg.hidden_size,
        spectral_radius=cfg.spectral_radius,
        density=density,
        leak_rate=cfg.leak_rate,
        input_scaling=1.0,
        ridge_alpha=1e-5,
        seed=42,
    )
    model = build_reservoir_model(cfg_model)

    # 4) Train
    os.makedirs('models', exist_ok=True)
    ckpt = cfg.ckpt_path  # kept for compatibility, saving not implemented with reservoirpy here
    # Offline fit in small chunks for memory safety
    fit_offline(model, train_loader, embeddings, max_batches=64)
    # Final quick eval (MSE on one-hot targets)
    final_val = evaluate_mse(model, val_loader, embeddings, max_batches=cfg.eval_batches)
    print("Training done. Last metrics:", {
        "val_mse": final_val,
    })

    # 5) Quick sample generation (light sampling)
    # Simple greedy sampling with reservoirpy: keep only last step prediction
    context = "Who are you Alice?"
    ids_ctx = tok.encode(context)
    V = tok.vocab_size
    E = embeddings
    seq = [E[i] for i in ids_ctx]  # list of (D,) arrays
    generated: list[int] = []
    # Run context to set reservoir state
    _ = model.run(np.stack(seq, axis=0))
    # Generate next tokens
    last_vec = seq[-1]
    for _ in range(100):
        out = model.step(last_vec)
        probs = out.astype(np.float64)
        probs = probs / max(1e-12, probs.sum())
        # temperature + nucleus sampling (approximate)
        logits = np.log(probs + 1e-12) / max(1e-6, cfg.temperature)
        p = np.exp(logits)
        p = p / p.sum()
        # top-k
        if cfg.top_k > 0:
            k = min(cfg.top_k, V)
            top_idx = np.argpartition(p, -k)[-k:]
            mask = np.ones_like(p, dtype=bool)
            mask[top_idx] = False
            p[mask] = 0
            p = p / p.sum()
        # top-p
        if cfg.top_p and 0 < cfg.top_p < 1.0:
            idx_sorted = np.argsort(p)[::-1]
            cumsum = np.cumsum(p[idx_sorted])
            keep = cumsum <= cfg.top_p
            if not np.any(keep):
                keep[0] = True
            mask = np.ones_like(p, dtype=bool)
            mask[idx_sorted[keep]] = False
            p[mask] = 0
            p = p / p.sum()
    next_id = int(np.random.choice(V, p=p))
    generated.append(next_id)
    last_vec = E[next_id]
    continuation = tok.decode(generated)
    print("Sample:")
    print(context + continuation)

if __name__ == "__main__":
    main()
