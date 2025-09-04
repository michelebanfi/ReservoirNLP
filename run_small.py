from __future__ import annotations
import os
import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer
from dataset import create_dataloaders
from model import ReservoirConfig, build_reservoir_model, make_random_embeddings
from train import fit_offline, fit_online, evaluate_classification
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
    # Use configurable embedding dimension
    embed_dim = cfg.embed_dim
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
        readout_type=getattr(cfg, 'readout_type', 'ridge'),
        ridge_alpha=getattr(cfg, 'ridge_alpha', 1e-5),
        rls_alpha=getattr(cfg, 'rls_alpha', 1e-1),
        rls_forgetting=getattr(cfg, 'rls_forgetting', 1.0),
        rls_fit_bias=getattr(cfg, 'rls_fit_bias', True),
        seed=42,
        n_reservoirs=cfg.n_reservoirs,
        use_positional_encoding=cfg.use_positional_encoding,
        pos_encoding_type=cfg.pos_encoding_type,
        pos_encoding_scale=cfg.pos_encoding_scale,
        max_sequence_length=cfg.max_sequence_length,
    )
    model, pos_encoding = build_reservoir_model(cfg_model, embeddings)

    # 4) Train
    os.makedirs('models', exist_ok=True)
    ckpt = cfg.ckpt_path  # kept for compatibility, saving not implemented with reservoirpy here
    # Train with chosen readout rule
    if cfg.readout_type.lower() == 'rls':
        print("Training with RLS (online)")
        fit_online(model, train_loader, embeddings, pos_encoding, cfg.pos_encoding_scale, max_batches=64)
    else:
        print("Training with Ridge (offline)")
        fit_offline(model, train_loader, embeddings, pos_encoding, cfg.pos_encoding_scale, max_batches=64)
    # Final evaluation using proper classification metrics
    final_metrics = evaluate_classification(model, val_loader, embeddings, pos_encoding, cfg.pos_encoding_scale, max_batches=cfg.eval_batches)
    print("Training done. Evaluation metrics:", final_metrics)

    # 5) Quick sample generation (light sampling)
    # Simple greedy sampling with reservoirpy: keep only last step prediction
    context = "Who are you Alice?"
    ids_ctx = tok.encode(context)
    V = tok.vocab_size
    E = embeddings
    seq = [E[i] for i in ids_ctx]  # list of (D,) arrays
    
    # Apply positional encoding to context if enabled
    if pos_encoding is not None:
        from model import apply_positional_encoding
        context_emb = np.stack(seq, axis=0)  # (ctx_len, D)
        context_emb_pos = apply_positional_encoding(context_emb, pos_encoding, scale_factor=cfg.pos_encoding_scale)
        seq = [context_emb_pos[i] for i in range(len(seq))]  # back to list
    
    generated: list[int] = []
    # Run context to set reservoir state
    _ = model.run(np.stack(seq, axis=0))
    # Generate next tokens
    def softmax_stable(logits: np.ndarray, temperature: float) -> np.ndarray:
        t = max(1e-6, float(temperature))
        z = logits / t
        z = z - np.max(z)
        e = np.exp(z)
        s = e.sum()
        if not np.isfinite(s) or s <= 0:
            return np.full_like(logits, 1.0 / logits.size)
        return e / s

    last_vec = seq[-1]
    current_pos = len(ids_ctx)  # position for next token
    for _ in range(100):
        # Get embedding prediction from reservoir
        out_embed = model.step(last_vec)  # (D,) embedding prediction
        
        # Convert embedding to vocabulary logits via similarity  
        similarities = out_embed @ E.T  # (V,) similarities with all embeddings
        logits = similarities.astype(np.float64)
        p = softmax_stable(logits, cfg.temperature)
        # top-k
        if cfg.top_k and cfg.top_k > 0:
            k = int(min(cfg.top_k, V))
            top_idx = np.argpartition(p, -k)[-k:]
            mask = np.ones_like(p, dtype=bool)
            mask[top_idx] = False
            p[mask] = 0.0
            s = p.sum()
            if s <= 0 or not np.isfinite(s):
                p = np.zeros_like(p)
                p[top_idx] = 1.0 / len(top_idx)
            else:
                p = p / s
        # top-p
        if cfg.top_p and 0.0 < float(cfg.top_p) < 1.0:
            idx_sorted = np.argsort(p)[::-1]
            p_sorted = p[idx_sorted]
            cumsum = np.cumsum(p_sorted)
            # keep at least one token
            cutoff = np.searchsorted(cumsum, float(cfg.top_p), side='right') + 1
            cutoff = max(1, min(cutoff, V))
            keep_idx = idx_sorted[:cutoff]
            mask = np.ones_like(p, dtype=bool)
            mask[keep_idx] = False
            p[mask] = 0.0
            s = p.sum()
            if s <= 0 or not np.isfinite(s):
                p = np.zeros_like(p)
                p[keep_idx] = 1.0 / len(keep_idx)
            else:
                p = p / s
        # Final sanitize
        if not np.all(np.isfinite(p)) or p.sum() <= 0:
            p = np.full(V, 1.0 / V)
        next_id = int(np.random.choice(V, p=p))
        generated.append(next_id)
        
        # Prepare next input with positional encoding if enabled
        next_emb = E[next_id]
        if pos_encoding is not None and current_pos < pos_encoding.shape[0]:
            next_emb = next_emb + pos_encoding[current_pos] * cfg.pos_encoding_scale
        
        last_vec = next_emb
        current_pos += 1
    continuation = tok.decode(generated)
    print("Sample:")
    print(context + continuation)

if __name__ == "__main__":
    main()
