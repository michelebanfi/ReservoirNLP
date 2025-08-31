from __future__ import annotations
import os
import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from tokenizer import CharTokenizer
from bpe_tokenizer import BPETokenizer
from dataset import create_dataloaders
from model import ESNLanguageModel, ESNConfig
from train import train, evaluate
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
    device = cfg.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

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

    # 3) Model
    cfg_model = ESNConfig(
        input_size=tok.vocab_size,
        hidden_size=cfg.hidden_size,
        spectral_radius=cfg.spectral_radius,
        sparsity=cfg.sparsity,
        leak_rate=cfg.leak_rate,
        readout_dim=cfg.readout_dim,
    )
    model = ESNLanguageModel(vocab_size=tok.vocab_size, cfg=cfg_model)
    # set dropout within readout if present
    if isinstance(model.readout, torch.nn.Sequential) and len(model.readout) >= 3:
        if isinstance(model.readout[2], torch.nn.Dropout):
            model.readout[2].p = cfg.dropout

    # 4) Train
    os.makedirs('models', exist_ok=True)
    ckpt = cfg.ckpt_path
    history = train(
        model, train_loader, val_loader, device,
        epochs=cfg.epochs, lr=cfg.lr, eval_interval=cfg.eval_interval, ckpt_path=ckpt,
        weight_decay=cfg.weight_decay, label_smoothing=cfg.label_smoothing, patience=cfg.patience,
    )
    # Final quick eval
    final_val = evaluate(model, val_loader, device)
    print("Training done. Last metrics:", {
        "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "val_loss": final_val,
        "steps": history["steps"][-1] if history["steps"] else None,
    })

    # 5) Quick sample generation (light sampling)
    model.eval()
    context = "Who are you Alice?"
    x = torch.tensor([tok.encode(context)], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(100):
            logits = model(x)
            last = logits[:, -1, :]
            # temperature + top-k/top-p
            last = last / max(1e-5, cfg.temperature)
            # top-k
            if cfg.top_k > 0:
                k = min(cfg.top_k, last.size(-1))
                vals, idxs = torch.topk(last, k=k, dim=-1)
                logits_filt = vals
                indices_map = idxs
            else:
                logits_filt = last
                indices_map = torch.arange(last.size(-1), device=last.device).unsqueeze(0)
            # top-p nucleus on the filtered set
            if cfg.top_p and 0 < cfg.top_p < 1.0:
                probs = torch.softmax(logits_filt, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
                cum = torch.cumsum(sorted_probs, dim=-1)
                mask = cum > cfg.top_p
                mask[..., 0] = False
                keep = ~mask
                # rebuild a masked distribution
                filtered = torch.full_like(sorted_probs, float('-inf'))
                filtered[keep] = torch.log(sorted_probs[keep] + 1e-9)
                # sample
                choice_sorted = torch.multinomial(torch.softmax(filtered, dim=-1), num_samples=1)
                choice = sorted_idx.gather(1, choice_sorted)
                next_id = indices_map.gather(1, choice)
            else:
                probs = torch.softmax(logits_filt, dim=-1)
                choice = torch.multinomial(probs, num_samples=1)
                next_id = indices_map.gather(1, choice)
            x = torch.cat([x, next_id], dim=1)
    out = ''.join(tok.decode(x[0].tolist()))
    print("Sample:")
    print(out)

if __name__ == "__main__":
    main()
