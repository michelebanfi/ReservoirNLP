import torch
import torch.nn as nn
from transformers import AutoTokenizer
from tqdm import tqdm

from src.config import Config
from src.model import NanoHRMv3
from src.train import train_step, compute_loss
from src.dataset import get_dataloaders

def overfit_test():
    print("=== Overfitting Test ===")
    cfg = Config()
    cfg.BATCH_SIZE = 5      # Small batch
    cfg.TRAIN_SIZE = 5      # 1 batch total
    cfg.EPOCHS = 200        # Enough to memorize
    cfg.LEARNING_RATE = 3e-4 # Slightly higher
    
    device = cfg.DEVICE
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    model = NanoHRMv3(tokenizer, cfg).to(device)
    
    # Get 1 batch
    train_loader, _ = get_dataloaders(tokenizer, cfg)
    batch = next(iter(train_loader))
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE)
    
    print("Training on single batch...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        metrics = train_step(model, batch, optimizer, cfg, epoch)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {metrics['loss']:.4f} (LM: {metrics['lm_loss']:.4f})")
            
        if metrics['lm_loss'] < 0.1:
            print("Memorization successful!")
            break
            
    # Test Generation
    print("\nTesting Verification Generation...")
    model.eval()
    with torch.no_grad():
        question = batch['raw_question'][0]
        context = batch['input_ids'][0] # Need to decode back to see context?
        target = batch['raw_answer'][0]
        
        print(f"Q: {question}")
        print(f"Target: {target}")
        
        # Simple Greedy Gen similar to query.py
        memory, src_mask = model.encode(batch['input_ids'][0:1].to(device))
        zH, zL = model.hrm_core.init_state(1, memory.size(1), device)
        
        # Run 1 segment
        zH, zL = model.hrm_core.forward_segment(zH, zL, memory, key_padding_mask=src_mask)
        
        decoder_input = torch.tensor([[0]], device=device)
        enhanced_memory = memory + zH
        
        toks = []
        for _ in range(20):
            tgt_emb = model.dec_pos(model.dec_embedding(decoder_input)) # Should use helper logic but manual ok
            # ... Copy paste logic from model.decode mostly
            # Wait, model.decode returns logits for provided labels.
            # We must run autoregressive here.
            # Reuse logic from query.py roughly
            tgt_len = decoder_input.size(1)
            tgt_causal_mask = torch.triu(torch.full((tgt_len, tgt_len), float('-inf'), device=device), diagonal=1)
            
            dec_out = model.decoder(
                tgt=tgt_emb,
                memory=enhanced_memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=src_mask
            )
            logits = model.lm_head(dec_out)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            if next_token.item() == 1: break # EOS
            toks.append(next_token.item())
            decoder_input = torch.cat([decoder_input, next_token], dim=1)
            
        pred = tokenizer.decode(toks)
        print(f"Prediction: {pred}")

if __name__ == "__main__":
    overfit_test()
