import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .config import Config
from .model import NanoHRMv3
from .dataset import get_dataloaders

def compute_loss(logits, labels):
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

def get_baseline_samples(tokenizer, config, num_samples=3):
    """
    Load Standard T5, generate answers for a few validation samples, then unload.
    Returns: list of dicts {question, context, target, baseline_answer}
    """
    print("Generating T5 Baseline Answers for Validation Comparison...")
    device = config.DEVICE
    
    # 1. Get Samples
    _, val_loader = get_dataloaders(tokenizer, config)
    # Get a batch
    batch = next(iter(val_loader)) # assuming batch size >= num_samples
    
    samples = []
    
    # 2. Load T5 (Frozen/Eval)
    t5_model = T5ForConditionalGeneration.from_pretrained(config.TOKENIZER_NAME).to(device)
    t5_model.eval()
    
    with torch.no_grad():
        for i in range(min(num_samples, len(batch['input_ids']))):
            input_ids = batch['input_ids'][i:i+1].to(device)
            target_ids = batch['labels'][i:i+1] # contains -100
            
            # Generate T5 Answer
            gen_out = t5_model.generate(input_ids, max_new_tokens=32)
            base_ans = tokenizer.decode(gen_out[0], skip_special_tokens=True)
            
            # Get Raw
            raw_q = batch['raw_question'][i]
            # reconstruct target text roughly or use raw_answer if available (dataset.py provided it)
            raw_tgt = batch['raw_answer'][i]
            context = "" # Not easily accessible unless we carry it. Dataset puts it in text.
            
            samples.append({
                'input_ids': input_ids,
                'question': raw_q,
                'target': raw_tgt,
                'baseline': base_ans
            })
            
    print("Baseline Generated. Unloading T5...")
    del t5_model
    torch.cuda.empty_cache()
    return samples

def run_validation_comparison(model, tokenizer, samples, epoch):
    """
    Run HRM on the samples and print comparison table.
    """
    model.eval()
    device = next(model.parameters()).device
    
    print(f"\n\n=== Validation Comparison (Epoch {epoch+1}) ===")
    print(f"{'Question':<50} | {'Target':<30} | {'T5 Baseline':<30} | {'HRM':<30}")
    print("-" * 150)
    
    with torch.no_grad():
        for s in samples:
            input_ids = s['input_ids'].to(device)
            
            # Enc
            memory, src_mask = model.encode(input_ids)
            B, L, D = memory.shape
            zH, zL = model.hrm_core.init_state(B, L, device)
            
            # Run 1 segment (or max segments) for inference?
            # Let's run fixed 2 segments for quick check, or loop until halt.
            # Mirror query.py logic roughly
            halted = False
            for m in range(4): # Limit check
                zH, zL = model.hrm_core.forward_segment(zH, zL, memory, key_padding_mask=src_mask)
                q_probs = model.hrm_core.get_q_values(zH)
                if q_probs[0,0] > q_probs[0,1] and m>=1:
                    break
            
            enhanced = memory + zH
            
            # Generate
            # We need to use autoregressive generation manually or wrap model
            # For quick visualization, we implement simple greedy loop here
            # reusing code from query.py logic but compact
            
            decoder_input = torch.tensor([[0]], device=device) # Pad/Start
            gen_toks = []
            for _ in range(32):
                logits = model.decode(enhanced, decoder_input, src_mask) # logits [1, Seq, Vocab]
                next_tok = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                if next_tok.item() == 1: break # EOS
                gen_toks.append(next_tok.item())
                decoder_input = torch.cat([decoder_input, next_tok], dim=1)
                
            hrm_ans = tokenizer.decode(gen_toks, skip_special_tokens=True)
            
            # Truncate for display
            q_disp = (s['question'][:47] + '..') if len(s['question']) > 47 else s['question']
            t_disp = (s['target'][:27] + '..') if len(s['target']) > 27 else s['target']
            b_disp = (s['baseline'][:27] + '..') if len(s['baseline']) > 27 else s['baseline']
            h_disp = (hrm_ans[:27] + '..') if len(hrm_ans) > 27 else hrm_ans
            
            print(f"{q_disp:<50} | {t_disp:<30} | {b_disp:<30} | {h_disp:<30}")
    print("-" * 150 + "\n")

def train_step(model, batch, optimizer, config, epoch):
    device = config.DEVICE
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    # 1. Encode
    memory, src_mask = model.encode(input_ids)
    
    B, L, D = memory.shape
    zH, zL = model.hrm_core.init_state(B, L, device)
    
    M_max = config.MAX_SEGMENTS
    total_loss = 0
    segment_count = 0
    
    for m in range(M_max):
        zH_in = zH.detach()
        zL_in = zL.detach()
        
        # Encoder 1-step gradient approximation
        if m == 0:
            current_memory = memory
        else:
            current_memory = memory.detach()
        
        zH, zL = model.hrm_core.forward_segment(zH_in, zL_in, current_memory, key_padding_mask=src_mask)
        
        # Decode
        # Enhanced memory
        enhanced_memory = current_memory + zH
        logits = model.decode(enhanced_memory, labels, src_mask)
        
        lm_loss = compute_loss(logits, labels)
        
        # Simplified Loss (Focus on LM for pretraining robustness first?)
        # Adding Q-loss if we want ACT. 
        # reusing logic from previous step:
        # For now, let's keep it simple: Just LM loss to verify T5 integration works.
        # We can add ACT loss back if it trains well.
        
        loss = lm_loss
        
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        optimizer.step()
        
        total_loss += loss.item()
        segment_count += 1
    
    return {'loss': total_loss / segment_count}

def train_main():
    print("Initializing Training...")
    cfg = Config()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    
    # 0. Comparison Setup
    validation_samples = get_baseline_samples(tokenizer, cfg)
    
    model = NanoHRMv3(tokenizer, cfg).to(cfg.DEVICE)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    train_loader, val_loader = get_dataloaders(tokenizer, cfg)
    
    # Optimizer (T5 usually requires scheduling, but standard AdamW ok for now)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    print("Starting Loop...")
    for epoch in range(cfg.EPOCHS):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        epoch_loss = 0
        for batch in pbar:
            metrics = train_step(model, batch, optimizer, cfg, epoch)
            epoch_loss += metrics['loss']
            pbar.set_postfix(metrics)
            
        # Comparison
        run_validation_comparison(model, tokenizer, validation_samples, epoch)
        
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_main()
