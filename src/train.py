import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

# ============== Scoring Utilities ==============
import re
import string

def normalize_text(text):
    """Normalize text for QA evaluation (lowercase, remove punctuation/articles)."""
    text = text.lower()
    # Remove punctuation
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Collapse whitespace
    text = ' '.join(text.split())
    return text

def compute_exact_match(prediction, target):
    """Check if normalized prediction matches normalized target."""
    return float(normalize_text(prediction) == normalize_text(target))

def compute_f1(prediction, target):
    """Compute token-level F1 score between prediction and target."""
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    
    if len(pred_tokens) == 0 or len(target_tokens) == 0:
        return float(pred_tokens == target_tokens)
    
    common = set(pred_tokens) & set(target_tokens)
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

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
    Run HRM on the samples, collect metrics, and print comparison table.
    Returns: (validation_samples, hrm_metrics, accuracy_metrics)
    """
    model.eval()
    device = next(model.parameters()).device
    
    print(f"\n\n=== Validation Comparison (Epoch {epoch+1}) ===")
    print(f"{'Question':<50} | {'Target':<30} | {'T5 Baseline':<30} | {'HRM':<30}")
    print("-" * 150)
    
    results = []
    
    # Metrics accumulators
    all_segments_used = []
    all_q_halt = []
    all_q_continue = []
    hrm_exact_matches = []
    hrm_f1_scores = []
    baseline_exact_matches = []
    baseline_f1_scores = []
    
    with torch.no_grad():
        for s in samples:
            input_ids = s['input_ids'].to(device)
            
            # Encode
            memory, src_mask = model.encode(input_ids)
            B, L, D = memory.shape
            zH, zL = model.hrm_core.init_state(B, L, device)
            
            # Run HRM reasoning with ACT
            segments_used = 0
            final_q_halt = 0.0
            final_q_continue = 0.0
            
            for m in range(4):  # Max segments
                zH, zL = model.hrm_core.forward_segment(zH, zL, memory, key_padding_mask=src_mask)
                q_probs = model.hrm_core.get_q_values(zH)
                final_q_halt = q_probs[0, 0].item()
                final_q_continue = q_probs[0, 1].item()
                segments_used = m + 1
                
                if final_q_halt > final_q_continue and m >= 1:
                    break
            
            all_segments_used.append(segments_used)
            all_q_halt.append(final_q_halt)
            all_q_continue.append(final_q_continue)
            
            enhanced_memory, enhanced_mask = model.prepare_enhanced_memory(memory, zH, src_mask)
            
            # Generate answer
            decoder_input = torch.tensor([[0]], device=device)
            gen_toks = []
            for _ in range(32):
                logits = model.generate_step(enhanced_memory, decoder_input, enhanced_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                if next_tok.item() == 1: break
                gen_toks.append(next_tok.item())
                decoder_input = torch.cat([decoder_input, next_tok], dim=1)
                
            hrm_ans = tokenizer.decode(gen_toks, skip_special_tokens=True)
            
            # Compute accuracy scores
            target = s['target']
            baseline = s['baseline']
            
            hrm_exact_matches.append(compute_exact_match(hrm_ans, target))
            hrm_f1_scores.append(compute_f1(hrm_ans, target))
            baseline_exact_matches.append(compute_exact_match(baseline, target))
            baseline_f1_scores.append(compute_f1(baseline, target))
            
            # Display
            q_disp = (s['question'][:47] + '..') if len(s['question']) > 47 else s['question']
            t_disp = (target[:27] + '..') if len(target) > 27 else target
            b_disp = (baseline[:27] + '..') if len(baseline) > 27 else baseline
            h_disp = (hrm_ans[:27] + '..') if len(hrm_ans) > 27 else hrm_ans
            
            print(f"{q_disp:<50} | {t_disp:<30} | {b_disp:<30} | {h_disp:<30}")
            
            results.append({
                'question': s['question'],
                'target': target,
                'baseline': baseline,
                'hrm': hrm_ans,
                'segments_used': segments_used,
                'q_halt': final_q_halt,
                'q_continue': final_q_continue,
            })
            
    print("-" * 150 + "\n")
    
    # Aggregate metrics
    hrm_metrics = {
        'avg_segments_used': sum(all_segments_used) / len(all_segments_used),
        'avg_q_halt': sum(all_q_halt) / len(all_q_halt),
        'avg_q_continue': sum(all_q_continue) / len(all_q_continue),
    }
    
    accuracy_metrics = {
        'hrm_exact_match': sum(hrm_exact_matches) / len(hrm_exact_matches),
        'hrm_f1': sum(hrm_f1_scores) / len(hrm_f1_scores),
        'baseline_exact_match': sum(baseline_exact_matches) / len(baseline_exact_matches),
        'baseline_f1': sum(baseline_f1_scores) / len(baseline_f1_scores),
    }
    
    return results, hrm_metrics, accuracy_metrics

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
        # Decode
        # Enhanced memory = Adapter(memory, zH)
        enhanced_memory, enhanced_mask = model.prepare_enhanced_memory(current_memory, zH, src_mask)
        
        logits = model.decode(enhanced_memory, labels, enhanced_mask)
        
        lm_loss = compute_loss(logits, labels)
        
        # Simplified Loss (Focus on LM for pretraining robustness first?)
        # Adding Q-loss if we want ACT. 
        # reusing logic from previous step:
        # For now, let's keep it simple: Just LM loss to verify T5 integration works.
        # We can add ACT loss back if it trains well.
        
        loss = lm_loss
        
        # Backward (accumulate gradients)
        loss.backward()
        
        total_loss += loss.item()
        segment_count += 1
        
    # Validation / Verify gradients?
    nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
    optimizer.step()
    optimizer.zero_grad()
    
    return {'loss': total_loss / segment_count}

def train_main():
    print("Initializing Training...")
    cfg = Config()
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    
    # 0. Comparison Setup
    validation_samples = get_baseline_samples(tokenizer, cfg)
    
    model = NanoHRMv3(tokenizer, cfg).to(cfg.DEVICE)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Curriculum Learning: Freeze T5 initially
    model.freeze_t5()
    print("T5 Backbone Frozen for initial training.")
    
    train_loader, val_loader = get_dataloaders(tokenizer, cfg)
    
    # Optimizer (Include all parameters, even frozen ones, so they are tracked when unfrozen)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    print("Starting Loop...")
    metrics_history = []
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    metrics_file = os.path.join(cfg.RESULTS_DIR, "metrics.json")
    
    for epoch in range(cfg.EPOCHS):
        # Curriculum: Unfreeze after N epochs (e.g., 2)
        if epoch == 2:
            model.unfreeze_t5()
            print("\n>>> Unfreezing T5 Backbone for Fine-tuning! <<<\n")
            
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        epoch_loss = 0
        steps = 0
        
        # Zero grad before batch
        optimizer.zero_grad()
        
        for batch in pbar:
            metrics = train_step(model, batch, optimizer, cfg, epoch)
            epoch_loss += metrics['loss']
            steps += 1
            pbar.set_postfix(metrics)
            
        avg_loss = epoch_loss / steps
        
        # Validation with comprehensive metrics
        comparisons, hrm_metrics, accuracy_metrics = run_validation_comparison(
            model, tokenizer, validation_samples, epoch
        )
        
        # Get model-level metrics (gate values, param counts)
        model_metrics = model.get_metrics()
        
        # Print summary
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Reasoning Gate: {model_metrics['reasoning_gate_effective']:.4f}")
        print(f"  Avg Segments: {hrm_metrics['avg_segments_used']:.2f}")
        print(f"  HRM EM/F1: {accuracy_metrics['hrm_exact_match']:.2%} / {accuracy_metrics['hrm_f1']:.2%}")
        print(f"  Baseline EM/F1: {accuracy_metrics['baseline_exact_match']:.2%} / {accuracy_metrics['baseline_f1']:.2%}")
        
        # Save comprehensive metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'loss': avg_loss,
            'model_metrics': model_metrics,
            'hrm_metrics': hrm_metrics,
            'accuracy': accuracy_metrics,
            'validation_samples': comparisons
        }
        metrics_history.append(epoch_metrics)
        with open(metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=4)
        
        if (epoch + 1) % 1 == 0:
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)

if __name__ == "__main__":
    train_main()
