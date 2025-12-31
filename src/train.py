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

def get_baseline_samples(tokenizer, config, num_samples=None):
    """
    Load Standard T5, generate answers for validation samples, then unload.
    Returns: list of dicts {question, context, target, baseline_answer, source}
    Explicitly samples from EACH dataset to ensure balanced validation.
    """
    if num_samples is None:
        num_samples = config.NUM_VAL_SAMPLES
    
    # Calculate samples per dataset (ensure we get from each)
    samples_per_ds = max(1, num_samples // len(config.DATASETS))
    
    print(f"Generating T5 Baseline for {samples_per_ds} samples x {len(config.DATASETS)} datasets...")
    device = config.DEVICE
    
    # Import here to avoid circular import
    from .dataset import UnifiedQADataset
    
    samples = []
    
    # Load T5 (Frozen/Eval)
    t5_model = T5ForConditionalGeneration.from_pretrained(config.TOKENIZER_NAME).to(device)
    t5_model.eval()
    
    with torch.no_grad():
        for dataset_name in config.DATASETS:
            print(f"  Loading {dataset_name} validation samples...")
            
            # Load a few samples from this specific dataset
            ds = UnifiedQADataset(tokenizer, dataset_name, 'validation', max_samples=samples_per_ds)
            
            for i in range(min(samples_per_ds, len(ds))):
                item = ds[i]
                input_ids = item['input_ids'].unsqueeze(0).to(device)
                
                # Generate T5 Answer
                gen_out = t5_model.generate(input_ids, max_new_tokens=64)
                base_ans = tokenizer.decode(gen_out[0], skip_special_tokens=True)
                
                samples.append({
                    'input_ids': input_ids,
                    'question': item['raw_question'],
                    'target': item['raw_answer'],
                    'baseline': base_ans,
                    'source': item['source'],
                    'difficulty': item['difficulty'],
                })
    
    print(f"Baseline Generated for {len(samples)} samples. Unloading T5...")
    del t5_model
    torch.cuda.empty_cache()
    return samples

def run_validation_comparison(model, tokenizer, samples, epoch):
    """
    Run HRM on the samples, collect metrics, and print comparison table.
    Returns: (validation_samples, hrm_metrics, accuracy_metrics)
    Now includes per-dataset tracking for multi-hop analysis.
    """
    model.eval()
    device = next(model.parameters()).device
    
    print(f"\n\n=== Validation Comparison (Epoch {epoch+1}) ===")
    print(f"{'Source':<10} | {'Question':<40} | {'Target':<25} | {'T5':<25} | {'HRM':<25} | {'Segs':<4}")
    print("-" * 140)
    
    results = []
    
    # Metrics accumulators (overall and per-dataset)
    all_segments_used = []
    all_q_halt = []
    all_q_continue = []
    hrm_exact_matches = []
    hrm_f1_scores = []
    baseline_exact_matches = []
    baseline_f1_scores = []
    
    # Per-dataset tracking
    per_dataset_metrics = {}
    
    with torch.no_grad():
        for s in samples:
            input_ids = s['input_ids'].to(device)
            source = s.get('source', 'unknown')
            
            # Initialize per-dataset tracking if needed
            if source not in per_dataset_metrics:
                per_dataset_metrics[source] = {
                    'segments': [], 'hrm_em': [], 'hrm_f1': [],
                    'baseline_em': [], 'baseline_f1': []
                }
            
            # Encode
            memory, src_mask = model.encode(input_ids)
            B, L, D = memory.shape
            zH, zL = model.hrm_core.init_state(B, L, device)
            
            # Run HRM reasoning with proper ACT (Graves 2016)
            cumulative_halt = 0.0
            epsilon = 0.01
            segments_used = 0
            halting_weights = []
            zH_states = []
            
            for m in range(4):  # Max segments during inference
                zH, zL = model.hrm_core.forward_segment(zH, zL, memory, key_padding_mask=src_mask)
                p_halt = model.hrm_core.get_q_values(zH)[0, 0].item()
                
                remainder = 1.0 - cumulative_halt
                segments_used = m + 1
                
                if remainder < epsilon:
                    break
                
                effective_halt = min(p_halt, remainder)
                cumulative_halt += effective_halt
                
                halting_weights.append(effective_halt)
                zH_states.append(zH.clone())
                
                if cumulative_halt >= 1.0 - epsilon:
                    break
            
            # Compute weighted combination of states
            if halting_weights:
                weight_sum = sum(halting_weights)
                if weight_sum > 0:
                    normalized_weights = [w / weight_sum for w in halting_weights]
                    final_zH = sum(w * st for w, st in zip(normalized_weights, zH_states))
                else:
                    final_zH = zH_states[-1]
            else:
                final_zH = zH
            
            final_q_halt = p_halt
            final_q_continue = 1.0 - p_halt
            
            all_segments_used.append(segments_used)
            all_q_halt.append(final_q_halt)
            all_q_continue.append(final_q_continue)
            
            enhanced_memory, enhanced_mask = model.prepare_enhanced_memory(memory, final_zH, src_mask)
            
            # Generate answer
            decoder_input = torch.tensor([[0]], device=device)
            gen_toks = []
            for _ in range(64):  # Increased for longer answers
                logits = model.generate_step(enhanced_memory, decoder_input, enhanced_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                if next_tok.item() == 1: break
                gen_toks.append(next_tok.item())
                decoder_input = torch.cat([decoder_input, next_tok], dim=1)
                
            hrm_ans = tokenizer.decode(gen_toks, skip_special_tokens=True)
            
            # Compute accuracy scores
            target = s['target']
            baseline = s['baseline']
            
            hrm_em = compute_exact_match(hrm_ans, target)
            hrm_f1 = compute_f1(hrm_ans, target)
            base_em = compute_exact_match(baseline, target)
            base_f1 = compute_f1(baseline, target)
            
            hrm_exact_matches.append(hrm_em)
            hrm_f1_scores.append(hrm_f1)
            baseline_exact_matches.append(base_em)
            baseline_f1_scores.append(base_f1)
            
            # Track per-dataset
            per_dataset_metrics[source]['segments'].append(segments_used)
            per_dataset_metrics[source]['hrm_em'].append(hrm_em)
            per_dataset_metrics[source]['hrm_f1'].append(hrm_f1)
            per_dataset_metrics[source]['baseline_em'].append(base_em)
            per_dataset_metrics[source]['baseline_f1'].append(base_f1)
            
            # Display (compact format)
            q_disp = (s['question'][:37] + '..') if len(s['question']) > 37 else s['question']
            t_disp = (target[:22] + '..') if len(target) > 22 else target
            b_disp = (baseline[:22] + '..') if len(baseline) > 22 else baseline
            h_disp = (hrm_ans[:22] + '..') if len(hrm_ans) > 22 else hrm_ans
            
            print(f"{source:<10} | {q_disp:<40} | {t_disp:<25} | {b_disp:<25} | {h_disp:<25} | {segments_used:<4}")
            
            results.append({
                'question': s['question'],
                'target': target,
                'baseline': baseline,
                'hrm': hrm_ans,
                'segments_used': segments_used,
                'q_halt': final_q_halt,
                'q_continue': final_q_continue,
                'source': source,
            })
            
    print("-" * 140)
    
    # Print per-dataset summary
    print("\n=== Per-Dataset Summary ===")
    for ds_name, metrics in per_dataset_metrics.items():
        avg_segs = sum(metrics['segments']) / len(metrics['segments']) if metrics['segments'] else 0
        avg_hrm_em = sum(metrics['hrm_em']) / len(metrics['hrm_em']) if metrics['hrm_em'] else 0
        avg_base_em = sum(metrics['baseline_em']) / len(metrics['baseline_em']) if metrics['baseline_em'] else 0
        print(f"  {ds_name}: Segs={avg_segs:.1f}, HRM_EM={avg_hrm_em:.0%}, Baseline_EM={avg_base_em:.0%}")
    print()
    
    # Aggregate metrics
    hrm_metrics = {
        'avg_segments_used': sum(all_segments_used) / len(all_segments_used),
        'avg_q_halt': sum(all_q_halt) / len(all_q_halt),
        'avg_q_continue': sum(all_q_continue) / len(all_q_continue),
        'per_dataset': {k: sum(v['segments'])/len(v['segments']) for k, v in per_dataset_metrics.items()},
    }
    
    accuracy_metrics = {
        'hrm_exact_match': sum(hrm_exact_matches) / len(hrm_exact_matches),
        'hrm_f1': sum(hrm_f1_scores) / len(hrm_f1_scores),
        'baseline_exact_match': sum(baseline_exact_matches) / len(baseline_exact_matches),
        'baseline_f1': sum(baseline_f1_scores) / len(baseline_f1_scores),
    }
    
    return results, hrm_metrics, accuracy_metrics

def train_step(model, batch, optimizer, config, epoch):
    """
    Training step with proper ACT (Graves 2016):
    - Accumulate halting probabilities until they sum to ~1
    - Weight intermediate states by halting probability
    - Add ponder cost regularizer to encourage efficiency
    """
    device = config.DEVICE
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    # 1. Encode
    memory, src_mask = model.encode(input_ids)
    
    B, L, D = memory.shape
    zH, zL = model.hrm_core.init_state(B, L, device)
    
    M_max = config.MAX_SEGMENTS
    epsilon = config.ACT_EPSILON
    tau = config.ACT_PONDER_COST_TAU
    
    # ACT state tracking (per batch element)
    cumulative_halt = torch.zeros(B, device=device)  # [B]
    halting_weights = []  # List of [B] tensors
    zH_states = []  # List of [B, L, D] tensors
    lm_losses = []  # LM loss at each step (for weighted combination)
    
    halting_probs_log = []  # For logging
    steps_taken = torch.zeros(B, device=device)  # Track steps per sample
    
    for m in range(M_max):
        zH_in = zH.detach()
        zL_in = zL.detach()
        
        # 1-step gradient approximation for memory
        if m == 0:
            current_memory = memory
        else:
            current_memory = memory.detach()
        
        # Forward HRM segment
        zH, zL = model.hrm_core.forward_segment(zH_in, zL_in, current_memory, key_padding_mask=src_mask)
        
        # Get halting probability [B, 1] -> [B]
        p_halt = model.hrm_core.get_q_values(zH).squeeze(-1)
        halting_probs_log.append(p_halt.mean().item())
        
        # Compute remainder (how much probability budget left)
        remainder = 1.0 - cumulative_halt  # [B]
        
        # Mask: which samples are still active (haven't halted yet)
        active_mask = (remainder > epsilon).float()  # [B]
        
        # Clamp p_halt to not exceed remainder
        effective_halt = torch.min(p_halt, remainder) * active_mask  # [B]
        
        # Accumulate halting probability
        cumulative_halt = cumulative_halt + effective_halt
        
        # Track steps
        steps_taken = steps_taken + active_mask
        
        # Store weighted state
        halting_weights.append(effective_halt)
        zH_states.append(zH)
        
        # Compute LM loss for this step
        enhanced_memory, enhanced_mask = model.prepare_enhanced_memory(current_memory, zH, src_mask)
        logits = model.decode(enhanced_memory, labels, enhanced_mask)
        step_lm_loss = compute_loss(logits, labels)
        lm_losses.append(step_lm_loss)
        
        # Check if all samples have halted
        if (remainder <= epsilon).all():
            break
    
    # ============== Combine States (Weighted Average) ==============
    # Stack weights: [M, B] -> normalize to sum to 1 per sample
    weight_stack = torch.stack(halting_weights, dim=0)  # [M, B]
    weight_sum = weight_stack.sum(dim=0, keepdim=True).clamp(min=1e-8)  # [1, B]
    normalized_weights = weight_stack / weight_sum  # [M, B]
    
    # Weighted sum of states: sum over M of weight * state
    # zH_states: list of [B, L, D], normalized_weights: [M, B]
    final_zH = torch.zeros_like(zH_states[0])
    for w, s in zip(normalized_weights, zH_states):
        # w: [B], s: [B, L, D] -> expand w to [B, 1, 1] for broadcasting
        final_zH = final_zH + w.unsqueeze(-1).unsqueeze(-1) * s
    
    # ============== Final LM Loss (on combined state) ==============
    enhanced_memory, enhanced_mask = model.prepare_enhanced_memory(memory, final_zH, src_mask)
    logits = model.decode(enhanced_memory, labels, enhanced_mask)
    final_lm_loss = compute_loss(logits, labels)
    
    # ============== Ponder Cost (Graves 2016) ==============
    # ρ = N + R where N = number of steps, R = remainder (1 - cumulative)
    # This encourages the model to halt quickly
    ponder_cost = steps_taken.mean() + (1.0 - cumulative_halt).mean()
    act_loss = tau * ponder_cost
    
    # ============== Total Loss ==============
    total_loss = final_lm_loss + act_loss
    
    # Backward
    total_loss.backward()
    
    nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
    optimizer.step()
    optimizer.zero_grad()
    
    return {
        'loss': final_lm_loss.item(),
        'ponder': ponder_cost.item(),
        'avg_p_halt': sum(halting_probs_log) / len(halting_probs_log) if halting_probs_log else 0,
        'avg_steps': steps_taken.mean().item(),
    }

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
