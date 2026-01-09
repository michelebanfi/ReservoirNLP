"""
TRM (Tiny Recursion Model) Training Script

Based on arXiv:2510.04871 - "Less is More: Recursive Reasoning with Tiny Networks"

Key differences from HRM training (train.py):
1. Simpler ACT: no extra forward pass, just BCE loss on (y_hat == y_true)
2. EMA on weights for better generalization
3. Deep supervision: N_supervision steps with early stopping
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

import torch
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

import torch.nn as nn
import torch.nn.functional as F
import json
import re
import string
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

from .config_trm import TRMConfig
from .model_trm import TinyRecursionModel, EMAModel
from .dataset import get_dataloaders, UnifiedQADataset


def compute_loss(logits, labels):
    """Cross-entropy loss with label smoothing"""
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))


# ============== Scoring Utilities ==============

def normalize_text(text):
    """Normalize text for QA evaluation"""
    text = text.lower()
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ' '.join(text.split())
    return text


def compute_exact_match(prediction, target):
    return float(normalize_text(prediction) == normalize_text(target))


def compute_f1(prediction, target):
    pred_tokens = normalize_text(prediction).split()
    target_tokens = normalize_text(target).split()
    
    if len(pred_tokens) == 0 or len(target_tokens) == 0:
        return float(pred_tokens == target_tokens)
    
    common = set(pred_tokens) & set(target_tokens)
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(target_tokens)
    return 2 * precision * recall / (precision + recall)


# ============== Baseline Generation ==============

def get_baseline_samples(tokenizer, config, num_samples=None):
    """Generate T5 baseline answers for validation comparison"""
    if num_samples is None:
        num_samples = config.NUM_VAL_SAMPLES
    
    samples_per_ds = max(1, num_samples // len(config.DATASETS))
    print(f"Generating T5 Baseline for {samples_per_ds} samples x {len(config.DATASETS)} datasets...")
    device = config.DEVICE
    
    samples = []
    t5_model = T5ForConditionalGeneration.from_pretrained(config.TOKENIZER_NAME).to(device)
    t5_model.eval()
    
    with torch.no_grad():
        for dataset_name in config.DATASETS:
            print(f"  Loading {dataset_name} validation samples...")
            ds = UnifiedQADataset(tokenizer, dataset_name, 'validation', max_samples=samples_per_ds)
            
            for i in range(min(samples_per_ds, len(ds))):
                item = ds[i]
                input_ids = item['input_ids'].unsqueeze(0).to(device)
                
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


# ============== Validation ==============

def run_validation_comparison(model, tokenizer, samples, epoch, ema=None):
    """
    Run TRM on validation samples and compare with T5 baseline.
    Optionally use EMA weights for evaluation.
    """
    # Apply EMA weights if available
    if ema is not None:
        ema.apply_shadow(model)
    
    model.eval()
    device = next(model.parameters()).device
    config = model.config
    
    print(f"\n=== Validation Comparison (Epoch {epoch+1}) ===")
    print(f"{'Source':<10} | {'Question':<40} | {'Target':<25} | {'T5':<25} | {'TRM':<25} | {'Steps':<5}")
    print("-" * 145)
    
    results = []
    all_supervision_steps = []
    trm_exact_matches, trm_f1_scores = [], []
    baseline_exact_matches, baseline_f1_scores = [], []
    per_dataset_metrics = {}
    
    with torch.no_grad():
        for s in samples:
            input_ids = s['input_ids'].to(device)
            source = s.get('source', 'unknown')
            
            if source not in per_dataset_metrics:
                per_dataset_metrics[source] = {
                    'steps': [], 'trm_em': [], 'trm_f1': [],
                    'baseline_em': [], 'baseline_f1': []
                }
            
            # 1. Encode input
            memory, src_mask = model.encode(input_ids)
            B, L, D = memory.shape
            
            # 2. Initialize y, z
            y, z = model.trm_core.init_state(B, L, device)
            
            # 3. Deep supervision steps
            supervision_steps = 0
            for step in range(config.N_SUPERVISION):
                (y, z), y_out, q_hat = model.trm_core.deep_recursion(
                    memory, y, z, key_padding_mask=src_mask
                )
                supervision_steps = step + 1
                
                # Early stopping if model predicts answer is correct
                # Only after minimum steps (matches training logic)
                if step >= config.MIN_SUPERVISION_STEPS - 1 and q_hat.mean().item() > 0.5:
                    break
            
            all_supervision_steps.append(supervision_steps)
            
            # 4. Generate answer using final y
            enhanced_memory, enhanced_mask = model.prepare_enhanced_memory(memory, y_out, src_mask)
            
            decoder_input = torch.tensor([[0]], device=device)
            gen_toks = []
            for _ in range(64):
                logits = model.generate_step(enhanced_memory, decoder_input, enhanced_mask)
                next_tok = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                if next_tok.item() == 1:  # EOS
                    break
                gen_toks.append(next_tok.item())
                decoder_input = torch.cat([decoder_input, next_tok], dim=1)
            
            trm_ans = tokenizer.decode(gen_toks, skip_special_tokens=True)
            
            # 5. Compute scores
            target = s['target']
            baseline = s['baseline']
            
            trm_em = compute_exact_match(trm_ans, target)
            trm_f1 = compute_f1(trm_ans, target)
            base_em = compute_exact_match(baseline, target)
            base_f1 = compute_f1(baseline, target)
            
            trm_exact_matches.append(trm_em)
            trm_f1_scores.append(trm_f1)
            baseline_exact_matches.append(base_em)
            baseline_f1_scores.append(base_f1)
            
            # Per-dataset tracking
            per_dataset_metrics[source]['steps'].append(supervision_steps)
            per_dataset_metrics[source]['trm_em'].append(trm_em)
            per_dataset_metrics[source]['trm_f1'].append(trm_f1)
            per_dataset_metrics[source]['baseline_em'].append(base_em)
            per_dataset_metrics[source]['baseline_f1'].append(base_f1)
            
            # Display
            q_disp = (s['question'][:37] + '..') if len(s['question']) > 37 else s['question']
            t_disp = (target[:22] + '..') if len(target) > 22 else target
            b_disp = (baseline[:22] + '..') if len(baseline) > 22 else baseline
            h_disp = (trm_ans[:22] + '..') if len(trm_ans) > 22 else trm_ans
            
            print(f"{source:<10} | {q_disp:<40} | {t_disp:<25} | {b_disp:<25} | {h_disp:<25} | {supervision_steps:<5}")
            
            results.append({
                'question': s['question'],
                'target': target,
                'baseline': baseline,
                'trm': trm_ans,
                'supervision_steps': supervision_steps,
                'source': source,
            })
    
    print("-" * 145)
    
    # Per-dataset summary
    print("\n=== Per-Dataset Summary ===")
    for ds_name, metrics in per_dataset_metrics.items():
        avg_steps = sum(metrics['steps']) / len(metrics['steps']) if metrics['steps'] else 0
        avg_trm_em = sum(metrics['trm_em']) / len(metrics['trm_em']) if metrics['trm_em'] else 0
        avg_base_em = sum(metrics['baseline_em']) / len(metrics['baseline_em']) if metrics['baseline_em'] else 0
        print(f"  {ds_name}: Steps={avg_steps:.1f}, TRM_EM={avg_trm_em:.0%}, Baseline_EM={avg_base_em:.0%}")
    print()
    
    # Restore original weights
    if ema is not None:
        ema.restore(model)
    
    # Aggregate metrics
    trm_metrics = {
        'avg_supervision_steps': sum(all_supervision_steps) / len(all_supervision_steps),
        'per_dataset': {k: sum(v['steps'])/len(v['steps']) for k, v in per_dataset_metrics.items()},
    }
    
    accuracy_metrics = {
        'trm_exact_match': sum(trm_exact_matches) / len(trm_exact_matches),
        'trm_f1': sum(trm_f1_scores) / len(trm_f1_scores),
        'baseline_exact_match': sum(baseline_exact_matches) / len(baseline_exact_matches),
        'baseline_f1': sum(baseline_f1_scores) / len(baseline_f1_scores),
    }
    
    return results, trm_metrics, accuracy_metrics


# ============== Training Step ==============

def train_step(model, batch, optimizer, config, ema=None):
    """
    TRM Training Step (Algorithm 3 from paper):
    
    For each sample:
    1. Initialize y, z
    2. For up to N_supervision steps:
       a. Run deep_recursion (T-1 no-grad + 1 with-grad)
       b. Compute LM loss + ACT loss (simple BCE on q_hat vs correctness)
       c. Backward + step
       d. Update EMA
       e. Early stop if q_hat > 0.5
    
    Key simplification: No extra forward pass for ACT (vs HRM)
    """
    device = config.DEVICE
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    
    # Get initial memory shape (needed for init_state)
    with torch.no_grad():
        memory_init, src_mask = model.encode(input_ids)
        B, L, D = memory_init.shape
    
    # Initialize y, z
    y, z = model.trm_core.init_state(B, L, device)
    
    total_lm_loss = 0.0
    total_act_loss = 0.0
    supervision_steps = 0
    
    # Deep Supervision Loop
    for step in range(config.N_SUPERVISION):
        # Re-encode at each step to get fresh computation graph
        # This is necessary when T5 is unfrozen, as backward() frees the graph
        memory, src_mask = model.encode(input_ids)
        optimizer.zero_grad()
        
        # Deep recursion: updates (y, z) with T-1 no-grad + 1 with-grad
        (y, z), y_out, q_hat = model.trm_core.deep_recursion(
            memory, y, z, key_padding_mask=src_mask
        )
        
        # Compute LM loss
        enhanced_memory, enhanced_mask = model.prepare_enhanced_memory(memory, y_out, src_mask)
        logits = model.decode(enhanced_memory, labels, enhanced_mask)
        lm_loss = compute_loss(logits, labels)
        
        # Compute ACT loss (simple BCE)
        # q_hat predicts: is the current answer correct?
        with torch.no_grad():
            predictions = logits.argmax(dim=-1)  # [B, L]
            # Simple correctness: check if first few tokens match
            # More sophisticated: compute EM on full decoded sequence
            correct_mask = (labels != -100)
            correct = ((predictions == labels) | ~correct_mask).all(dim=1).float()  # [B]
        
        act_loss = F.binary_cross_entropy(q_hat.squeeze(-1), correct)
        
        # Total loss
        total_loss = lm_loss + act_loss
        
        # Backward
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        optimizer.step()
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        total_lm_loss += lm_loss.item()
        total_act_loss += act_loss.item()
        supervision_steps = step + 1
        
        # Early stopping: if model predicts answer is correct
        # Only allow halting after minimum supervision steps
        if step >= config.MIN_SUPERVISION_STEPS - 1 and q_hat.mean().item() > 0.5:
            break
    
    return {
        'lm_loss': total_lm_loss / supervision_steps,
        'act_loss': total_act_loss / supervision_steps,
        'supervision_steps': supervision_steps,
    }


# ============== Main Training Loop ==============

def train_main():
    print("=" * 60)
    print("TRM (Tiny Recursion Model) Training")
    print("Based on arXiv:2510.04871")
    print("=" * 60)
    
    config = TRMConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    # Get baseline samples for comparison
    validation_samples = get_baseline_samples(tokenizer, config)
    
    # Create model
    model = TinyRecursionModel(tokenizer, config).to(config.DEVICE)
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    model_metrics = model.get_metrics()
    print(f"  T5: {model_metrics['t5_params_M']:.2f}M")
    print(f"  TRM Core: {model_metrics['trm_params_M']:.2f}M")
    
    # Initialize EMA
    ema = None
    if config.USE_EMA:
        print(f"\nEMA enabled with decay={config.EMA_DECAY}")
        ema = EMAModel(model, decay=config.EMA_DECAY)
    
    # Curriculum Learning: Freeze T5 initially
    if config.FREEZE_T5_EPOCHS > 0:
        model.freeze_t5()
        print(f"T5 Frozen for first {config.FREEZE_T5_EPOCHS} epochs.")
    
    # Data loaders
    train_loader, val_loader = get_dataloaders(tokenizer, config)
    
    # Optimizer with separate Q-head learning rate
    # Q-head learns slower to prevent dominating after T5 unfreezing
    q_head_params = list(model.trm_core.q_head.parameters())
    q_head_ids = set(id(p) for p in q_head_params)
    other_params = [p for p in model.parameters() if id(p) not in q_head_ids]
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': config.LEARNING_RATE},
        {'params': q_head_params, 'lr': config.LEARNING_RATE * config.Q_HEAD_LR_MULTIPLIER},
    ], weight_decay=config.WEIGHT_DECAY)
    print(f"Optimizer: Q-head LR = {config.LEARNING_RATE * config.Q_HEAD_LR_MULTIPLIER:.2e} ({config.Q_HEAD_LR_MULTIPLIER}x main)")
    
    print("\nStarting Training...")
    metrics_history = []
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    metrics_file = os.path.join(config.RESULTS_DIR, "trm_metrics.json")
    
    for epoch in range(config.EPOCHS):
        # Curriculum: Unfreeze T5 after initial epochs
        if epoch == config.FREEZE_T5_EPOCHS and config.FREEZE_T5_EPOCHS > 0:
            model.unfreeze_t5()
            print(f"\n>>> Unfreezing T5 at epoch {epoch+1} <<<\n")
            # Re-init EMA with unfrozen params
            if config.USE_EMA:
                ema = EMAModel(model, decay=config.EMA_DECAY)
        
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        epoch_lm_loss = 0
        epoch_act_loss = 0
        epoch_steps = 0
        batches = 0
        
        for batch in pbar:
            metrics = train_step(model, batch, optimizer, config, ema)
            
            epoch_lm_loss += metrics['lm_loss']
            epoch_act_loss += metrics['act_loss']
            epoch_steps += metrics['supervision_steps']
            batches += 1
            
            pbar.set_postfix({
                'lm': f"{metrics['lm_loss']:.3f}",
                'act': f"{metrics['act_loss']:.3f}",
                'steps': metrics['supervision_steps'],
            })
        
        avg_lm_loss = epoch_lm_loss / batches
        avg_act_loss = epoch_act_loss / batches
        avg_steps = epoch_steps / batches
        
        # Validation (using EMA weights if available)
        comparisons, trm_metrics, accuracy_metrics = run_validation_comparison(
            model, tokenizer, validation_samples, epoch, ema
        )
        
        # Print Summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  LM Loss: {avg_lm_loss:.4f}, ACT Loss: {avg_act_loss:.4f}")
        print(f"  Avg Supervision Steps: {avg_steps:.2f}")
        print(f"  TRM EM/F1: {accuracy_metrics['trm_exact_match']:.2%} / {accuracy_metrics['trm_f1']:.2%}")
        print(f"  Baseline EM/F1: {accuracy_metrics['baseline_exact_match']:.2%} / {accuracy_metrics['baseline_f1']:.2%}")
        
        # Save metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'lm_loss': avg_lm_loss,
            'act_loss': avg_act_loss,
            'avg_supervision_steps': avg_steps,
            'trm_metrics': trm_metrics,
            'accuracy': accuracy_metrics,
            'validation_samples': comparisons,
        }
        metrics_history.append(epoch_metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        
        # Save model
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'ema_shadow': ema.shadow if ema else None,
        }, config.MODEL_SAVE_PATH)
    
    print("\nTraining Complete!")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_main()
