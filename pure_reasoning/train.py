"""
Pure Reasoning Architecture - Training Script

Trains the pure reasoning model on QA datasets.
No next-token prediction, uses span prediction loss.
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
import argparse
from tqdm import tqdm

from .config import PureReasoningConfig
from .model import PureReasoningModel
from .dataset import get_span_dataloaders


def compute_span_f1(pred_start, pred_end, true_start, true_end):
    """Compute span-level F1 score"""
    # Get predicted and true spans
    pred_set = set(range(pred_start, pred_end + 1))
    true_set = set(range(true_start, true_end + 1))
    
    if len(pred_set) == 0 or len(true_set) == 0:
        return float(pred_set == true_set)
    
    intersection = pred_set & true_set
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(true_set)
    
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def run_validation(model, val_loader, config):
    """Run validation and compute metrics"""
    model.eval()
    device = config.DEVICE
    
    total_em = 0
    total_f1 = 0
    total_samples = 0
    per_dataset = {}
    all_steps = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            sources = batch['sources']
            
            outputs = model(input_ids, attention_mask, 
                          start_positions, end_positions)
            
            pred_starts = outputs['start_logits'].argmax(dim=-1)
            pred_ends = outputs['end_logits'].argmax(dim=-1)
            
            all_steps.append(outputs['supervision_steps'])
            
            for i in range(input_ids.size(0)):
                ps = pred_starts[i].item()
                pe = pred_ends[i].item()
                ts = start_positions[i].item()
                te = end_positions[i].item()
                source = sources[i]
                
                # Exact match
                em = float(ps == ts and pe == te)
                f1 = compute_span_f1(ps, pe, ts, te)
                
                total_em += em
                total_f1 += f1
                total_samples += 1
                
                # Per-dataset tracking
                if source not in per_dataset:
                    per_dataset[source] = {'em': 0, 'f1': 0, 'n': 0, 'steps': []}
                per_dataset[source]['em'] += em
                per_dataset[source]['f1'] += f1
                per_dataset[source]['n'] += 1
                per_dataset[source]['steps'].append(outputs['supervision_steps'])
    
    # Aggregate
    avg_em = total_em / total_samples if total_samples > 0 else 0
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0
    avg_steps = sum(all_steps) / len(all_steps) if all_steps else 0
    
    per_dataset_summary = {}
    for ds, metrics in per_dataset.items():
        per_dataset_summary[ds] = {
            'em': metrics['em'] / metrics['n'] if metrics['n'] > 0 else 0,
            'f1': metrics['f1'] / metrics['n'] if metrics['n'] > 0 else 0,
            'avg_steps': sum(metrics['steps']) / len(metrics['steps']) if metrics['steps'] else 0,
        }
    
    return {
        'exact_match': avg_em,
        'f1': avg_f1,
        'avg_steps': avg_steps,
        'per_dataset': per_dataset_summary,
    }


def train_epoch(model, train_loader, optimizer, scheduler, config, epoch):
    """Train for one epoch"""
    model.train()
    device = config.DEVICE
    
    total_span_loss = 0
    total_act_loss = 0
    total_steps = 0
    batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        outputs = model(
            input_ids, 
            attention_mask,
            start_positions=start_positions,
            end_positions=end_positions,
        )
        
        # Combined loss
        loss = outputs['span_loss'] + 0.1 * outputs['act_loss']
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_span_loss += outputs['span_loss'].item() if torch.is_tensor(outputs['span_loss']) else outputs['span_loss']
        total_act_loss += outputs['act_loss'].item() if torch.is_tensor(outputs['act_loss']) else outputs['act_loss']
        total_steps += outputs['supervision_steps']
        batches += 1
        
        pbar.set_postfix({
            'span': f"{outputs['span_loss']:.3f}" if torch.is_tensor(outputs['span_loss']) else f"{outputs['span_loss']:.3f}",
            'act': f"{outputs['act_loss']:.3f}" if torch.is_tensor(outputs['act_loss']) else f"{outputs['act_loss']:.3f}",
            'steps': outputs['supervision_steps'],
        })
    
    return {
        'span_loss': total_span_loss / batches,
        'act_loss': total_act_loss / batches,
        'avg_steps': total_steps / batches,
    }


def get_optimizer(model, config):
    """Create optimizer with separate Q-head learning rate"""
    q_head_params = list(model.reasoning.q_head.parameters())
    q_head_ids = set(id(p) for p in q_head_params)
    other_params = [p for p in model.parameters() if id(p) not in q_head_ids]
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': config.LEARNING_RATE},
        {'params': q_head_params, 'lr': config.LEARNING_RATE * config.Q_HEAD_LR_MULTIPLIER},
    ], weight_decay=config.WEIGHT_DECAY)
    
    return optimizer


def train_main(args=None):
    """Main training loop"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--samples', type=int, default=None)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--cpu', action='store_true', help='Force CPU even if CUDA available')
    args = parser.parse_args(args)
    
    print("=" * 60)
    print("Pure Reasoning Architecture Training")
    print("No T5, No Next-Token Prediction")
    print("=" * 60)
    
    config = PureReasoningConfig()
    
    # Override device if --cpu flag is set
    if args.cpu:
        print("Forcing CPU mode (--cpu flag)")
        config.DEVICE = "cpu"
    
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.samples:
        config.SAMPLES_PER_DATASET = args.samples
    
    # Create directories
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    # Load data
    train_loader, val_loader, tokenizer = get_span_dataloaders(config)
    
    # Create model
    print(f"\nUsing device: {config.DEVICE}")
    model = PureReasoningModel(config).to(config.DEVICE)
    metrics = model.get_metrics()
    print(f"\nModel Parameters: {metrics['total_params_M']:.2f}M")
    print(f"  Encoder: {metrics['encoder_params_M']:.2f}M")
    print(f"  Reasoning: {metrics['reasoning_params_M']:.2f}M")
    print(f"  Heads: {metrics['heads_params_M']:.2f}M")
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
    
    # Optimizer and scheduler
    optimizer = get_optimizer(model, config)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[config.LEARNING_RATE, config.LEARNING_RATE * config.Q_HEAD_LR_MULTIPLIER],
        total_steps=total_steps,
        pct_start=0.1,
    )
    
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    print(f"Q-head LR: {config.LEARNING_RATE * config.Q_HEAD_LR_MULTIPLIER:.2e}\n")
    
    metrics_history = []
    metrics_file = os.path.join(config.RESULTS_DIR, "pure_reasoning_metrics.json")
    
    for epoch in range(start_epoch, config.EPOCHS):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, config, epoch)
        
        # Validate
        val_metrics = run_validation(model, val_loader, config)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Span Loss: {train_metrics['span_loss']:.4f}, ACT Loss: {train_metrics['act_loss']:.4f}")
        print(f"  Avg Steps: {train_metrics['avg_steps']:.2f}")
        print(f"  Val EM: {val_metrics['exact_match']:.2%}, Val F1: {val_metrics['f1']:.2%}")
        print("  Per-Dataset:")
        for ds, m in val_metrics['per_dataset'].items():
            print(f"    {ds}: EM={m['em']:.2%}, F1={m['f1']:.2%}, Steps={m['avg_steps']:.1f}")
        
        # Save metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
        }
        metrics_history.append(epoch_metrics)
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        
        # Save model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config.to_dict(),
        }, config.MODEL_SAVE_PATH)
        
        print()
    
    print("Training Complete!")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_main()
