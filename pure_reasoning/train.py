"""
Pure Reasoning Architecture - Training Script

Trains the pure reasoning model on QA datasets.
Uses cross-entropy loss for text generation.
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
import random
from tqdm import tqdm

from .config import PureReasoningConfig
from .model import PureReasoningModel
from .dataset import get_dataloaders


def compute_text_metrics(pred_text, gold_text):
    """
    Compute exact match and F1 for text strings.
    """
    # Normalize: lowercase, strip
    pred_text = pred_text.lower().strip()
    gold_text = gold_text.lower().strip()
    
    # Exact match
    em = float(pred_text == gold_text)
    
    # F1: token overlap
    pred_tokens = pred_text.split()
    gold_tokens = gold_text.split()
    
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return em, float(pred_text == gold_text)
    
    common = set(pred_tokens) & set(gold_tokens)
    if len(common) == 0:
        return em, 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    
    return em, f1


def run_validation(model, val_loader, tokenizer, config, num_samples_to_log=5):
    """Run validation and compute metrics"""
    model.eval()
    device = config.DEVICE
    
    total_em = 0
    total_f1 = 0
    total_samples = 0
    per_dataset = {}
    all_steps = []
    
    # Collect sample generations for logging
    sample_generations = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sources = batch['sources']
            raw_questions = batch['raw_questions']
            raw_answers = batch['raw_answers']
            
            # Generate predictions
            generated_ids = model.generate(input_ids, attention_mask, tokenizer, 
                                           max_len=config.MAX_ANSWER_LEN)
            
            for i in range(input_ids.size(0)):
                # Decode prediction
                pred_ids = generated_ids[i].tolist()
                # Remove special tokens for display
                pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
                gold_text = raw_answers[i]
                
                em, f1 = compute_text_metrics(pred_text, gold_text)
                
                total_em += em
                total_f1 += f1
                total_samples += 1
                
                source = sources[i]
                
                # Per-dataset tracking
                if source not in per_dataset:
                    per_dataset[source] = {'em': 0, 'f1': 0, 'n': 0}
                per_dataset[source]['em'] += em
                per_dataset[source]['f1'] += f1
                per_dataset[source]['n'] += 1
                
                # Collect samples for logging
                if len(sample_generations) < num_samples_to_log * 3:  # Collect more, sample later
                    sample_generations.append({
                        'source': source,
                        'question': raw_questions[i][:200],  # Truncate for readability
                        'gold': gold_text,
                        'predicted': pred_text[:200],
                    })
    
    # Sample random generations to log
    if len(sample_generations) > num_samples_to_log:
        sample_generations = random.sample(sample_generations, num_samples_to_log)
    
    # Aggregate
    avg_em = total_em / total_samples if total_samples > 0 else 0
    avg_f1 = total_f1 / total_samples if total_samples > 0 else 0
    
    per_dataset_summary = {}
    for ds, metrics in per_dataset.items():
        per_dataset_summary[ds] = {
            'em': metrics['em'] / metrics['n'] if metrics['n'] > 0 else 0,
            'f1': metrics['f1'] / metrics['n'] if metrics['n'] > 0 else 0,
        }
    
    return {
        'exact_match': avg_em,
        'f1': avg_f1,
        'per_dataset': per_dataset_summary,
        'sample_generations': sample_generations,
    }


def train_epoch(model, train_loader, optimizer, scheduler, config, epoch):
    """Train for one epoch with PonderNet losses"""
    model.train()
    device = config.DEVICE
    
    total_loss = 0
    total_rec_loss = 0
    total_reg_loss = 0
    total_expected_steps = 0
    batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch in pbar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        decoder_input_ids = batch['decoder_input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids, 
            attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )
        
        # PonderNet loss already includes reconstruction + regularization
        loss = outputs['loss']
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()
        
        total_loss += loss.item()
        total_rec_loss += outputs.get('rec_loss', 0)
        total_reg_loss += outputs.get('reg_loss', 0)
        total_expected_steps += outputs.get('expected_steps', outputs['supervision_steps'])
        batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.3f}",
            'steps': f"{outputs.get('expected_steps', 0):.2f}",
        })
    
    return {
        'loss': total_loss / batches,
        'rec_loss': total_rec_loss / batches,
        'reg_loss': total_reg_loss / batches,
        'avg_steps': total_expected_steps / batches,
    }


def get_optimizer(model, config):
    """Create optimizer with separate halting network learning rate"""
    halting_params = list(model.reasoning.halting_net.parameters())
    halting_ids = set(id(p) for p in halting_params)
    other_params = [p for p in model.parameters() if id(p) not in halting_ids]
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': config.LEARNING_RATE},
        {'params': halting_params, 'lr': config.LEARNING_RATE * config.HALTING_LR_MULTIPLIER},
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
    print("Pure Reasoning Architecture Training (Generative)")
    print("Encoder + ReasoningCore + Decoder")
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
    train_loader, val_loader, tokenizer = get_dataloaders(config)
    
    # Create model
    print(f"\nUsing device: {config.DEVICE}")
    model = PureReasoningModel(config).to(config.DEVICE)
    metrics = model.get_metrics()
    print(f"\nModel Parameters: {metrics['total_params_M']:.2f}M")
    print(f"  Encoder: {metrics['encoder_params_M']:.2f}M")
    print(f"  Reasoning: {metrics['reasoning_params_M']:.2f}M")
    print(f"  Decoder: {metrics['decoder_params_M']:.2f}M")
    
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
        max_lr=[config.LEARNING_RATE, config.LEARNING_RATE * config.HALTING_LR_MULTIPLIER],
        total_steps=total_steps,
        pct_start=0.1,
    )
    
    print(f"\nStarting training for {config.EPOCHS} epochs...")
    print(f"Halting Network LR: {config.LEARNING_RATE * config.HALTING_LR_MULTIPLIER:.2e}\n")
    
    metrics_history = []
    metrics_file = os.path.join(config.RESULTS_DIR, "pure_reasoning_metrics.json")
    
    for epoch in range(start_epoch, config.EPOCHS):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, config, epoch)
        
        # Validate
        val_metrics = run_validation(model, val_loader, tokenizer, config)
        
        # Print summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Avg Steps: {train_metrics['avg_steps']:.2f}")
        print(f"  Val EM: {val_metrics['exact_match']:.2%}, Val F1: {val_metrics['f1']:.2%}")
        print("  Per-Dataset:")
        for ds, m in val_metrics['per_dataset'].items():
            print(f"    {ds}: EM={m['em']:.2%}, F1={m['f1']:.2%}")
        
        # Print sample generations
        print("\n  Sample Generations:")
        for sample in val_metrics['sample_generations'][:3]:
            print(f"    [{sample['source']}] Q: {sample['question'][:80]}...")
            print(f"      Gold: {sample['gold']}")
            print(f"      Pred: {sample['predicted']}")
        
        # Save metrics
        epoch_metrics = {
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': {
                'exact_match': val_metrics['exact_match'],
                'f1': val_metrics['f1'],
                'per_dataset': val_metrics['per_dataset'],
                'sample_generations': val_metrics['sample_generations'],
            },
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
