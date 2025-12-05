"""
Symmetric Activation Experiment - Fractal Symmetry Hypothesis

This experiment tests the hypothesis that symmetric activation functions (f(x) = f(-x))
can create a "Concept Folding" mechanism where the network learns to ignore polarity
and focuses on underlying magnitudes/relationships.

The Hybrid Model interleaves standard layers (for detail preservation) with 
symmetric layers (for concept abstraction).

Usage:
    python symmetric_activation_experiment.py --config local   # Quick local test
    python symmetric_activation_experiment.py --config gpu     # Full GPU experiment
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import argparse
import math
import os

# === CONFIGURATION ===

# Local testing configuration (quick validation ~1-2 min)
LOCAL_CONFIG = {
    'vocab_size': 10000,
    'd_model': 64,
    'n_heads': 2,
    'n_layers': 2,
    'dropout': 0.1,
    'n_experts': 4,
    'top_k_experts': 2,
    'batch_size': 16,
    'context_length': 64,
    'num_samples': 2000,
    'num_epochs': 1,
    'learning_rate': 1e-3,
}

# GPU configuration for 16GB VRAM (full experiment)
GPU_CONFIG = {
    'vocab_size': 10000,
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 6,
    'dropout': 0.1,
    'n_experts': 8,
    'top_k_experts': 2,
    'batch_size': 32,
    'context_length': 128,
    'num_samples': 50000,
    'num_epochs': 5,
    'learning_rate': 1e-3,
}

# === SYMMETRIC ACTIVATION FUNCTIONS ===

class CosineActivation(nn.Module):
    """Symmetric activation: cos(x). Periodic and symmetric around 0."""
    def forward(self, x):
        return torch.cos(x)

class SquaredActivation(nn.Module):
    """Symmetric activation: x². Simple parabolic symmetric function."""
    def forward(self, x):
        return x ** 2

class GaussianActivation(nn.Module):
    """Symmetric activation: exp(-x²). Bell-curve symmetric function."""
    def forward(self, x):
        return torch.exp(-x ** 2)

# === SHARED COMPONENTS ===

class RMSNorm(nn.Module):
    """Root Mean Square Normalization (from Qwen3)"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class QKNorm(nn.Module):
    """QK Normalization for attention (from Qwen3)"""
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=eps)
    
    def forward(self, q, k):
        return self.norm(q), self.norm(k)

class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention with QK-Norm"""
    def __init__(self, d_model, n_heads, context_length, dropout):
        super().__init__()
        self.num_heads = n_heads
        self.head_size = d_model // n_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.qk_norm = QKNorm(self.head_size)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        q_norm, k_norm = self.qk_norm(q, k)
        
        wei = q_norm @ k_norm.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        out = wei @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.dropout(self.proj(out))
        return out

# === EXPERT MODULES ===

class StandardExpert(nn.Module):
    """Standard Expert using SwiGLU (SiLU-based gating)"""
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = 4 * d_model
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
    
    def forward(self, x):
        # SwiGLU: silu(W1(x)) * W3(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class SymmetricExpert(nn.Module):
    """
    Symmetric Expert using a symmetric activation function.
    The symmetric property f(x) = f(-x) creates "Concept Folding".
    """
    def __init__(self, d_model, activation_type='cosine'):
        super().__init__()
        hidden_dim = 4 * d_model
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        
        # Select symmetric activation
        if activation_type == 'cosine':
            self.activation = CosineActivation()
        elif activation_type == 'squared':
            self.activation = SquaredActivation()
        elif activation_type == 'gaussian':
            self.activation = GaussianActivation()
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")
    
    def forward(self, x):
        # Symmetric activation with gating (similar structure to SwiGLU but symmetric)
        return self.w2(self.activation(self.w1(x)) * self.w3(x))

# === MOE LAYERS ===

class StandardMoE(nn.Module):
    """Standard Mixture of Experts (baseline)"""
    def __init__(self, d_model, n_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([StandardExpert(d_model) for _ in range(n_experts)])
        self.gating_network = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        B, T, C = x.shape
        x_reshaped = x.view(-1, C)
        
        logits = self.gating_network(x_reshaped)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype)
        
        output = torch.zeros_like(x_reshaped)
        
        for i, expert in enumerate(self.experts):
            token_indices, top_k_indices = torch.where(indices == i)
            if token_indices.numel() > 0:
                expert_weights = weights[token_indices, top_k_indices].unsqueeze(-1)
                expert_output = expert(x_reshaped[token_indices])
                output.index_add_(0, token_indices, expert_output * expert_weights)
        
        return output.view(B, T, C)

class HybridMoE(nn.Module):
    """
    Hybrid Mixture of Experts with alternating standard and symmetric experts.
    This creates the "Fractal Symmetry" architecture where some experts
    focus on preserving details while others abstract concepts.
    """
    def __init__(self, d_model, n_experts, top_k, symmetric_activation='cosine'):
        super().__init__()
        self.experts = nn.ModuleList()
        
        # Alternate between standard and symmetric experts
        for i in range(n_experts):
            if i % 2 == 0:
                self.experts.append(StandardExpert(d_model))
            else:
                self.experts.append(SymmetricExpert(d_model, symmetric_activation))
        
        self.gating_network = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        B, T, C = x.shape
        x_reshaped = x.view(-1, C)
        
        logits = self.gating_network(x_reshaped)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype)
        
        output = torch.zeros_like(x_reshaped)
        
        for i, expert in enumerate(self.experts):
            token_indices, top_k_indices = torch.where(indices == i)
            if token_indices.numel() > 0:
                expert_weights = weights[token_indices, top_k_indices].unsqueeze(-1)
                expert_output = expert(x_reshaped[token_indices])
                output.index_add_(0, token_indices, expert_output * expert_weights)
        
        return output.view(B, T, C)

# === TRANSFORMER BLOCKS ===

class StandardTransformerBlock(nn.Module):
    """Standard Transformer Block with MoE (baseline)"""
    def __init__(self, d_model, n_heads, n_experts, top_k, context_length, dropout):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, n_heads, context_length, dropout)
        self.moe = StandardMoE(d_model, n_experts, top_k)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

class HybridTransformerBlock(nn.Module):
    """Hybrid Transformer Block with symmetric activation MoE"""
    def __init__(self, d_model, n_heads, n_experts, top_k, context_length, dropout, 
                 symmetric_activation='cosine'):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, n_heads, context_length, dropout)
        self.moe = HybridMoE(d_model, n_experts, top_k, symmetric_activation)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

# === MODELS ===

class TinyQwen3(nn.Module):
    """Baseline TinyQwen3 Model (standard architecture)"""
    def __init__(self, config, device):
        super().__init__()
        self.config = config
        self.device = device
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embedding_table = nn.Embedding(config['context_length'], config['d_model'])
        self.blocks = nn.Sequential(*[
            StandardTransformerBlock(
                config['d_model'], config['n_heads'], 
                config['n_experts'], config['top_k_experts'],
                config['context_length'], config['dropout']
            ) for _ in range(config['n_layers'])
        ])
        self.ln_f = RMSNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_view = logits.view(B*T, C)
            targets_view = targets.view(B*T)
            loss = F.cross_entropy(logits_view, targets_view)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['context_length']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

class HybridSymmetricQwen3(nn.Module):
    """
    Hybrid Symmetric TinyQwen3 Model
    
    Uses alternating standard and symmetric experts within the MoE layer.
    This creates the "Fractal Symmetry" effect where:
    - Standard experts preserve directional/detailed information
    - Symmetric experts abstract concepts (folding +/- into same representation)
    """
    def __init__(self, config, device, symmetric_activation='cosine'):
        super().__init__()
        self.config = config
        self.device = device
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embedding_table = nn.Embedding(config['context_length'], config['d_model'])
        self.blocks = nn.Sequential(*[
            HybridTransformerBlock(
                config['d_model'], config['n_heads'], 
                config['n_experts'], config['top_k_experts'],
                config['context_length'], config['dropout'],
                symmetric_activation
            ) for _ in range(config['n_layers'])
        ])
        self.ln_f = RMSNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_view = logits.view(B*T, C)
            targets_view = targets.view(B*T)
            loss = F.cross_entropy(logits_view, targets_view)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['context_length']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

# === DATASET ===

class TinyStoriesDataset(Dataset):
    """Dataset for TinyStories with tokenization"""
    def __init__(self, data, tokenizer, context_length, model_vocab_size):
        self.data = data
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.model_vocab_size = model_vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        tokens = self.tokenizer.encode(
            text, return_tensors='pt', 
            max_length=self.context_length + 1, 
            padding='max_length', truncation=True
        )
        tokens = tokens.squeeze(0)
        tokens = tokens % self.model_vocab_size
        x = tokens[:-1]
        y = tokens[1:]
        return x, y

# === TRAINING ===

def train_model(model, train_loader, config, device, model_name):
    """Train a model and return loss history"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    losses = []
    
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    for epoch in range(config['num_epochs']):
        model.train()
        epoch_losses = []
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            logits, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            if step % 50 == 0:
                print(f"  Epoch {epoch+1}/{config['num_epochs']}, Step {step}, Loss: {loss.item():.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
    
    return losses

def evaluate_perplexity(model, data_loader, device):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_batches = 0
    
    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
            _, loss = model(xb, yb)
            total_loss += loss.item()
            total_batches += 1
    
    avg_loss = total_loss / max(total_batches, 1)
    perplexity = math.exp(avg_loss)
    return perplexity

# === VISUALIZATION ===

def plot_comparison(baseline_losses, hybrid_losses, config, output_dir='output'):
    """Generate comparison plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = range(1, len(baseline_losses) + 1)
    
    plt.figure(figsize=(14, 5))
    
    # Loss comparison
    plt.subplot(1, 2, 1)
    plt.plot(epochs, baseline_losses, 'b-o', label='Standard TinyQwen3', linewidth=2)
    plt.plot(epochs, hybrid_losses, 'r-s', label='Hybrid Symmetric', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Final loss bar chart
    plt.subplot(1, 2, 2)
    models = ['Standard\nTinyQwen3', 'Hybrid\nSymmetric']
    final_losses = [baseline_losses[-1], hybrid_losses[-1]]
    colors = ['#3498db', '#e74c3c']
    bars = plt.bar(models, final_losses, color=colors, edgecolor='black', linewidth=1.2)
    plt.ylabel('Final Training Loss', fontsize=12)
    plt.title('Final Loss Comparison', fontsize=14)
    
    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{loss:.4f}', ha='center', va='bottom', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/symmetric_comparison_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_dir}/symmetric_comparison_loss.png")

# === MAIN ===

def main():
    parser = argparse.ArgumentParser(description='Symmetric Activation Experiment')
    parser.add_argument('--config', type=str, default='local', choices=['local', 'gpu'],
                        help='Configuration to use: local (quick test) or gpu (full experiment)')
    parser.add_argument('--activation', type=str, default='cosine', 
                        choices=['cosine', 'squared', 'gaussian'],
                        help='Symmetric activation function to use')
    args = parser.parse_args()
    
    # Select configuration
    config = LOCAL_CONFIG if args.config == 'local' else GPU_CONFIG
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Configuration: {args.config}")
    print(f"Symmetric activation: {args.activation}")
    
    if device == 'cuda':
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Load tokenizer and dataset
    print("\nLoading tokenizer and dataset...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    
    dataset = load_dataset("roneneldan/TinyStories", split="train")
    num_samples = min(config['num_samples'], len(dataset))
    subset_data = dataset.select(range(num_samples))
    print(f"Using {num_samples} samples")
    
    train_dataset = TinyStoriesDataset(
        subset_data, tokenizer, 
        config['context_length'], config['vocab_size']
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], 
        shuffle=True, num_workers=0
    )
    
    # Create models
    print("\nInitializing models...")
    baseline_model = TinyQwen3(config, device).to(device)
    hybrid_model = HybridSymmetricQwen3(config, device, args.activation).to(device)
    
    # Count parameters
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    hybrid_params = sum(p.numel() for p in hybrid_model.parameters())
    print(f"Standard TinyQwen3 parameters: {baseline_params:,}")
    print(f"Hybrid Symmetric parameters: {hybrid_params:,}")
    
    # Train both models
    baseline_losses = train_model(baseline_model, train_loader, config, device, "Standard TinyQwen3")
    hybrid_losses = train_model(hybrid_model, train_loader, config, device, "Hybrid Symmetric")
    
    # Evaluate perplexity
    print("\nEvaluating perplexity...")
    baseline_ppl = evaluate_perplexity(baseline_model, train_loader, device)
    hybrid_ppl = evaluate_perplexity(hybrid_model, train_loader, device)
    print(f"Standard TinyQwen3 Perplexity: {baseline_ppl:.2f}")
    print(f"Hybrid Symmetric Perplexity: {hybrid_ppl:.2f}")
    
    # Generate comparison plots
    plot_comparison(baseline_losses, hybrid_losses, config)
    
    # Generate sample text
    print("\n" + "="*50)
    print("GENERATION COMPARISON")
    print("="*50)
    
    prompt = "Once upon a time"
    start_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    start_tokens = start_tokens % config['vocab_size']
    
    print(f"\nPrompt: {prompt}")
    
    baseline_model.eval()
    hybrid_model.eval()
    
    generated_baseline = baseline_model.generate(start_tokens.clone(), max_new_tokens=50)
    generated_hybrid = hybrid_model.generate(start_tokens.clone(), max_new_tokens=50)
    
    print(f"\n[Standard TinyQwen3]:")
    print(tokenizer.decode(generated_baseline[0], skip_special_tokens=True))
    
    print(f"\n[Hybrid Symmetric ({args.activation})]:")
    print(tokenizer.decode(generated_hybrid[0], skip_special_tokens=True))
    
    # Summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Configuration: {args.config}")
    print(f"Symmetric Activation: {args.activation}")
    print(f"Final Loss - Standard: {baseline_losses[-1]:.4f}")
    print(f"Final Loss - Hybrid: {hybrid_losses[-1]:.4f}")
    print(f"Perplexity - Standard: {baseline_ppl:.2f}")
    print(f"Perplexity - Hybrid: {hybrid_ppl:.2f}")
    
    improvement = (baseline_losses[-1] - hybrid_losses[-1]) / baseline_losses[-1] * 100
    if improvement > 0:
        print(f"\n✓ Hybrid model shows {improvement:.1f}% improvement in final loss!")
    else:
        print(f"\n✗ Standard model outperforms by {-improvement:.1f}%")
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()
