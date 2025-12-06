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
    python symmetric_activation_experiment.py --config fast --activation all  # Test all activations
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
import json
from datetime import datetime

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

# Fast configuration for quick experimentation
FAST_CONFIG = {
    'vocab_size': 10000,
    'd_model': 128,
    'n_heads': 4,
    'n_layers': 3,
    'dropout': 0.1,
    'n_experts': 4,
    'top_k_experts': 2,
    'batch_size': 32,
    'context_length': 64,
    'num_samples': 5000,
    'num_epochs': 2,
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

# All available symmetric activations
ALL_ACTIVATIONS = [
    'cosine',
    'squared', 
    'gaussian',
    'scaled_cosine',
    'shifted_symmetric',
    'leaky_symmetric',
]

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

class ScaledCosineActivation(nn.Module):
    """
    Symmetric activation: cos(α·x) where α is learnable.
    The learnable scale controls oscillation frequency.
    """
    def __init__(self, init_scale=1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(init_scale))
    
    def forward(self, x):
        return torch.cos(self.scale * x)

class ShiftedSymmetricActivation(nn.Module):
    """
    Symmetric activation: cos(x) + 1.
    Keeps output in positive range [0, 2] while maintaining symmetry.
    """
    def forward(self, x):
        return torch.cos(x) + 1.0

class LeakySymmetricActivation(nn.Module):
    """
    Hybrid activation: 0.5·ReLU(x) + 0.5·cos(x)
    Combines asymmetric gradient flow with symmetric abstraction.
    """
    def __init__(self, relu_weight=0.5, cos_weight=0.5):
        super().__init__()
        self.relu_weight = relu_weight
        self.cos_weight = cos_weight
    
    def forward(self, x):
        return self.relu_weight * F.relu(x) + self.cos_weight * torch.cos(x)

def get_activation(activation_type):
    """Factory function to create activation by name."""
    activations = {
        'cosine': CosineActivation,
        'squared': SquaredActivation,
        'gaussian': GaussianActivation,
        'scaled_cosine': ScaledCosineActivation,
        'shifted_symmetric': ShiftedSymmetricActivation,
        'leaky_symmetric': LeakySymmetricActivation,
    }
    if activation_type not in activations:
        raise ValueError(f"Unknown activation type: {activation_type}. Available: {list(activations.keys())}")
    return activations[activation_type]()

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

class SymmetricAttention(nn.Module):
    """
    Attention with symmetric activation applied to attention scores.
    Tests the hypothesis of symmetry in the attention mechanism itself.
    """
    def __init__(self, d_model, n_heads, context_length, dropout, activation_type='cosine'):
        super().__init__()
        self.num_heads = n_heads
        self.head_size = d_model // n_heads
        self.d_model = d_model
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.qk_norm = QKNorm(self.head_size)
        self.symmetric_activation = get_activation(activation_type)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        q_norm, k_norm = self.qk_norm(q, k)
        
        # Apply symmetric activation to attention scores
        wei = q_norm @ k_norm.transpose(-2, -1) * self.head_size**-0.5
        wei = self.symmetric_activation(wei)  # Apply symmetric activation
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
        self.activation = get_activation(activation_type)
    
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

class FullSymmetricMoE(nn.Module):
    """MoE where ALL experts use symmetric activation (for layer-wise experiment)"""
    def __init__(self, d_model, n_experts, top_k, symmetric_activation='cosine'):
        super().__init__()
        self.experts = nn.ModuleList([
            SymmetricExpert(d_model, symmetric_activation) for _ in range(n_experts)
        ])
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

class SymmetricAttentionBlock(nn.Module):
    """Transformer Block with symmetric activation in attention"""
    def __init__(self, d_model, n_heads, n_experts, top_k, context_length, dropout,
                 symmetric_activation='cosine'):
        super().__init__()
        self.sa = SymmetricAttention(d_model, n_heads, context_length, dropout, symmetric_activation)
        self.moe = StandardMoE(d_model, n_experts, top_k)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

class FullSymmetricTransformerBlock(nn.Module):
    """Transformer Block with all symmetric experts (for layer-wise experiment)"""
    def __init__(self, d_model, n_heads, n_experts, top_k, context_length, dropout,
                 symmetric_activation='cosine'):
        super().__init__()
        self.sa = MultiHeadAttention(d_model, n_heads, context_length, dropout)
        self.moe = FullSymmetricMoE(d_model, n_experts, top_k, symmetric_activation)
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
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, greedy=False):
        """
        Generate text with improved sampling.
        
        Args:
            idx: Starting token indices [B, T]
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Limit sampling to top-k tokens (0 = no limit)
            greedy: If True, use greedy decoding (always pick most likely)
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['context_length']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]  # [B, vocab_size]
            
            if greedy:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                # Apply temperature
                if temperature > 0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                
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
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, greedy=False):
        """Generate text with improved sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['context_length']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if greedy:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                if temperature > 0:
                    logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx

class SymmetricAttentionQwen3(nn.Module):
    """Model with symmetric activation in attention mechanism"""
    def __init__(self, config, device, symmetric_activation='cosine'):
        super().__init__()
        self.config = config
        self.device = device
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embedding_table = nn.Embedding(config['context_length'], config['d_model'])
        self.blocks = nn.Sequential(*[
            SymmetricAttentionBlock(
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
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, greedy=False):
        """Generate text with improved sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['context_length']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if greedy:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                if temperature > 0:
                    logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        self.train()
        return idx

class LayerWiseSymmetricQwen3(nn.Module):
    """
    Layer-wise alternation: alternate full standard and symmetric layers.
    Even layers use standard MoE, odd layers use full symmetric MoE.
    """
    def __init__(self, config, device, symmetric_activation='cosine'):
        super().__init__()
        self.config = config
        self.device = device
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['d_model'])
        self.position_embedding_table = nn.Embedding(config['context_length'], config['d_model'])
        
        blocks = []
        for i in range(config['n_layers']):
            if i % 2 == 0:
                blocks.append(StandardTransformerBlock(
                    config['d_model'], config['n_heads'],
                    config['n_experts'], config['top_k_experts'],
                    config['context_length'], config['dropout']
                ))
            else:
                blocks.append(FullSymmetricTransformerBlock(
                    config['d_model'], config['n_heads'],
                    config['n_experts'], config['top_k_experts'],
                    config['context_length'], config['dropout'],
                    symmetric_activation
                ))
        self.blocks = nn.Sequential(*blocks)
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
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, greedy=False):
        """Generate text with improved sampling."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config['context_length']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            
            if greedy:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                if temperature > 0:
                    logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float('-inf')
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

def train_model(model, train_loader, config, device, model_name, verbose=True):
    """Train a model and return loss history"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    losses = []
    
    if verbose:
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
            
            if verbose and step % 50 == 0:
                print(f"  Epoch {epoch+1}/{config['num_epochs']}, Step {step}, Loss: {loss.item():.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        if verbose:
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

def plot_comparison(results, config, output_dir='output'):
    """Generate comparison plots for all models"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine number of epochs from results
    first_key = list(results.keys())[0]
    epochs = range(1, len(results[first_key]['losses']) + 1)
    
    # Colors for different models
    colors = plt.cm.tab10(range(len(results)))
    
    plt.figure(figsize=(16, 6))
    
    # Loss comparison
    plt.subplot(1, 2, 1)
    for i, (name, data) in enumerate(results.items()):
        plt.plot(epochs, data['losses'], '-o', label=name, linewidth=2, color=colors[i])
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14)
    plt.legend(fontsize=8, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Final loss bar chart
    plt.subplot(1, 2, 2)
    names = list(results.keys())
    final_losses = [results[name]['losses'][-1] for name in names]
    bars = plt.bar(range(len(names)), final_losses, color=colors, edgecolor='black', linewidth=1.2)
    plt.xticks(range(len(names)), [n.replace(' ', '\n') for n in names], fontsize=8)
    plt.ylabel('Final Training Loss', fontsize=12)
    plt.title('Final Loss Comparison', fontsize=14)
    
    # Add value labels on bars
    for bar, loss in zip(bars, final_losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{loss:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/symmetric_comparison_all.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to {output_dir}/symmetric_comparison_all.png")

def save_results_json(results, config, args, output_dir='output'):
    """Save results to JSON for tracking"""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        'timestamp': timestamp,
        'config': args.config,
        'activation': args.activation,
        'results': {
            name: {
                'final_loss': data['losses'][-1],
                'perplexity': data['perplexity'],
                'losses': data['losses']
            }
            for name, data in results.items()
        }
    }
    
    filepath = f'{output_dir}/results_{timestamp}.json'
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {filepath}")

# === MAIN ===

def run_single_experiment(activation, config, device, train_loader, tokenizer, verbose=True):
    """Run experiment for a single activation type"""
    results = {}
    
    # Baseline
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing activation: {activation}")
        print(f"{'='*60}")
    
    baseline_model = TinyQwen3(config, device).to(device)
    baseline_losses = train_model(baseline_model, train_loader, config, device, 
                                   "Standard TinyQwen3", verbose)
    baseline_ppl = evaluate_perplexity(baseline_model, train_loader, device)
    results['Standard'] = {'losses': baseline_losses, 'perplexity': baseline_ppl, 'model': baseline_model}
    
    # Hybrid MoE with activation
    hybrid_model = HybridSymmetricQwen3(config, device, activation).to(device)
    hybrid_losses = train_model(hybrid_model, train_loader, config, device,
                                 f"Hybrid-{activation}", verbose)
    hybrid_ppl = evaluate_perplexity(hybrid_model, train_loader, device)
    results[f'Hybrid-{activation}'] = {'losses': hybrid_losses, 'perplexity': hybrid_ppl, 'model': hybrid_model}
    
    return results

def run_all_experiments(config, device, train_loader, tokenizer):
    """Run all experiment types with cosine activation as base"""
    results = {}
    activation = 'cosine'  # Use cosine for structural experiments
    
    print(f"\n{'='*60}")
    print("Running comprehensive experiment suite")
    print(f"{'='*60}")
    
    # 1. Baseline
    print("\n[1/6] Training Standard baseline...")
    baseline_model = TinyQwen3(config, device).to(device)
    baseline_losses = train_model(baseline_model, train_loader, config, device, 
                                   "Standard TinyQwen3", verbose=True)
    baseline_ppl = evaluate_perplexity(baseline_model, train_loader, device)
    results['Standard'] = {'losses': baseline_losses, 'perplexity': baseline_ppl, 'model': baseline_model}
    
    # 2. Hybrid MoE (original experiment)
    print("\n[2/6] Training Hybrid MoE (cosine)...")
    hybrid_model = HybridSymmetricQwen3(config, device, 'cosine').to(device)
    hybrid_losses = train_model(hybrid_model, train_loader, config, device,
                                 "Hybrid-cosine", verbose=True)
    hybrid_ppl = evaluate_perplexity(hybrid_model, train_loader, device)
    results['Hybrid-cosine'] = {'losses': hybrid_losses, 'perplexity': hybrid_ppl, 'model': hybrid_model}
    
    # 3. Scaled Cosine
    print("\n[3/6] Training Hybrid MoE (scaled_cosine)...")
    scaled_model = HybridSymmetricQwen3(config, device, 'scaled_cosine').to(device)
    scaled_losses = train_model(scaled_model, train_loader, config, device,
                                 "Hybrid-scaled_cosine", verbose=True)
    scaled_ppl = evaluate_perplexity(scaled_model, train_loader, device)
    results['Hybrid-scaled_cos'] = {'losses': scaled_losses, 'perplexity': scaled_ppl, 'model': scaled_model}
    
    # 4. Leaky Symmetric
    print("\n[4/6] Training Hybrid MoE (leaky_symmetric)...")
    leaky_model = HybridSymmetricQwen3(config, device, 'leaky_symmetric').to(device)
    leaky_losses = train_model(leaky_model, train_loader, config, device,
                                "Hybrid-leaky_symmetric", verbose=True)
    leaky_ppl = evaluate_perplexity(leaky_model, train_loader, device)
    results['Hybrid-leaky'] = {'losses': leaky_losses, 'perplexity': leaky_ppl, 'model': leaky_model}
    
    # 5. Symmetric Attention
    print("\n[5/6] Training Symmetric Attention...")
    attn_model = SymmetricAttentionQwen3(config, device, 'cosine').to(device)
    attn_losses = train_model(attn_model, train_loader, config, device,
                               "SymmetricAttention", verbose=True)
    attn_ppl = evaluate_perplexity(attn_model, train_loader, device)
    results['SymAttn'] = {'losses': attn_losses, 'perplexity': attn_ppl, 'model': attn_model}
    
    # 6. Layer-wise alternation
    print("\n[6/6] Training Layer-wise alternation...")
    layer_model = LayerWiseSymmetricQwen3(config, device, 'cosine').to(device)
    layer_losses = train_model(layer_model, train_loader, config, device,
                                "LayerWise", verbose=True)
    layer_ppl = evaluate_perplexity(layer_model, train_loader, device)
    results['LayerWise'] = {'losses': layer_losses, 'perplexity': layer_ppl, 'model': layer_model}
    
    return results

def generate_comparison(results, tokenizer, config, device):
    """Generate and compare text from all models"""
    print("\n" + "="*60)
    print("GENERATION COMPARISON")
    print("="*60)
    
    prompt = "Once upon a time"
    start_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    start_tokens = start_tokens % config['vocab_size']
    
    print(f"\nPrompt: {prompt}")
    print("-" * 40)
    
    for name, data in results.items():
        model = data['model']
        model.eval()
        
        # Generate with improved settings
        generated = model.generate(
            start_tokens.clone(), 
            max_new_tokens=50,
            temperature=0.8,
            top_k=50,
            greedy=False
        )
        
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"\n[{name}]:")
        print(text[:200])  # Limit output length

def print_summary_table(results):
    """Print a formatted summary table"""
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    # Header
    print(f"{'Model':<25} {'Final Loss':>12} {'Perplexity':>12} {'Δ vs Std':>12}")
    print("-"*70)
    
    # Get baseline loss
    baseline_loss = results['Standard']['losses'][-1]
    
    # Sort by loss
    sorted_results = sorted(results.items(), key=lambda x: x[1]['losses'][-1])
    
    for name, data in sorted_results:
        loss = data['losses'][-1]
        ppl = data['perplexity']
        delta = ((baseline_loss - loss) / baseline_loss) * 100
        delta_str = f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
        
        # Highlight best result
        marker = " ✓" if name != 'Standard' and delta > 0 else ""
        print(f"{name:<25} {loss:>12.4f} {ppl:>12.2f} {delta_str:>12}{marker}")
    
    print("-"*70)
    
    # Find best model
    best_name = sorted_results[0][0]
    if best_name != 'Standard':
        improvement = ((baseline_loss - sorted_results[0][1]['losses'][-1]) / baseline_loss) * 100
        print(f"\n✓ Best model: {best_name} ({improvement:.1f}% improvement)")
    else:
        print(f"\n✗ Standard baseline performed best")

def main():
    parser = argparse.ArgumentParser(description='Symmetric Activation Experiment')
    parser.add_argument('--config', type=str, default='local', choices=['local', 'fast', 'gpu'],
                        help='Configuration to use: local (quick test), fast (iteration), or gpu (full)')
    parser.add_argument('--activation', type=str, default='cosine', 
                        choices=ALL_ACTIVATIONS + ['all'],
                        help='Symmetric activation function to use, or "all" to test all variants')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Generation temperature (default: 0.8)')
    parser.add_argument('--top_k', type=int, default=50,
                        help='Top-k sampling limit (default: 50)')
    parser.add_argument('--greedy', action='store_true',
                        help='Use greedy decoding for generation')
    args = parser.parse_args()
    
    # Select configuration
    if args.config == 'local':
        config = LOCAL_CONFIG
    elif args.config == 'fast':
        config = FAST_CONFIG
    else:
        config = GPU_CONFIG
    
    # Device setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Configuration: {args.config}")
    print(f"Activation: {args.activation}")
    
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
    
    # Run experiments
    if args.activation == 'all':
        results = run_all_experiments(config, device, train_loader, tokenizer)
    else:
        results = run_single_experiment(args.activation, config, device, train_loader, tokenizer)
    
    # Evaluate perplexity (already done during training)
    print("\nPerplexity Results:")
    for name, data in results.items():
        print(f"  {name}: {data['perplexity']:.2f}")
    
    # Generate comparison plots
    plot_comparison(results, config)
    
    # Save results to JSON
    save_results_json(results, config, args)
    
    # Generate sample text from all models
    generate_comparison(results, tokenizer, config, device)
    
    # Print summary table
    print_summary_table(results)
    
    print("\nExperiment complete!")

if __name__ == "__main__":
    main()
