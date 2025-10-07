import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import math

# --- Configuration ---
# Based on the Qwen3 technical report, adjusted for a tiny educational model.
vocab_size = 10000      # Size of our vocabulary
d_model = 256           # The main dimension of the model
n_heads = 4             # Number of attention heads
n_layers = 4            # Number of transformer blocks
dropout = 0.1           # Dropout rate
n_experts = 8           # Number of experts in the MoE layer
top_k_experts = 2       # Number of experts to route each token to
batch_size = 32         # How many sequences to process at once
context_length = 128    # Maximum context length for predictions
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# --- Architectural Components from Qwen3 Paper ---

# 1. RMSNorm (Root Mean Square Normalization)
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# 2. QK-Norm (Applied to Query and Key vectors in attention)
# As per the paper, this is added to the attention mechanism for stability.
class QKNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        # Using RMSNorm for QK normalization as is common
        self.norm = RMSNorm(d_model, eps=eps)
    
    def forward(self, q, k):
        return self.norm(q), self.norm(k)

# 3. Standard Self-Attention Head
class AttentionHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(d_model, head_size, bias=False)
        self.query = nn.Linear(d_model, head_size, bias=False)
        self.value = nn.Linear(d_model, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        # Generate Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Note: In a full implementation, RoPE would be applied to Q and K here.
        
        # Calculate attention scores
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Perform the weighted aggregation of the values
        out = wei @ v
        return out

# 4. Multi-Head Attention
# Note: Qwen3 uses Grouped Query Attention (GQA). For simplicity, we use standard MHA.
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.qk_norm = QKNorm(head_size)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        
        # Generate Q, K, V for all heads at once
        qkv = self.qkv(x)  # (B, T, 3 * d_model)
        q, k, v = qkv.split(d_model, dim=2)
        
        # Reshape for multi-head attention: (B, T, num_heads, head_size)
        q = q.view(B, T, self.num_heads, self.head_size).transpose(1, 2)  # (B, num_heads, T, head_size)
        k = k.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_size).transpose(1, 2)
        
        # Apply QK-Norm to each head
        q_norm, k_norm = self.qk_norm(q, k)
        
        # Compute attention
        wei = q_norm @ k_norm.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Apply attention to values
        out = wei @ v  # (B, num_heads, T, head_size)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, d_model)
        out = self.dropout(self.proj(out))
        return out

# 5. Mixture of Experts (MoE) Layer
class Expert(nn.Module):
    """An expert module using SwiGLU, as described in the Qwen3 paper."""
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = 4 * d_model
        # The first linear layer outputs 2x hidden_dim for the gate and the value
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
    
    def forward(self, x):
        # SwiGLU: silu(W1(x)) * W3(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    def __init__(self, d_model, n_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(n_experts)])
        self.gating_network = nn.Linear(d_model, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x):
        B, T, C = x.shape
        x_reshaped = x.view(-1, C)
        
        logits = self.gating_network(x_reshaped)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1, dtype=torch.float).to(x.dtype)
        
        output = torch.zeros_like(x_reshaped)
        
        # This is a simplified, non-optimized implementation for clarity.
        for i, expert in enumerate(self.experts):
            # Find which tokens are routed to this expert
            token_indices, top_k_indices = torch.where(indices == i)
            if token_indices.numel() > 0:
                expert_weights = weights[token_indices, top_k_indices].unsqueeze(-1)
                expert_output = expert(x_reshaped[token_indices])
                output.index_add_(0, token_indices, expert_output * expert_weights)
        
        return output.view(B, T, C)

# --- 7. Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_experts, top_k):
        super().__init__()
        head_size = d_model // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.moe = MoE(d_model, n_experts, top_k)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x):
        # Pre-normalization architecture
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

# --- 8. The Main Model ---
class TinyQwen3(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(context_length, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads, n_experts, top_k_experts) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d_model) # Final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
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
            idx_cond = idx[:, -context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx

# --- 9. Data Preparation ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

class TinyStoriesDataset(Dataset):
    def __init__(self, data, tokenizer, context_length, model_vocab_size):
        self.data = data
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.model_vocab_size = model_vocab_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        tokens = self.tokenizer.encode(text, return_tensors='pt', max_length=self.context_length + 1, padding='max_length', truncation=True)
        tokens = tokens.squeeze(0)
        # Ensure token IDs are within the model's vocab size
        # Map any token >= model_vocab_size to a token within range (using modulo)
        tokens = tokens % self.model_vocab_size
        x = tokens[:-1]
        y = tokens[1:]
        return x, y

print("Loading and preparing dataset...")
dataset = load_dataset("roneneldan/TinyStories")
train_data = dataset['train']
small_train_data = train_data.select(range(20000)) 

train_dataset = TinyStoriesDataset(small_train_data, tokenizer, context_length, vocab_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- 10. Training ---
model = TinyQwen3().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
num_epochs = 3

print("Starting training...")
for epoch in range(num_epochs):
    for i, (xb, yb) in enumerate(train_loader):
        xb, yb = xb.to(device), yb.to(device)
        
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Step {i}, Loss: {loss.item():.4f}")

print("Training finished.")

# --- 11. Generation Example ---
print("\n--- Generating some text ---")
prompt = "Once upon a time, in a land far away"
start_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
generated_tokens = model.generate(start_tokens, max_new_tokens=50)
generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(generated_text)
