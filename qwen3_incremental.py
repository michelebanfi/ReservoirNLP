import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import os

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

#LOCAL TESTING CONFIGURATION
# vocab_size = 10000      # Size of our vocabulary
# d_model = 64           # The main dimension of the model
# n_heads = 2             # Number of attention heads
# n_layers = 2            # Number of transformer blocks
# dropout = 0.1           # Dropout rate
# n_experts = 2           # Number of experts in the MoE layer
# top_k_experts = 2       # Number of experts to route each token to
# batch_size = 32         # How many sequences to process at once
# context_length = 128    # Maximum context length for predictions

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
class QKNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=eps)
    
    def forward(self, q, k):
        return self.norm(q), self.norm(k)

# 3. Multi-Head Attention
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
        
        qkv = self.qkv(x)
        q, k, v = qkv.split(d_model, dim=2)
        
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

# 4. Mixture of Experts (MoE) Layer
class Expert(nn.Module):
    """An expert module using SwiGLU."""
    def __init__(self, d_model):
        super().__init__()
        hidden_dim = 4 * d_model
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
    
    def forward(self, x):
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
        
        for i, expert in enumerate(self.experts):
            token_indices, top_k_indices = torch.where(indices == i)
            if token_indices.numel() > 0:
                expert_weights = weights[token_indices, top_k_indices].unsqueeze(-1)
                expert_output = expert(x_reshaped[token_indices])
                output.index_add_(0, token_indices, expert_output * expert_weights)
        
        return output.view(B, T, C)

# 5. Transformer Block
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_experts, top_k):
        super().__init__()
        head_size = d_model // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.moe = MoE(d_model, n_experts, top_k)
        self.ln1 = RMSNorm(d_model)
        self.ln2 = RMSNorm(d_model)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.moe(self.ln2(x))
        return x

# 6. The Main Model
class TinyQwen3(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(context_length, d_model)
        self.blocks = nn.Sequential(*[TransformerBlock(d_model, n_heads, n_experts, top_k_experts) for _ in range(n_layers)])
        self.ln_f = RMSNorm(d_model)
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

# --- Data Preparation ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# Generic Dataset Class for Curriculum Learning
class CurriculumDataset(Dataset):
    def __init__(self, data, tokenizer, context_length, model_vocab_size, formatting_fn):
        self.data = data
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.model_vocab_size = model_vocab_size
        self.formatting_fn = formatting_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Format the text using the provided function
        text = self.formatting_fn(self.data[idx])
        
        # Tokenize the text
        tokens = self.tokenizer.encode(text, return_tensors='pt', max_length=self.context_length + 1, padding='max_length', truncation=True)
        tokens = tokens.squeeze(0)
        
        # This is a simplification to keep tokens within our small vocab size.
        # In a real-world scenario, you would train a tokenizer or use a model's native one.
        tokens = tokens % self.model_vocab_size
        
        x = tokens[:-1]
        y = tokens[1:]
        return x, y

# --- Formatting Functions for each Dataset ---
def format_tinystories(example):
    return example['text']

def format_squad(example):
    # Combine context, question, and answer into a single string
    context = example['context']
    question = example['question']
    # SQuAD answers is a dict with 'text' as a list
    answer = example['answers']['text'][0] if example['answers']['text'] else "unknown"
    return f"Context: {context}\nQuestion: {question}\nAnswer: {answer}"

def format_logicnli(example):
    # The label is already a string in this dataset, not a numeric value
    label_text = example['label']
    return f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}\nLabel: {label_text}"

# --- Training Stage Definition ---
training_stages = [
    {
        "name": "Stage 1: Foundational Language",
        "dataset_name": "roneneldan/TinyStories",
        "dataset_split": "train",
        "num_samples": 20000,
        "num_epochs": 2,
        "learning_rate": 1e-3,
        "formatting_fn": format_tinystories,
        "generation_prompt": "Once upon a time"
    },
    {
        "name": "Stage 2: Simple Reasoning & QA",
        "dataset_name": "squad",  # Using the canonical squad dataset
        "dataset_split": "train",
        "num_samples": 10000,
        "num_epochs": 3,
        "learning_rate": 5e-4,
        "formatting_fn": format_squad,
        "generation_prompt": "Context: The Amazon rainforest is a moist broadleaf forest.\nQuestion: What kind of forest is the Amazon?\nAnswer:"
    },
    {
        "name": "Stage 3: Logical Inference",
        "dataset_name": "tasksource/LogicNLI",
        "dataset_split": "train",
        "num_samples": 15000,
        "num_epochs": 3,
        "learning_rate": 1e-4,
        "formatting_fn": format_logicnli,
        "generation_prompt": "Premise: If it is raining, the ground is wet.\nHypothesis: It is not raining.\nLabel:"
    }
]

# --- Main Training Loop ---
model = TinyQwen3().to(device)
model_file_path = "tinyqwen3_model.pth"

# Check if a partially trained model exists
start_stage_idx = 0
if os.path.exists(model_file_path):
    print(f"Loading pre-trained model from {model_file_path}")
    model.load_state_dict(torch.load(model_file_path))
    # You could add logic here to figure out which stage to resume from if needed
    
for i, stage in enumerate(training_stages):
    print("\n" + "="*50)
    print(f"STARTING: {stage['name']}")
    print("="*50)

    # 1. Load and prepare dataset for the current stage
    print("Loading and preparing dataset...")
    dataset_name = stage['dataset_name']
    dataset_config = stage.get('dataset_config', None) # .get() handles missing keys gracefully
    
    # Load the dataset from Hugging Face
    # Use trust_remote_code=True to handle legacy dataset formats
    try:
        full_dataset = load_dataset(dataset_name, name=dataset_config, split=stage['dataset_split'], trust_remote_code=True)
    except Exception as e:
        print(f"Failed to load dataset with trust_remote_code=True, trying without it...")
        full_dataset = load_dataset(dataset_name, name=dataset_config, split=stage['dataset_split'])
    
    # Select a subset of data for faster training
    # Make sure we don't request more samples than available
    num_samples = min(stage['num_samples'], len(full_dataset))
    print(f"Using {num_samples} samples (requested: {stage['num_samples']}, available: {len(full_dataset)})")
    subset_data = full_dataset.select(range(num_samples))

    # Create the custom PyTorch Dataset and DataLoader
    train_dataset = CurriculumDataset(subset_data, tokenizer, context_length, vocab_size, stage['formatting_fn'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 2. Set up optimizer for the current stage
    optimizer = torch.optim.AdamW(model.parameters(), lr=stage['learning_rate'])
    num_epochs = stage['num_epochs']

    # 3. Run the training loop for the current stage
    print("Starting training...")
    for epoch in range(num_epochs):
        for step, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            
            logits, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            
            if step % 200 == 0:
                print(f"Stage {i+1}, Epoch {epoch+1}/{num_epochs}, Step {step}, Loss: {loss.item():.4f}")

    print("Stage finished.")

    # 4. Save model and generate example text
    print("Saving model...")
    torch.save(model.state_dict(), f"tinyqwen3_stage_{i+1}.pth") # Save stage-specific model
    torch.save(model.state_dict(), model_file_path) # Overwrite main model file

    print(f"\n--- Generating text after {stage['name']} ---")
    prompt = stage['generation_prompt']
    start_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    # Ensure tokens are within our vocabulary size
    start_tokens = start_tokens % vocab_size
    generated_tokens = model.generate(start_tokens, max_new_tokens=50)
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    print(generated_text)

print("\nFull curriculum training finished.")
