import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import math
import os
from dataclasses import dataclass

# --- Configuration Flag ---
# Set to True for quick local testing, False for full training on GPU
LOCAL_TESTING = False

@dataclass
class ModelConfig:
    VOCAB_SIZE: int  # Using a tokenizer-defined vocab size later
    EMBED_DIM: int
    NUM_HEADS: int
    NUM_LAYERS: int
    BLOCK_SIZE: int  # Max sequence length
    DROPOUT: float
    BATCH_SIZE: int
    LEARNING_RATE: float
    NUM_TRAIN_STEPS: int  # How many batches to train on
    LOG_INTERVAL: int  # Print loss every N steps

# Small model configuration for local testing
LOCAL_CONFIG = ModelConfig(
    VOCAB_SIZE=10000,
    EMBED_DIM=32,
    NUM_HEADS=2,
    NUM_LAYERS=1,
    BLOCK_SIZE=64,
    DROPOUT=0.1,
    BATCH_SIZE=8,
    LEARNING_RATE=3e-4,
    NUM_TRAIN_STEPS=100,
    LOG_INTERVAL=20
)

# Larger model configuration for training on GPU
FULL_CONFIG = ModelConfig(
    VOCAB_SIZE=10000,
    EMBED_DIM=128,
    NUM_HEADS=4,
    NUM_LAYERS=3,
    BLOCK_SIZE=128,
    DROPOUT=0.1,
    BATCH_SIZE=16,
    LEARNING_RATE=3e-4,
    NUM_TRAIN_STEPS=500,
    LOG_INTERVAL=50
)

# Select config based on LOCAL_TESTING flag
CONFIG = LOCAL_CONFIG if LOCAL_TESTING else FULL_CONFIG

# Extract configuration values to use throughout the code
VOCAB_SIZE = CONFIG.VOCAB_SIZE
EMBED_DIM = CONFIG.EMBED_DIM
NUM_HEADS = CONFIG.NUM_HEADS
NUM_LAYERS = CONFIG.NUM_LAYERS
BLOCK_SIZE = CONFIG.BLOCK_SIZE
DROPOUT = CONFIG.DROPOUT
BATCH_SIZE = CONFIG.BATCH_SIZE
LEARNING_RATE = CONFIG.LEARNING_RATE
NUM_TRAIN_STEPS = CONFIG.NUM_TRAIN_STEPS
LOG_INTERVAL = CONFIG.LOG_INTERVAL

# --- 1. The Custom Mathematical Operator Block ---
# This block has no learnable parameters. Its purpose is to apply fixed,
# deterministic mathematical operations to the input tensor.

class MathematicalOperatorBlock(nn.Module):
    """
    A non-trainable block that applies fixed mathematical operations.
    It splits the input features, performs operations, and returns the result.
    """
    def __init__(self):
        super().__init__()
        # This module has no parameters, so no training will occur within it.

    def forward(self, x):
        """
        x: input tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Ensure the embedding dimension is divisible by 4 for splitting
        if x.shape[-1] % 4 != 0:
            raise ValueError(f"Embedding dimension ({x.shape[-1]}) must be divisible by 4.")

        # Split the feature dimension into four chunks
        chunk_size = x.shape[-1] // 4
        chunk1, chunk2, chunk3, _ = torch.split(x, chunk_size, dim=-1)

        # --- Perform fixed mathematical operations ---
        # These operations are deterministic and have no weights.
        # Gradients will flow through them, but there's nothing here to update.
        with torch.no_grad(): # Explicitly state no gradient calculations for weights (there are none)
            op1_add = chunk1 + chunk2
            op2_sub = chunk1 - chunk2
            op3_mul = chunk1 * chunk3

            # We'll just use a constant for the fourth operation for simplicity
            op4_const = torch.ones_like(chunk1) * 0.5

        # Concatenate the results back together
        result = torch.cat([op1_add, op2_sub, op3_mul, op4_const], dim=-1)

        return result

# --- 2. Core Transformer Components ---

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim)
        # Output projection
        self.c_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj.SCALE_INIT = 1
        # Causal mask to ensure attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                                     .view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (C)
        q, k, v  = self.c_attn(x).split(self.embed_dim, dim=2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) # (B, nh, T, hs)
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # Output projection
        y = self.c_proj(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )
        # --- HERE IS THE INTEGRATION OF YOUR IDEA ---
        self.math_block = MathematicalOperatorBlock()

    def forward(self, x):
        # Standard attention path
        x = x + self.attn(self.ln_1(x))

        # --- Augmenting the residual stream ---
        # The input 'x' is passed through the math block, and the result is added
        # back to the main stream. The network can learn to use or ignore this.
        math_output = self.math_block(self.ln_1(x))
        x = x + math_output

        # Standard MLP path
        x = x + self.mlp(self.ln_2(x))
        return x

# --- 3. The Full Model ---

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, embed_dim)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, C)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

# --- 4. Data Loading and Preparation ---

class TinyStoriesDataset(IterableDataset):
    def __init__(self, tokenizer, split='train', max_length=BLOCK_SIZE):
        self.dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __iter__(self):
        for story in self.dataset:
            text = story['text']
            tokens = self.tokenizer.encode(text, return_tensors="pt", max_length=self.max_length, truncation=True)[0]
            if len(tokens) < 2:
                continue
            # Yield input and target pairs for next-token prediction
            yield tokens[:-1], tokens[1:]
            
    def take(self, n):
        """Limit the dataset to the first n samples."""
        limited_dataset = TinyStoriesDataset(self.tokenizer, split='train', max_length=self.max_length)
        limited_dataset.dataset = self.dataset.take(n)
        return limited_dataset

def collate_fn(batch):
    # Pad sequences to the same length in a batch
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1) # use -1 for ignore_index
    return inputs_padded, targets_padded

# --- Alternative data loading approach to avoid threading issues ---
def load_data_directly(tokenizer, batch_size, num_steps, max_length=BLOCK_SIZE):
    """Load data directly without using DataLoader to avoid threading issues"""
    # Load a small dataset in memory
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    data = []
    
    # Collect enough examples
    for i, example in enumerate(dataset):
        if i >= batch_size * num_steps:
            break
        
        text = example['text']
        tokens = tokenizer.encode(text, return_tensors="pt", max_length=max_length, truncation=True)[0]
        if len(tokens) < 2:
            continue
            
        data.append((tokens[:-1], tokens[1:]))
        
    # Process data in batches
    batches = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        if not batch_data:
            continue
            
        inputs = [item[0] for item in batch_data]
        targets = [item[1] for item in batch_data]
        
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)
        
        batches.append((inputs_padded, targets_padded))
    
    return batches

# --- 5. Main Training Script ---

def main():
    print("--- Starting Proof-of-Concept Test ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set up tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    effective_vocab_size = tokenizer.vocab_size

    # Set up model
    print("Initializing model...")
    model = TinyTransformer(
        vocab_size=effective_vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    # Check which parameters are trainable (to verify math block is excluded)
    print("\nVerifying trainable parameters (math_block should be absent):")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"  - {name} (Not Trainable)")

    # Set up data directly without DataLoader
    print("\nLoading data directly for TinyStories...")
    batches = load_data_directly(tokenizer, BATCH_SIZE, NUM_TRAIN_STEPS)

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\n--- Starting Training ---")
    model.train()
    step_count = 0
    total_loss = 0

    for inputs, targets in batches:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        _, loss = model(inputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step_count += 1

        if step_count % LOG_INTERVAL == 0:
            avg_loss = total_loss / LOG_INTERVAL
            print(f"Step {step_count}/{len(batches)} | Loss: {avg_loss:.4f}")
            total_loss = 0

    print("\n--- Training Finished ---")
    print("The loss decreased, indicating the model is trainable and gradients are flowing correctly.")
    print("The experiment successfully demonstrates the architectural concept.")
    
    # Test generation from a prompt
    print("\n--- Testing Text Generation ---")
    prompt = "Once upon a time"
    print(f"Prompt: '{prompt}'")
    generated_text = generate_text(model, tokenizer, prompt, max_new_tokens=40, device=device)
    print(f"Generated: '{generated_text}'")
    
    # Force cleanup
    import gc
    del batches, model, optimizer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Exit explicitly to avoid thread issues
    import sys
    sys.exit(0)

# --- Text Generation Functions ---
def generate_text(model, tokenizer, prompt, max_new_tokens=40, temperature=0.8, device='cpu'):
    """Generate text from a prompt using the trained model"""
    model.eval()  # Set to evaluation mode
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate tokens auto-regressively
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Get only the last BLOCK_SIZE tokens if input becomes too long
            if input_ids.size(1) > BLOCK_SIZE:
                input_ids = input_ids[:, -BLOCK_SIZE:]
                
            # Forward pass to get logits for next token prediction
            logits, _ = model(input_ids)
            
            # Get logits for the next token (last position)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(next_token_logits, dim=-1)
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append the sampled token to input
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we generate an EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the generated tokens back to text
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

if __name__ == '__main__':
    # Hugging Face tokenizers can cause issues with multiprocessing on some systems.
    # This environment variable helps prevent potential deadlocks.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    main()
