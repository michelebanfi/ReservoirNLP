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
# Set to 'math' for SimpleMath dataset, 'story' for TinyStories dataset
TASK_TYPE = 'math'  # Options: 'math' or 'story'

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
    DATASET_SIZE: int  # Maximum number of examples to use for training

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
    LOG_INTERVAL=20,
    DATASET_SIZE=500  # Small dataset for quick testing
)

# Larger model configuration for training on GPU
FULL_CONFIG = ModelConfig(
    VOCAB_SIZE=10000,
    EMBED_DIM=256,
    NUM_HEADS=4,
    NUM_LAYERS=3,
    BLOCK_SIZE=128,
    DROPOUT=0.1,
    BATCH_SIZE=16,
    LEARNING_RATE=3e-4,
    NUM_TRAIN_STEPS=50000,
    LOG_INTERVAL=50,
    DATASET_SIZE=80000  # Larger dataset for full training
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
DATASET_SIZE = CONFIG.DATASET_SIZE

# --- 1. The Custom Mathematical Operator Block ---
# This block has no learnable parameters. Its purpose is to apply fixed,
# deterministic mathematical operations to the input tensor.

class MathematicalOperatorBlock(nn.Module):
    """
    Previously gradients were blocked with torch.no_grad(), making this
    addition almost inert for learning. Now gradients flow. A single
    scalar gate lets the network learn to suppress the block.
    """
    def __init__(self):
        super().__init__()
        # Gate (trainable); initialized near 0 so model can opt-in gradually.
        self.gate = nn.Parameter(torch.zeros(1))

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
        op1_add = chunk1 + chunk2          # linear mix
        op2_sub = chunk1 - chunk2          # contrast
        op3_mul = chunk1 * chunk3          # quadratic interaction
        op4_const = torch.full_like(chunk1, 0.5)  # constant bias channel

        result = torch.cat([op1_add, op2_sub, op3_mul, op4_const], dim=-1)

        # Normalize to avoid blowing up residual variance after concat.
        result = result - result.mean(dim=-1, keepdim=True)
        result = result / (result.std(dim=-1, keepdim=True) + 1e-6)

        # Apply small gate (sigmoid keeps it in [0,1])
        return torch.sigmoid(self.gate) * result

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
            nn.SiLU(),
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

# --- Standard Transformer Block (without Math Layer) ---
class StandardTransformerBlock(nn.Module):
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

    def forward(self, x):
        # Standard attention path with residual connection
        x = x + self.attn(self.ln_1(x))
        # Standard MLP path with residual connection
        x = x + self.mlp(self.ln_2(x))
        return x

# --- 3. The Full Models ---

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

class StandardTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, embed_dim)
        self.blocks = nn.Sequential(*[StandardTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
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

# --- SimpleMath Dataset Handler ---
class SimpleMathDataset:
    """Dataset handler for the SimpleMath arithmetic dataset"""
    def __init__(self, tokenizer, split='train', max_length=BLOCK_SIZE):
        # Load the dataset
        self.dataset = load_dataset("ProCreations/SimpleMath", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Handle different column name possibilities
        if 'instruction' in item:
            question = item['instruction']
            answer = str(item['output'])
        elif 'text' in item:
            text = item['text']
            parts = text.split('=')
            if len(parts) == 2:
                question = parts[0].strip() + ' ='
                answer = parts[1].strip()
            else:
                return None
        else:
            # Use first two columns
            cols = list(item.keys())
            if len(cols) >= 2:
                question = str(item[cols[0]])
                answer = str(item[cols[1]])
            else:
                return None
        
        # Create input-output pair
        full_text = f"{question} {answer}"
        
        # Tokenize
        tokens = self.tokenizer.encode(full_text, return_tensors="pt", 
                                      max_length=self.max_length, truncation=True)[0]
        
        if len(tokens) < 2:
            return None
            
        return tokens[:-1], tokens[1:], answer  # input, target, numeric_answer
    
    def get_batch(self, batch_size):
        """Get a batch of data"""
        batch_data = []
        for i in range(batch_size):
            if i < len(self.dataset):
                item = self.__getitem__(i)
                if item is not None:
                    batch_data.append(item)
        return batch_data

def load_simple_math_data(tokenizer, batch_size, max_length=BLOCK_SIZE):
    """Load SimpleMath dataset and prepare batches"""
    # Load dataset with size limit
    dataset = load_dataset("ProCreations/SimpleMath", split=f"train[:{DATASET_SIZE}]")
    data = []
    
    # Print column names to debug
    if len(dataset) > 0:
        print(f"Dataset columns: {dataset.column_names}")
        print(f"First example: {dataset[0]}")
    
    for example in dataset:
        # The dataset has columns based on the structure shown in the attachment
        # It appears to be a simple format with the math problem and answer
        # Try to access the actual column names
        if 'instruction' in example:
            question = example['instruction']
            answer = str(example['output'])
        elif 'text' in example:
            # Sometimes datasets have 'text' column
            text = example['text']
            # Parse text to extract question and answer
            parts = text.split('=')
            if len(parts) == 2:
                question = parts[0].strip() + ' ='
                answer = parts[1].strip()
            else:
                continue
        else:
            # Try to find the first two columns (likely question and answer)
            cols = list(example.keys())
            if len(cols) >= 2:
                question = str(example[cols[0]])
                answer = str(example[cols[1]])
            else:
                print(f"Warning: Unknown dataset format. Columns: {cols}")
                continue
        
        # Format the text as "question answer"
        full_text = f"{question} {answer}"
        
        # Tokenize
        tokens = tokenizer.encode(full_text, return_tensors="pt", 
                                 max_length=max_length, truncation=True)[0]
        
        if len(tokens) < 2:
            continue
        
        # Store (input, target, numeric_answer)
        try:
            numeric_answer = int(''.join(filter(str.isdigit, answer)))
        except ValueError:
            # If we can't parse the answer as int, skip this example
            continue
            
        data.append((tokens[:-1], tokens[1:], numeric_answer))
    
    print(f"Loaded {len(data)} examples from SimpleMath dataset")
    
    # Create batches
    batches = []
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i+batch_size]
        if not batch_data:
            continue
        
        inputs = [item[0] for item in batch_data]
        targets = [item[1] for item in batch_data]
        answers = [item[2] for item in batch_data]
        
        inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
        targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1)
        
        batches.append((inputs_padded, targets_padded, answers))
    
    return batches

# --- Alternative data loading approach to avoid threading issues ---
def load_data_directly(tokenizer, batch_size, num_steps, max_length=BLOCK_SIZE):
    """Load data directly without using DataLoader to avoid threading issues"""
    # Load dataset with explicit size limit based on configuration
    dataset = load_dataset("roneneldan/TinyStories", split=f"train[:{DATASET_SIZE}]", streaming=False)
    data = []
    
    # Process all loaded examples
    for example in dataset:
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

# --- Evaluation Utilities ---
def extract_number_from_tokens(tokens, tokenizer):
    """Extract the first number from a sequence of tokens"""
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    # Extract numbers from text
    import re
    numbers = re.findall(r'\d+', text)
    if numbers:
        try:
            return int(numbers[0])
        except ValueError:
            return None
    return None

def calculate_math_accuracy(model, data_batches, tokenizer, device):
    """Calculate numerical accuracy for math predictions"""
    model.eval()
    total_distance = 0
    total_correct = 0
    total_samples = 0
    distances = []
    
    with torch.no_grad():
        for batch_data in data_batches:
            if len(batch_data) == 3:  # Math dataset with answers
                inputs, targets, true_answers = batch_data
            else:
                continue
                
            inputs = inputs.to(device)
            logits, _ = model(inputs)
            
            # Get predicted tokens (greedy decoding)
            predicted_tokens = torch.argmax(logits, dim=-1)
            
            # For each sample in batch
            for i in range(len(true_answers)):
                pred_tokens = predicted_tokens[i].cpu().tolist()
                predicted_num = extract_number_from_tokens(pred_tokens, tokenizer)
                true_num = true_answers[i]
                
                if predicted_num is not None:
                    distance = abs(predicted_num - true_num)
                    total_distance += distance
                    distances.append(distance)
                    
                    if predicted_num == true_num:
                        total_correct += 1
                
                total_samples += 1
    
    if total_samples == 0:
        return {'accuracy': 0, 'avg_distance': 0, 'distances': []}
    
    accuracy = (total_correct / total_samples) * 100
    avg_distance = total_distance / total_samples
    
    return {
        'accuracy': accuracy,
        'avg_distance': avg_distance,
        'distances': distances,
        'total_samples': total_samples,
        'total_correct': total_correct
    }

def calculate_perplexity(model, data_batches, device):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch_data in data_batches:
            # Handle both story and math batches
            if len(batch_data) == 3:  # Math dataset
                inputs, targets, _ = batch_data
            else:  # Story dataset
                inputs, targets = batch_data
                
            inputs, targets = inputs.to(device), targets.to(device)
            _, loss = model(inputs, targets)
            
            # Count non-padding tokens
            non_pad_mask = (targets != -1)
            num_tokens = non_pad_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Perplexity = exp(average negative log-likelihood)
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity

def compare_models(math_model, std_model, tokenizer, prompt, device, max_new_tokens=40):
    """Generate text using both models and show them side by side"""
    import time
    
    # Generate with Math model
    if device == 'cuda':
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        math_text = generate_text(math_model, tokenizer, prompt, max_new_tokens, device=device)
        end_time.record()
        torch.cuda.synchronize()
        math_time = start_time.elapsed_time(end_time) / 1000  # convert to seconds
        
        # Generate with Standard model
        start_time.record()
        std_text = generate_text(std_model, tokenizer, prompt, max_new_tokens, device=device)
        end_time.record()
        torch.cuda.synchronize()
        std_time = start_time.elapsed_time(end_time) / 1000  # convert to seconds
    else:
        # Use time.time() for CPU
        start = time.time()
        math_text = generate_text(math_model, tokenizer, prompt, max_new_tokens, device=device)
        math_time = time.time() - start
        
        start = time.time()
        std_text = generate_text(std_model, tokenizer, prompt, max_new_tokens, device=device)
        std_time = time.time() - start
    
    print(f"=== Generation Comparison (Prompt: '{prompt}') ===")
    print(f"Math Model ({math_time:.3f}s): '{math_text}'")
    print(f"Std Model ({std_time:.3f}s): '{std_text}'")

# --- 5. Main Training Script ---

def main():
    print("--- Starting Comparison Test: Math vs Standard Transformer ---")
    print(f"Task Type: {TASK_TYPE}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Set up tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    effective_vocab_size = tokenizer.vocab_size

    # Set up both models
    print("Initializing models...")
    math_model = TinyTransformer(
        vocab_size=effective_vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)
    
    std_model = StandardTransformer(
        vocab_size=effective_vocab_size,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS
    ).to(device)
    
    print(f"Math Model has {sum(p.numel() for p in math_model.parameters() if p.requires_grad):,} trainable parameters.")
    print(f"Standard Model has {sum(p.numel() for p in std_model.parameters() if p.requires_grad):,} trainable parameters.")
    print(f"Using {'LOCAL' if LOCAL_TESTING else 'FULL'} configuration:")
    print(f"  - Dataset size: {DATASET_SIZE} examples")
    print(f"  - Model size: EMBED_DIM={EMBED_DIM}, NUM_LAYERS={NUM_LAYERS}, NUM_HEADS={NUM_HEADS}")

    # Load data based on task type
    if TASK_TYPE == 'math':
        print("\nLoading SimpleMath dataset...")
        batches = load_simple_math_data(tokenizer, BATCH_SIZE)
    else:
        print("\nLoading TinyStories dataset...")
        batches = load_data_directly(tokenizer, BATCH_SIZE, NUM_TRAIN_STEPS)
    
    # Split batches for training and evaluation
    train_ratio = 0.9
    train_size = int(len(batches) * train_ratio)
    train_batches = batches[:train_size]
    eval_batches = batches[train_size:]
    
    # Store metrics for comparison
    model_metrics = {
        'math_model': {'train_losses': [], 'train_time': 0, 'perplexity': 0},
        'std_model': {'train_losses': [], 'train_time': 0, 'perplexity': 0}
    }

    # --- Train Math Model ---
    print("\n=== Training Math-Augmented Transformer ===")
    model_metrics['math_model'] = train_model(math_model, train_batches, eval_batches, device, tokenizer)
    
    # --- Train Standard Model ---
    print("\n=== Training Standard Transformer ===")
    model_metrics['std_model'] = train_model(std_model, train_batches, eval_batches, device, tokenizer)
    
    # --- Compare Results ---
    print("\n=== Comparison Results ===")
    print(f"Math Model - Training time: {model_metrics['math_model']['train_time']:.2f}s, "
          f"Final Loss: {model_metrics['math_model']['train_losses'][-1]:.4f}, "
          f"Perplexity: {model_metrics['math_model']['perplexity']:.2f}")
    
    if TASK_TYPE == 'math' and 'math_accuracy' in model_metrics['math_model']:
        math_acc = model_metrics['math_model']['math_accuracy']
        print(f"  Math Accuracy: {math_acc['accuracy']:.2f}%, Avg Distance: {math_acc['avg_distance']:.2f}")
          
    print(f"Standard Model - Training time: {model_metrics['std_model']['train_time']:.2f}s, "
          f"Final Loss: {model_metrics['std_model']['train_losses'][-1]:.4f}, "
          f"Perplexity: {model_metrics['std_model']['perplexity']:.2f}")
    
    if TASK_TYPE == 'math' and 'math_accuracy' in model_metrics['std_model']:
        std_acc = model_metrics['std_model']['math_accuracy']
        print(f"  Math Accuracy: {std_acc['accuracy']:.2f}%, Avg Distance: {std_acc['avg_distance']:.2f}")
    
    # Compare generated text
    if TASK_TYPE == 'math':
        prompts = [
            "45 + 60 =",
            "123 + 456 =",
            "1000 + 2000 ="
        ]
    else:
        prompts = [
            "Once upon a time", 
            "The little dog", 
            "In a world where"
        ]
    
    for prompt in prompts:
        compare_models(math_model, std_model, tokenizer, prompt, device)
    
    # Plot training losses and math metrics if available
    try:
        import matplotlib.pyplot as plt
        
        # Training loss comparison
        plt.figure(figsize=(10, 6))
        plt.plot(model_metrics['math_model']['train_losses'], label='Math Model')
        plt.plot(model_metrics['std_model']['train_losses'], label='Standard Model')
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.savefig('loss_comparison.png')
        print("\nSaved training loss comparison chart to 'loss_comparison.png'")
        
        # If math task, plot distance distributions
        if TASK_TYPE == 'math':
            if 'math_accuracy' in model_metrics['math_model'] and 'math_accuracy' in model_metrics['std_model']:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.hist(model_metrics['math_model']['math_accuracy']['distances'], bins=50, alpha=0.7, label='Math Model')
                plt.xlabel('Distance from True Answer')
                plt.ylabel('Frequency')
                plt.title('Math Model: Prediction Distance Distribution')
                plt.legend()
                
                plt.subplot(1, 2, 2)
                plt.hist(model_metrics['std_model']['math_accuracy']['distances'], bins=50, alpha=0.7, label='Standard Model', color='orange')
                plt.xlabel('Distance from True Answer')
                plt.ylabel('Frequency')
                plt.title('Standard Model: Prediction Distance Distribution')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig('math_accuracy_comparison.png')
                print("Saved math accuracy comparison chart to 'math_accuracy_comparison.png'")
    except ImportError:
        print("\nMatplotlib not available - skipping plot generation")
    
    # Force cleanup
    import gc
    del batches, math_model, std_model
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print("\n--- Comparison Finished ---")
    sys.exit(0)

def train_model(model, train_batches, eval_batches, device, tokenizer):
    """Train a model and return metrics"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Initialize metrics
    metrics = {'train_losses': [], 'train_time': 0, 'perplexity': 0}
    
    # Training loop
    model.train()
    total_loss = 0
    step_count = 0
    
    # Timing
    start_time = time.time()
    
    for batch_data in train_batches:
        # Handle both story and math batches
        if len(batch_data) == 3:  # Math dataset
            inputs, targets, _ = batch_data
        else:  # Story dataset
            inputs, targets = batch_data
            
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        _, loss = model(inputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        total_loss += loss_val
        metrics['train_losses'].append(loss_val)
        step_count += 1

        if step_count % LOG_INTERVAL == 0:
            avg_loss = total_loss / LOG_INTERVAL
            print(f"Step {step_count}/{len(train_batches)} | Loss: {avg_loss:.4f}")
            total_loss = 0
    
    # Record training time
    metrics['train_time'] = time.time() - start_time
    
    # Calculate perplexity on eval set
    perplexity = calculate_perplexity(model, eval_batches, device)
    metrics['perplexity'] = perplexity
    print(f"Evaluation - Perplexity: {perplexity:.2f}")
    
    # If math task, calculate math-specific metrics
    if TASK_TYPE == 'math':
        math_metrics = calculate_math_accuracy(model, eval_batches, tokenizer, device)
        metrics['math_accuracy'] = math_metrics
        print(f"Math Accuracy: {math_metrics['accuracy']:.2f}% | "
              f"Avg Distance: {math_metrics['avg_distance']:.2f} | "
              f"Correct: {math_metrics['total_correct']}/{math_metrics['total_samples']}")
    
    return metrics

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
    
    # For timing measurements
    import time
    import sys
    
    main()
