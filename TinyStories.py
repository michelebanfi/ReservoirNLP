import os
import torch

# -----------------------------------------------------------------------------
# Shared Configuration for Both Models
# -----------------------------------------------------------------------------

SHARED_CONFIG = {
    # Data parameters
    'MAX_STORIES': 2000,                # Number of stories to use for training
    
    # Training parameters
    'EPOCHS': 4,                       # Number of training epochs
    'BATCH_SIZE': 32,                  # Batch size for training
    'BLOCK_SIZE': 128,                 # Context size for training
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Evaluation parameters
    'EVAL_INTERVAL': 1000,              # Steps between evaluations
    'EVAL_ITER': 50,                   # Number of batches for evaluation
    'MAX_OUT_TOKENS': 100,             # Number of tokens to generate for samples
    
    # File paths
    'RESERVOIR_SAVE_PATH': 'models/deep_reservoir_trained.pt',
    'TRANSFORMER_SAVE_PATH': 'models/tiny_lm_trained.pt'
}

print(f"Using device: {SHARED_CONFIG['DEVICE']}")
os.makedirs("models", exist_ok=True)

# -----------------------------------------------------------------------------
# Import Common Libraries
# -----------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
from transformers import GPT2TokenizerFast
from datasets import load_dataset
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Data Download and Tokenization
# -----------------------------------------------------------------------------

# Initialize tokenizer
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def download_and_save_dataset(max_stories=None):
    """Downloads the TinyStories dataset and saves a subset if specified."""
    data_dir = "data"
    if max_stories:
        train_path = os.path.join(data_dir, f"TinyStories-train-{max_stories}.txt")
        valid_path = os.path.join(data_dir, f"TinyStories-valid-{max_stories}.txt")
    else:
        train_path = os.path.join(data_dir, "TinyStories-train.txt")
        valid_path = os.path.join(data_dir, "TinyStories-valid.txt")

    if os.path.exists(train_path) and os.path.exists(valid_path):
        print(f"Dataset files already exist: {train_path}, {valid_path}")
        return train_path, valid_path

    print("Downloading TinyStories dataset from Hugging Face...")
    os.makedirs(data_dir, exist_ok=True)
    ds = load_dataset("roneneldan/TinyStories")

    print(f"Saving training split to {train_path}...")
    with open(train_path, 'w', encoding='utf-8') as f:
        for i, story in enumerate(tqdm(ds['train'])):
            if max_stories and i >= max_stories:
                break
            f.write(story['text'] + '\n')

    print(f"Saving validation split to {valid_path}...")
    with open(valid_path, 'w', encoding='utf-8') as f:
        val_stories_to_save = max_stories // 10 if max_stories else None
        for i, story in enumerate(tqdm(ds['validation'])):
            if val_stories_to_save and i >= val_stories_to_save:
                break
            f.write(story['text'] + '\n')
    return train_path, valid_path

def pre_tokenize_dataset(path, save_path):
    print(f"Running tokenization for {path}...")
    with open(path, 'r', encoding='utf-8') as file:
        text = file.read()
        tokens = tokenizer.encode(text)
        np.save(save_path, np.array(tokens, dtype=np.int32))
        print(f"Saved tokenized file to binary {save_path}")

class TinyStoriesDataset(data.Dataset):
    def __init__(self, tokenized_path, block_size: int):
        self.block_size = block_size
        self.data = np.load(tokenized_path, mmap_mode='r')

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        source = torch.from_numpy(chunk[:-1].astype(np.int64))
        target = torch.from_numpy(chunk[1:].astype(np.int64))
        return source, target
    
    # -----------------------------------------------------------------------------
# Download and Prepare Dataset
# -----------------------------------------------------------------------------

# Download dataset
train_txt_path, val_txt_path = download_and_save_dataset(max_stories=SHARED_CONFIG["MAX_STORIES"])

# Tokenize training data
train_tokenized_path = train_txt_path.replace('.txt', '.npy')
if not os.path.exists(train_tokenized_path):
    pre_tokenize_dataset(train_txt_path, train_tokenized_path)

# Tokenize validation data
val_tokenized_path = val_txt_path.replace('.txt', '.npy')
if not os.path.exists(val_tokenized_path):
    pre_tokenize_dataset(val_txt_path, val_tokenized_path)

# Create data loaders
train_dataset = TinyStoriesDataset(train_tokenized_path, SHARED_CONFIG['BLOCK_SIZE'])
train_loader = data.DataLoader(train_dataset, batch_size=SHARED_CONFIG['BATCH_SIZE'], shuffle=True)

val_dataset = TinyStoriesDataset(val_tokenized_path, SHARED_CONFIG['BLOCK_SIZE'])
val_loader = data.DataLoader(val_dataset, batch_size=SHARED_CONFIG['BATCH_SIZE'])

print(f"Data preparation complete! Tokenizer vocabulary size: {tokenizer.vocab_size}")


class OptimizedParallelReservoir(nn.Module):
    """
    Memory and computation optimized reservoir with gradient checkpointing support.
    """
    def __init__(self, input_size, hidden_size, window_size, spectral_radius=1.2,
                 sparsity=0.1, activation='tanh', device='cuda', leak_rate=0.95):
        super().__init__()
        self.window_size = window_size
        self.device = device
        self.hidden_size = hidden_size
        
        # Use smaller precision for reservoir weights to save memory
        self.projection = nn.Linear(input_size * window_size, hidden_size, bias=False, dtype=torch.float16).to(device)
        self._initialize_weights(spectral_radius, sparsity)
        
        # This simulates a leaky memory by weighting recent tokens in the window more heavily.
        # It's pre-computed and registered as a buffer to be moved to the device automatically.
        if leak_rate < 1.0:
            decay_weights = torch.pow(leak_rate, torch.arange(window_size - 1, -1, -1, device=self.device))
            self.register_buffer('decay_weights', decay_weights.view(1, 1, 1, -1))
        else:
            self.register_buffer('decay_weights', None)
        
        # Precompute activation function
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'leaky_relu':
            self.activation = lambda x: F.leaky_relu(x, 0.01, inplace=True)  # Small negative slope, inplace
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError("Unsupported activation function")

    def _initialize_weights(self, spectral_radius, sparsity):
        # Freeze reservoir weights
        for param in self.projection.parameters():
            param.requires_grad = False

        with torch.no_grad():
            W = self.projection.weight.data
            
            # More efficient spectral radius normalization
            if W.shape[0] == W.shape[1]:
                # Use power iteration for large matrices (more efficient than full eigendecomposition)
                if W.shape[0] > 512:
                    current_radius = self._power_iteration_spectral_radius(W)
                else:
                    eigenvalues = torch.linalg.eigvals(W)
                    current_radius = torch.max(torch.abs(eigenvalues))
                W *= spectral_radius / current_radius
            
            # Apply sparsity mask
            mask = (torch.rand(W.shape, device=self.device) > sparsity)
            W *= mask.to(W.dtype)

    def _power_iteration_spectral_radius(self, W, num_iterations=10):
        """Estimate spectral radius using power iteration (more efficient for large matrices)"""
        v = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
        for _ in range(num_iterations):
            v = torch.mv(W, v)
            v = v / torch.norm(v)
        return torch.norm(torch.mv(W, v))

    def forward(self, x):
        # More memory-efficient windowing using stride tricks
        batch_size, seq_len, input_size = x.shape
        
        # Pad only what we need
        if self.window_size > 1:
            padding = (0, 0, self.window_size - 1, 0)
            x_padded = F.pad(x, padding, "constant", 0)
        else:
            x_padded = x
            
        # Use unfold for efficient windowing
        windows = x_padded.unfold(1, self.window_size, 1)
        
        if self.decay_weights is not None:
            # The shape of windows is (batch, seq_len, input_size, window_size).
            # We multiply by our decay weights along the window_size dimension.
            windows = windows * self.decay_weights
        
        windows_flat = windows.contiguous().view(batch_size, seq_len, -1)
        
        # Convert to float16 for computation, back to float32 for output
        windows_flat = windows_flat.to(torch.float16)
        output = self.activation(self.projection(windows_flat))
        return output.to(torch.float32)


class OptimizedReservoirBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reservoirs = nn.ModuleList()
        total_reservoir_size = 0
        
        for res_config in config['reservoirs_per_block']:
            self.reservoirs.append(
                OptimizedParallelReservoir(
                    input_size=config['embedding_dim'],
                    hidden_size=res_config['reservoir_size'],
                    window_size=res_config['window_size'],
                    spectral_radius=res_config['spectral_radius'],
                    sparsity=res_config.get('sparsity', 0.1),
                    activation=res_config.get('activation', 'gelu'),  # GELU often works better
                    device=config['DEVICE']
                )
            )
            total_reservoir_size += res_config['reservoir_size']

        # Simplified readout with better initialization
        self.readout = nn.Sequential(
            nn.LayerNorm(total_reservoir_size),  # Add layer norm for stability
            nn.Linear(total_reservoir_size, config['readout_hidden_size']),
            nn.GELU(),
            nn.Dropout(config.get('dropout', 0.1)),  # Reduced dropout
            nn.Linear(config['readout_hidden_size'], config['embedding_dim'])
        )
        
        self.gate = nn.Sequential(
            nn.Linear(config['embedding_dim'], config['embedding_dim']),
            nn.Sigmoid()
        )
        
        # Initialize readout weights properly
        self._init_readout_weights()

    def _init_readout_weights(self):
        for module in self.readout.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        for module in self.gate.modules():
             if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # Use gradient checkpointing for memory efficiency during training
        if self.training and hasattr(torch.utils.checkpoint, 'checkpoint'):
            reservoir_states = [torch.utils.checkpoint.checkpoint(res, x, use_reentrant=False) 
                              for res in self.reservoirs]
        else:
            reservoir_states = [res(x) for res in self.reservoirs]
            
        combined_states = torch.cat(reservoir_states, dim=-1)
        update_vector = self.readout(combined_states)
        
        g = self.gate(x)
        return x * g + update_vector * (1.0 - g)


class OptimizedDeepReservoirModel(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        
        # Trainable embedding with proper initialization
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        
        # Add positional encoding for better sequence understanding
        self.pos_encoding = nn.Parameter(
            torch.randn(config['max_seq_len'], config['embedding_dim']) * 0.02
        )
        
        # Input projection to stabilize training
        self.input_proj = nn.Sequential(
            nn.LayerNorm(config['embedding_dim']),
            nn.Linear(config['embedding_dim'], config['embedding_dim']),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList([
            OptimizedReservoirBlock(config) for _ in range(config['num_blocks'])
        ])
        
        # Output head with better initialization
        self.output_norm = nn.LayerNorm(config['embedding_dim'])
        self.final_head = nn.Linear(config['embedding_dim'], vocab_size, bias=False)
        
        # Tie weights between embedding and output (common practice)
        if config.get('tie_weights', True):
            self.final_head.weight = self.embedding.weight
        
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module != self.final_head:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, idx):
        batch_size, seq_len = idx.shape
        
        # Embedding + positional encoding
        x_emb = self.embedding(idx)
        pos_emb = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = self.input_proj(x_emb + pos_emb)
        
        # The loop becomes simpler as each block now handles its own gated update.
        for block in self.blocks:
            x = block(x)  # The block now returns the updated x directly
        
        # Output projection
        x = self.output_norm(x)
        logits = self.final_head(x)
        return logits


@torch.no_grad()
def eval_reservoir_model(model, data_loader, config):
    """Evaluates the reservoir model with mixed precision."""
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    num_batches = 0
    
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        for x, y in data_loader:
            if num_batches >= config['EVAL_ITER']:
                break
            x, y = x.to(config['DEVICE']), y.to(config['DEVICE'])
            logits = model(x)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), y.view(B * T))
            total_loss += loss.item()
            num_batches += 1
    
    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


@torch.no_grad()
def generate_from_reservoir(model, context_str, max_new_tokens, config, tokenizer):
    """Generates text using temperature, top-k, and top-p (nucleus) sampling."""
    model.eval()
    start_indices = tokenizer.encode(context_str)
    context = torch.tensor(start_indices, dtype=torch.long, device=config['DEVICE']).unsqueeze(0)
    
    temperature = config.get('temperature', 0.8)
    top_k = config.get('top_k', 50)
    top_p = config.get('top_p', 0.9)

    for _ in range(max_new_tokens):
        current_context = context if context.size(1) <= config['BLOCK_SIZE'] else context[:, -config['BLOCK_SIZE']:]
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(current_context)
        
        logits = logits[:, -1, :] / temperature
        
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # --- THIS IS THE LINE TO FIX ---
            # We get a 1D tensor from boolean indexing, so we unsqueeze it to make it 2D.
            indices_to_remove = sorted_indices[sorted_indices_to_remove].unsqueeze(0)
            # --- END FIX ---

            logits.scatter_(1, indices_to_remove, float('-inf'))

        if top_k > 0:
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            filtered_logits = torch.full_like(logits, float('-inf'))
            filtered_logits.scatter_(1, top_k_indices, top_k_logits)
            logits = filtered_logits
        
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        
        context = torch.cat((context, idx_next), dim=1)

    model.train()
    return tokenizer.decode(context.squeeze().tolist())


def create_optimized_reservoir_config(shared_config, train_loader):
    """Create optimized reservoir config based on dataset size and shared config."""
    
    # Calculate training steps based on actual dataset
    steps_per_epoch = len(train_loader)
    
    # FIXED: Account for gradient accumulation in total steps calculation
    accumulation_steps = 4  # Define this first
    batches_per_step = accumulation_steps
    actual_steps_per_epoch = steps_per_epoch // batches_per_step
    total_steps = actual_steps_per_epoch * shared_config['EPOCHS']
    
    # Calculate warmup and scheduling parameters
    warmup_steps = min(1000, total_steps // 10)
    
    config = {
        # Model architecture
        'embedding_dim': 384,  # Slightly smaller for efficiency
        'num_blocks': 3,       # More blocks, each smaller
        'max_seq_len': shared_config['BLOCK_SIZE'] * 2,  # Based on your block size
        'tie_weights': True,   # Tie embedding and output weights
        'dropout': 0.1,
        
        'reservoirs_per_block': [
            # Added 'leak_rate' to each reservoir's config
            {'reservoir_size': 192, 'window_size': min(64, shared_config['BLOCK_SIZE']//2), 
             'spectral_radius': 0.95, 'sparsity': 0.2, 'activation': 'gelu', 'leak_rate': 0.95},
            {'reservoir_size': 256, 'window_size': min(32, shared_config['BLOCK_SIZE']//4), 
             'spectral_radius': 1.05, 'sparsity': 0.15, 'activation': 'gelu', 'leak_rate': 0.9},
        ],
        'readout_hidden_size': 192,
        
        # Training params from shared config
        'BATCH_SIZE': shared_config['BATCH_SIZE'],
        'BLOCK_SIZE': shared_config['BLOCK_SIZE'],
        'EVAL_INTERVAL': shared_config['EVAL_INTERVAL'],
        'EVAL_ITER': shared_config['EVAL_ITER'],
        'DEVICE': shared_config['DEVICE'],
        'EPOCHS': shared_config['EPOCHS'],
        'SAVE_PATH': shared_config['RESERVOIR_SAVE_PATH'],
        
        # Calculated training parameters
        'steps_per_epoch': actual_steps_per_epoch,
        'total_steps': total_steps,
        'warmup_steps': warmup_steps,
        'accumulation_steps': accumulation_steps,  # Gradient accumulation
        
        # Optimized learning parameters
        'LR': 3e-4,           # Higher initial learning rate
        'weight_decay': 0.01,  # L2 regularization
        
        # Generation parameters
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.9,
    }
    
    print(f"Training configuration:")
    print(f"  Dataset size: {len(train_loader.dataset):,} sequences")
    print(f"  Steps per epoch: {steps_per_epoch:,}")
    print(f"  Total training steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Effective batch size: {config['BATCH_SIZE'] * config['accumulation_steps']}")
    
    return config


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Creates a cosine annealing scheduler with warmup based on actual training steps."""
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup
            return step / warmup_steps
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# Training function with all optimizations
def train_optimized_reservoir_model(model, train_loader, val_loader, config, tokenizer, shared_config):
    """Optimized training loop with mixed precision, gradient accumulation, and cosine scheduling."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['LR'], 
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)  # Better betas for language modeling
    )
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, 
        config['warmup_steps'], 
        config['total_steps']
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Gradient accumulation
    accumulation_steps = config['accumulation_steps']
    
    train_losses = []
    val_perplexities = []
    learning_rates = []
    steps = []
    
    total_batches = 0
    optimizer_steps = 0
    
    print("Starting Optimized Reservoir Model training...")
    print(f"Training for {config['EPOCHS']} epochs, {config['total_steps']} total steps")
    
    model.train()
    optimizer.zero_grad()
    
    for epoch in range(config['EPOCHS']):
        print(f"\n--- Epoch {epoch + 1}/{config['EPOCHS']} ---")
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        
        for x, y in pbar:
            total_batches += 1
            x, y = x.to(config['DEVICE']), y.to(config['DEVICE'])
            
            # Mixed precision forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(x)
                B, T, C = logits.shape
                loss = criterion(logits.view(B * T, C), y.view(B * T))
                loss = loss / accumulation_steps  # Scale loss for accumulation
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            if total_batches % accumulation_steps == 0:
                optimizer_steps += 1
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            train_losses.append(loss.item() * accumulation_steps)  # Unscale for logging
            current_lr = scheduler.get_last_lr()[0]
            learning_rates.append(current_lr)
            
            pbar.set_postfix({
                'loss': f"{loss.item() * accumulation_steps:.4f}",
                'lr': f"{current_lr:.2e}",
                'opt_step': f"{optimizer_steps}/{config['total_steps']}"
            })
            
            # Evaluation
            if optimizer_steps > 0 and optimizer_steps % (config['EVAL_INTERVAL'] // accumulation_steps) == 0:
                val_loss = eval_reservoir_model(model, val_loader, config)
                perplexity = np.exp(val_loss)
                val_perplexities.append(perplexity)
                steps.append(optimizer_steps)  # FIXED: Log optimizer steps instead of batch steps
                
                print(f"\nOptimizer Step {optimizer_steps}: Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")
                
                # Generate sample
                generated_text = generate_from_reservoir(
                    model, "Once upon a time", shared_config['MAX_OUT_TOKENS'], config, tokenizer
                )
                print("Generated:", generated_text[:150] + "...")
            
            # Stop if we've reached the total steps
            if optimizer_steps >= config['total_steps']:
                print(f"\nReached target optimizer steps ({config['total_steps']}). Stopping training.")
                break
        
        # Break from epoch loop if we've reached total steps
        if optimizer_steps >= config['total_steps']:
            break

    print(f"\nTraining completed! Total batches: {total_batches}, Optimizer steps: {optimizer_steps}")

    return {
        'train_losses': train_losses,
        'val_perplexities': val_perplexities,
        'learning_rates': learning_rates,
        'steps': steps
    }

optimized_reservoir_config = create_optimized_reservoir_config(SHARED_CONFIG, train_loader)

optimized_model = OptimizedDeepReservoirModel(
    vocab_size=tokenizer.vocab_size,
    config=optimized_reservoir_config
).to(optimized_reservoir_config['DEVICE'])

print(f"Optimized model: {sum(p.numel() for p in optimized_model.parameters() if p.requires_grad):,} trainable parameters")


# -----------------------------------------------------------------------------
# Train the Reservoir Model
# -----------------------------------------------------------------------------

# Train the model
training_history = train_optimized_reservoir_model(
    optimized_model, 
    train_loader, 
    val_loader, 
    optimized_reservoir_config,
    tokenizer,
    SHARED_CONFIG
)

# Plot results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(training_history['train_losses'])
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(training_history['steps'], training_history['val_perplexities'], marker='o')
plt.title('Validation Perplexity')
plt.xlabel('Steps')
plt.ylabel('Perplexity')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(training_history['learning_rates'])
plt.title('Learning Rate Schedule')
plt.xlabel('Batch')
plt.ylabel('Learning Rate')
plt.grid(True)

plt.tight_layout()
plt.show()

# Save the model
torch.save(optimized_model.state_dict(), optimized_reservoir_config['SAVE_PATH'])


print("=" * 80)
print("Model Comparison - Text Generation")
print("=" * 80)

prompt = "Once upon a time, there was a little"
print(f"Prompt: '{prompt}'\n")

# Generate from Reservoir model
reservoir_text = generate_from_reservoir(
    optimized_model, 
    prompt, 
    SHARED_CONFIG['MAX_OUT_TOKENS'], 
    optimized_reservoir_config,
    tokenizer
)
print("Reservoir Model Output:")
print("-" * 50)
print(reservoir_text)
print("\n")