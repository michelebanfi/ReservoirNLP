import os
import torch

# -----------------------------------------------------------------------------
# Shared Configuration for Both Models
# -----------------------------------------------------------------------------

SHARED_CONFIG = {
    # Data parameters
    'MAX_STORIES': 10000,                # Number of stories to use for training
    
    # Training parameters
    'EPOCHS': 3,                       # Number of training epochs
    'BATCH_SIZE': 64,                  # Batch size for training
    'BLOCK_SIZE': 128,                 # Context size for training
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Evaluation parameters
    'EVAL_INTERVAL': 2000,              # Steps between evaluations
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

class OptimizedTinyStoriesDataset(data.Dataset):
    def __init__(self, tokenized_path, block_size: int):
        self.block_size = block_size
        # OPTIMIZATION: Load entire dataset into memory if it fits (faster than mmap for small datasets)
        data_size = os.path.getsize(tokenized_path)
        if data_size < 1e9:  # Less than 1GB, load into RAM
            self.data = np.load(tokenized_path)
            self.use_mmap = False
        else:
            self.data = np.load(tokenized_path, mmap_mode='r')
            self.use_mmap = True
        
        print(f"Dataset loaded {'in memory' if not self.use_mmap else 'with mmap'}: {len(self.data):,} tokens")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        if self.use_mmap:
            chunk = chunk.copy()  # Copy from mmap to avoid issues
        
        source = torch.from_numpy(chunk[:-1].astype(np.int64))
        target = torch.from_numpy(chunk[1:].astype(np.int64))
        return source, target
    
# -----------------------------------------------------------------------------
# Download and Prepare Dataset
# -----------------------------------------------------------------------------

def prepare_data_efficiently(shared_config):
    """More efficient data preparation."""
    
    # Use more workers for data loading
    num_workers = min(4, os.cpu_count())
    
    train_txt_path, val_txt_path = download_and_save_dataset(max_stories=shared_config["MAX_STORIES"])
    
    train_tokenized_path = train_txt_path.replace('.txt', '.npy')
    if not os.path.exists(train_tokenized_path):
        pre_tokenize_dataset(train_txt_path, train_tokenized_path)
    
    val_tokenized_path = val_txt_path.replace('.txt', '.npy')
    if not os.path.exists(val_tokenized_path):
        pre_tokenize_dataset(val_txt_path, val_tokenized_path)
    
    # Use optimized dataset
    train_dataset = OptimizedTinyStoriesDataset(train_tokenized_path, shared_config['BLOCK_SIZE'])
    train_loader = data.DataLoader(
        train_dataset, 
        batch_size=shared_config['BATCH_SIZE'], 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_dataset = OptimizedTinyStoriesDataset(val_tokenized_path, shared_config['BLOCK_SIZE'])
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=shared_config['BATCH_SIZE'],
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader

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
# train_dataset = TinyStoriesDataset(train_tokenized_path, SHARED_CONFIG['BLOCK_SIZE'])
# train_loader = data.DataLoader(train_dataset, batch_size=SHARED_CONFIG['BATCH_SIZE'], shuffle=True)

# val_dataset = TinyStoriesDataset(val_tokenized_path, SHARED_CONFIG['BLOCK_SIZE'])
# val_loader = data.DataLoader(val_dataset, batch_size=SHARED_CONFIG['BATCH_SIZE'])
train_loader, val_loader = prepare_data_efficiently(SHARED_CONFIG)

print(f"Data preparation complete! Tokenizer vocabulary size: {tokenizer.vocab_size}")


# REPLACE current OptimizedParallelReservoir with:
class TrueReservoir(nn.Module):
    def __init__(self, input_size, hidden_size, spectral_radius=1.2, 
                 sparsity=0.1, leak_rate=0.95, device='cuda'):
        super().__init__()
        self.hidden_size = hidden_size
        self.leak_rate = leak_rate
        self.device = device
        
        # Fixed reservoir weights (non-trainable)
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_res = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Freeze reservoir weights
        for param in self.W_in.parameters():
            param.requires_grad = False
        for param in self.W_res.parameters():
            param.requires_grad = False
            
        self._initialize_weights(spectral_radius, sparsity)
        self.activation = torch.tanh
        self.register_buffer('initial_state', torch.zeros(1, hidden_size))
    
    def _initialize_weights(self, spectral_radius, sparsity):
        import math
        # Input weights - random uniform
        nn.init.kaiming_uniform_(self.W_in.weight, a=math.sqrt(5))
        
        # Reservoir weights
        with torch.no_grad():
            W = self.W_res.weight.data
            mask = (torch.rand(W.shape, device=self.device) > sparsity).float()
            W.normal_(0, 1)
            W *= mask
            
            # Scale to desired spectral radius
            eigenvalues = torch.linalg.eigvals(W)
            current_radius = torch.max(torch.abs(eigenvalues))
            W *= spectral_radius / (current_radius + 1e-8)
    
    def forward(self, x, initial_state=None):
        batch_size, seq_len, _ = x.shape
        states = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.device)
        prev_state = initial_state if initial_state is not None else self.initial_state.expand(batch_size, self.hidden_size)
        
        for t in range(seq_len):
            input_t = x[:, t, :]
            input_part = self.W_in(input_t)
            reservoir_part = self.W_res(prev_state)
            new_state = self.activation(input_part + reservoir_part)
            
            # Leaky integration
            if self.leak_rate < 1.0:
                new_state = self.leak_rate * new_state + (1.0 - self.leak_rate) * prev_state
                
            states[:, t, :] = new_state
            prev_state = new_state
            
        return states


class NVARReservoir(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.delay = config.get('nvar_delay', 5)
        self.degree = config.get('nvar_degree', 2)
        self.input_size = config['embedding_dim']
        
        # Calculate feature dimension after expansion
        linear_terms = self.delay * self.input_size
        quadratic_terms = (linear_terms * (linear_terms + 1)) // 2
        self.feature_dim = 1 + linear_terms + quadratic_terms  # constant + linear + quadratic
        
        # Trainable readout (only this part is trained)
        self.readout = nn.Sequential(
            nn.Linear(self.feature_dim, config['embedding_dim']),
            nn.GELU(),
            nn.Linear(config['embedding_dim'], config['embedding_dim'])
        )
        
        # Register buffer for delayed indices
        self.register_buffer('delay_indices', torch.arange(self.delay-1, -1, -1))
    
    def polynomial_expansion(self, x):
        """Creates polynomial features from delayed embeddings"""
        batch_size, seq_len, embed_dim = x.shape
        
        # Create delayed embeddings
        x_padded = F.pad(x, (0, 0, self.delay-1, 0))
        delayed = x_padded.unfold(1, self.delay, 1)  # [B, T, delay, embed_dim]
        
        # Reshape to [B*T, delay*embed_dim]
        features = delayed.reshape(-1, self.delay * embed_dim)
        
        # Create polynomial features (constant + linear + quadratic)
        constant = torch.ones(features.size(0), 1, device=features.device)
        linear = features
        
        # Quadratic terms (only upper triangle to avoid duplicates)
        quadratic_list = []
        for i in range(features.size(1)):
            for j in range(i, features.size(1)):
                quadratic_list.append(features[:, i] * features[:, j])
        quadratic = torch.stack(quadratic_list, dim=1)
        
        # Combine all features
        all_features = torch.cat([constant, linear, quadratic], dim=1)
        return all_features.reshape(batch_size, seq_len, -1)
    
    def forward(self, x):
        # Create polynomial features
        features = self.polynomial_expansion(x)
        # Apply readout
        return self.readout(features)


class HybridReservoirBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # True reservoir component
        self.reservoir = TrueReservoir(
            input_size=config['embedding_dim'],
            hidden_size=config['reservoir_size'],
            spectral_radius=config['spectral_radius'],
            sparsity=config.get('sparsity', 0.1),
            leak_rate=config.get('leak_rate', 0.95),
            device=config['DEVICE']
        )
        
        # SSM-inspired selective mechanism
        self.delta_proj = nn.Sequential(
            nn.Linear(config['embedding_dim'], 1),
            nn.Softplus()
        )
        
        # State transition parameters (frozen like reservoir)
        self.A = nn.Parameter(-torch.ones(config['reservoir_size']) * 10, requires_grad=False)
        self.B = nn.Linear(config['embedding_dim'], config['reservoir_size'], bias=False)
        for p in self.B.parameters():
            p.requires_grad = False
            
        # Readout layer
        self.readout = nn.Sequential(
            nn.LayerNorm(config['reservoir_size']),
            nn.Linear(config['reservoir_size'], config['embedding_dim']),
            nn.GELU()
        )
        
        # Hybrid gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(config['embedding_dim'], config['embedding_dim']),
            nn.Sigmoid()
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(config['embedding_dim'])

    def forward(self, x, initial_state=None):
        x_norm = self.norm(x)
        
        # Reservoir path
        reservoir_states = self.reservoir(x_norm, initial_state)
        reservoir_out = self.readout(reservoir_states)
        
        # SSM path with selective updates
        B = self.B(x_norm)
        delta = self.delta_proj(x_norm).squeeze(-1)  # [B, L]
        
        # Discretize A
        A_bar = torch.exp(self.A * delta.unsqueeze(-1))  # [B, L, state_size]
        B_bar = B * (1 - A_bar) / (self.A + 1e-8)  # [B, L, state_size]
        
        # State update
        h0 = initial_state if initial_state is not None else torch.zeros_like(B[:, 0, :])
        ssm_states = [h0]
        for i in range(x_norm.size(1)):
            h = A_bar[:, i, :] * ssm_states[-1] + B_bar[:, i, :]
            ssm_states.append(h)
        ssm_states = torch.stack(ssm_states[1:], dim=1)
        
        # Combine paths
        gate = self.gate(x)
        return reservoir_out * gate + ssm_states * (1 - gate)


class EnhancedDeepReservoirModel(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(config['max_seq_len'], config['embedding_dim']) * 0.02
        )
        
        # Reservoir blocks - use hybrid implementation
        self.blocks = nn.ModuleList([
            HybridReservoirBlock({
                **config,
                'reservoir_size': config['reservoir_size'] // (i + 1)
            }) for i in range(config['num_blocks'])
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(config['embedding_dim'])
        self.lm_head = nn.Linear(config['embedding_dim'], vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Reservoir warmup parameters
        self.warmup_steps = config.get('reservoir_warmup', 5)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, initial_states=None):
        batch_size, seq_len = idx.shape
        
        # Token embeddings
        tok_emb = self.embedding(idx)
        
        # Positional embeddings
        pos_emb = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combined embeddings
        x = tok_emb + pos_emb
        
        # Process through reservoir blocks with state management
        current_states = initial_states
        for i, block in enumerate(self.blocks):
            # Apply warmup for the first few tokens
            if self.training and seq_len > self.warmup_steps:
                # Process warmup tokens first
                warmup_x = x[:, :self.warmup_steps]
                warmup_states = block.reservoir.initial_state.expand(batch_size, -1)
                for _ in range(self.warmup_steps):
                    warmup_states = block(warmup_x[:, _:_+1], warmup_states)
                
                # Now process the main sequence with warmed-up state
                x = block(x[:, self.warmup_steps:], warmup_states)
            else:
                x = block(x, current_states[i] if current_states else None)
                if current_states is None:
                    current_states = [torch.zeros(batch_size, block.reservoir.hidden_size, 
                                                device=idx.device) for _ in self.blocks]
                current_states[i] = x[:, -1, :]  # Store final state
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits, current_states


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
    """Evaluates the reservoir model with state management."""
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    num_batches = 0
    
    for x, y in data_loader:
        if num_batches >= config['EVAL_ITER']:
            break
            
        x, y = x.to(config['DEVICE']), y.to(config['DEVICE'])
        
        # Reset states for each batch
        initial_states = None
        if hasattr(model, 'blocks') and len(model.blocks) > 0 and hasattr(model.blocks[0], 'reservoir'):
            # EnhancedDeepReservoirModel
            logits, _ = model(x, initial_states)
        else:
            # OptimizedDeepReservoirModel (original)
            logits = model(x)
        
        B, T, C = logits.shape
        loss = criterion(logits.view(B * T, C), y.view(B * T))
        total_loss += loss.item()
        num_batches += 1
        
    model.train()
    return total_loss / num_batches if num_batches > 0 else float('inf')


@torch.no_grad()
def generate_from_reservoir(model, context_str, max_new_tokens, config, tokenizer):
    """Generates text with proper reservoir state management."""
    model.eval()
    start_indices = tokenizer.encode(context_str)
    context = torch.tensor(start_indices, dtype=torch.long, device=config['DEVICE']).unsqueeze(0)
    
    # Process initial context to warm up reservoir states
    initial_context = context if context.size(1) <= config['BLOCK_SIZE'] else context[:, -config['BLOCK_SIZE']:]
    
    # Check if this is the enhanced model with state management
    if hasattr(model, 'blocks') and len(model.blocks) > 0 and hasattr(model.blocks[0], 'reservoir'):
        # EnhancedDeepReservoirModel
        _, states = model(initial_context)
        
        # Generate new tokens
        for _ in range(max_new_tokens):
            current_token = context[:, -1:]
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, states = model(current_token, states)
            
            logits = logits[:, -1, :] / config.get('temperature', 0.8)
            
            # Apply top-k and top-p sampling
            if config.get('top_p', 0.9) > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > config['top_p']
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove].unsqueeze(0)
                logits.scatter_(1, indices_to_remove, float('-inf'))
                
            if config.get('top_k', 50) > 0:
                top_k_logits, top_k_indices = torch.topk(logits, config['top_k'])
                filtered_logits = torch.full_like(logits, float('-inf'))
                filtered_logits.scatter_(1, top_k_indices, top_k_logits)
                logits = filtered_logits
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)
    else:
        # Original OptimizedDeepReservoirModel
        for _ in range(max_new_tokens):
            current_context = context if context.size(1) <= config['BLOCK_SIZE'] else context[:, -config['BLOCK_SIZE']:]
            
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(current_context)
            
            logits = logits[:, -1, :] / config.get('temperature', 0.8)
            
            if config.get('top_p', 0.9) > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > config['top_p']
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove].unsqueeze(0)
                logits.scatter_(1, indices_to_remove, float('-inf'))

            if config.get('top_k', 50) > 0:
                top_k_logits, top_k_indices = torch.topk(logits, config['top_k'])
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
    
    steps_per_epoch = len(train_loader)
    accumulation_steps = 8  # INCREASE accumulation to reduce optimizer calls
    actual_steps_per_epoch = steps_per_epoch // accumulation_steps
    total_steps = actual_steps_per_epoch * shared_config['EPOCHS']
    warmup_steps = min(500, total_steps // 20)  # Shorter warmup
    
    config = {
        # OPTIMIZATION: Smaller, faster architecture
        'embedding_dim': 256,  # Reduced from 384
        'num_blocks': 2,       # Reduced from 3
        'max_seq_len': shared_config['BLOCK_SIZE'] * 2,
        'tie_weights': True,
        'dropout': 0.05,       # Reduced dropout
        
        # OPTIMIZATION: Simpler reservoir configuration
        'reservoirs_per_block': [
            {'reservoir_size': 128, 'window_size': 16, 'spectral_radius': 1.0, 
             'sparsity': 0.3, 'activation': 'relu', 'leak_rate': 1.0},  # No leakage for speed
            {'reservoir_size': 192, 'window_size': 32, 'spectral_radius': 1.0, 
             'sparsity': 0.25, 'activation': 'relu', 'leak_rate': 1.0},
        ],
        'readout_hidden_size': 128,  # Reduced
        
        # Training optimizations
        'BATCH_SIZE': shared_config['BATCH_SIZE'],
        'BLOCK_SIZE': shared_config['BLOCK_SIZE'],
        'EVAL_INTERVAL': shared_config['EVAL_INTERVAL'] * 2,  # Less frequent evaluation
        'EVAL_ITER': 25,  # Fewer evaluation batches
        'DEVICE': shared_config['DEVICE'],
        'EPOCHS': shared_config['EPOCHS'],
        'SAVE_PATH': shared_config['RESERVOIR_SAVE_PATH'],
        
        'steps_per_epoch': actual_steps_per_epoch,
        'total_steps': total_steps,
        'warmup_steps': warmup_steps,
        'accumulation_steps': accumulation_steps,
        
        # OPTIMIZATION: More aggressive learning parameters
        'LR': 5e-4,
        'weight_decay': 0.01,
        
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


def create_enhanced_reservoir_config(shared_config, train_loader):
    """Create enhanced reservoir config with NVAR and SSM elements."""
    steps_per_epoch = len(train_loader)
    accumulation_steps = 8
    actual_steps_per_epoch = steps_per_epoch // accumulation_steps
    total_steps = actual_steps_per_epoch * shared_config['EPOCHS']
    warmup_steps = min(500, total_steps // 20)
    
    config = {
        # Architecture parameters
        'embedding_dim': 256,
        'num_blocks': 2,
        'max_seq_len': shared_config['BLOCK_SIZE'] * 2,
        'tie_weights': True,
        
        # Reservoir parameters
        'reservoir_size': 384,  # Larger reservoir for better capacity
        'spectral_radius': 1.0,
        'sparsity': 0.2,
        'leak_rate': 0.95,
        'reservoir_warmup': 5,  # Warmup steps for reservoir stability
        
        # NVAR parameters (if using pure NVAR path)
        'nvar_delay': 5,
        'nvar_degree': 2,
        
        # Training optimizations
        'BATCH_SIZE': shared_config['BATCH_SIZE'],
        'BLOCK_SIZE': shared_config['BLOCK_SIZE'],
        'EVAL_INTERVAL': shared_config['EVAL_INTERVAL'] * 2,
        'EVAL_ITER': 25,
        'DEVICE': shared_config['DEVICE'],
        'EPOCHS': shared_config['EPOCHS'],
        'SAVE_PATH': shared_config['RESERVOIR_SAVE_PATH'],
        'steps_per_epoch': actual_steps_per_epoch,
        'total_steps': total_steps,
        'warmup_steps': warmup_steps,
        'accumulation_steps': accumulation_steps,
        
        # Learning parameters
        'LR': 5e-4,
        'weight_decay': 0.01,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.9,
    }
    
    return config


@torch.no_grad()
def evaluate_reservoir_quality(model, data_loader, config, num_samples=10):
    """Evaluates key reservoir properties: memory capacity and state separation."""
    model.eval()
    memory_capacities = []
    state_separation = []
    
    # Sample a few sequences for evaluation
    samples = []
    for x, _ in data_loader:
        if len(samples) >= num_samples:
            break
        samples.append(x.to(config['DEVICE']))
    
    # Check if this is the enhanced model
    if not (hasattr(model, 'blocks') and len(model.blocks) > 0 and hasattr(model.blocks[0], 'reservoir')):
        model.train()
        return {'memory_capacity': 0, 'state_separation': 0}
    
    # Evaluate memory capacity
    for x in samples:
        _, states = model(x)
        
        # Memory capacity test - predict past inputs from current state
        for block_idx, block_states in enumerate(states):
            mc = 0
            for k in range(1, min(10, x.size(1))):
                # Predict x(t-k) from state at time t
                pred = block_states[:, k:]  # States at time t
                target = x[:, :-k]  # Inputs at time t-k
                
                # Simple linear readout for memory capacity test
                with torch.no_grad():
                    # Solve least squares problem: W * states = inputs
                    pred_flat = pred.reshape(-1, pred.size(-1))
                    target_flat = target.reshape(-1, target.size(-1))
                    if pred_flat.size(0) > 0 and target_flat.size(0) > 0:
                        try:
                            W = torch.linalg.lstsq(pred_flat, target_flat, driver='gels')[0]
                            pred_target = pred_flat @ W
                            
                            # Calculate memory capacity
                            mc_k = 1 - (torch.norm(pred_target - target_flat) ** 2) / (torch.norm(target_flat) ** 2 + 1e-8)
                            mc += mc_k.item()
                        except:
                            pass
            
            memory_capacities.append(mc / min(10, x.size(1)))
    
    # Evaluate state separation
    for i in range(len(samples)):
        for j in range(i+1, len(samples)):
            x1 = samples[i]
            x2 = samples[j]
            
            _, states1 = model(x1)
            _, states2 = model(x2)
            
            # Compare states for different inputs
            for block_states1, block_states2 in zip(states1, states2):
                # Distance between states for different inputs
                dist_diff = torch.norm(block_states1 - block_states2, dim=-1).mean().item()
                
                # Distance between states for same input (should be small)
                x1_copy = x1.clone()
                _, states1_copy = model(x1_copy)
                dist_same = torch.norm(block_states1 - states1_copy[0], dim=-1).mean().item()
                
                # State separation ratio
                separation = dist_diff / (dist_same + 1e-8)
                state_separation.append(separation)
    
    model.train()
    return {
        'memory_capacity': np.mean(memory_capacities) if memory_capacities else 0,
        'state_separation': np.mean(state_separation) if state_separation else 0
    }


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
    
    # OPTIMIZATION: Use fused AdamW if available
    try:
        from torch.optim import AdamW
        optimizer = AdamW(model.parameters(), lr=config['LR'], weight_decay=config['weight_decay'],
                         betas=(0.9, 0.95), fused=True)  # Fused optimizer
    except:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['LR'], weight_decay=config['weight_decay'],
                                     betas=(0.9, 0.95))
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, config['warmup_steps'], config['total_steps'])
    scaler = torch.cuda.amp.GradScaler()
    
    # OPTIMIZATION: Pre-allocate lists with expected size
    expected_samples = config['total_steps'] * config['accumulation_steps']
    train_losses = []
    val_perplexities = []
    learning_rates = []
    steps = []
    
    total_batches = 0
    optimizer_steps = 0
    accumulation_steps = config['accumulation_steps']
    
    print("Starting Fast Reservoir Model training...")
    
    model.train()
    optimizer.zero_grad()
    
    # OPTIMIZATION: Reduce Python overhead in training loop
    device = config['DEVICE']
    eval_interval = config['EVAL_INTERVAL'] // accumulation_steps
    
    for epoch in range(config['EPOCHS']):
        print(f"\n--- Epoch {epoch + 1}/{config['EPOCHS']} ---")
        
        # OPTIMIZATION: Remove tqdm if it's causing overhead, or reduce update frequency
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", 
                   mininterval=1.0)  # Update less frequently
        
        for x, y in pbar:
            total_batches += 1
            
            # OPTIMIZATION: Move to device without non_blocking (can cause issues)
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                logits = model(x)
                B, T, C = logits.shape
                loss = criterion(logits.view(B * T, C), y.view(B * T))
                loss = loss / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if total_batches % accumulation_steps == 0:
                optimizer_steps += 1
                
                # OPTIMIZATION: Skip gradient clipping if not necessary
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                # OPTIMIZATION: Update progress less frequently
                if optimizer_steps % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f"{loss.item() * accumulation_steps:.4f}",
                        'lr': f"{current_lr:.2e}",
                        'step': f"{optimizer_steps}/{config['total_steps']}"
                    })
            
            # Store metrics less frequently
            if total_batches % accumulation_steps == 0:
                train_losses.append(loss.item() * accumulation_steps)
                learning_rates.append(scheduler.get_last_lr()[0])
            
            # Less frequent evaluation
            if optimizer_steps > 0 and optimizer_steps % eval_interval == 0:
                val_loss = eval_reservoir_model(model, val_loader, config)
                perplexity = np.exp(val_loss)
                val_perplexities.append(perplexity)
                steps.append(optimizer_steps)
                
                print(f"\nStep {optimizer_steps}: Val Loss: {val_loss:.4f}, Perplexity: {perplexity:.4f}")
                
                # OPTIMIZATION: Generate shorter samples during training
                generated_text = generate_from_reservoir(
                    model, "Once upon a time", 50, config, tokenizer  # Shorter generation
                )
                print("Generated:", generated_text[:100] + "...")
            
            if optimizer_steps >= config['total_steps']:
                print(f"\nReached target steps. Stopping.")
                break
        
        if optimizer_steps >= config['total_steps']:
            break
    
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

# Create enhanced configuration
enhanced_config = create_enhanced_reservoir_config(SHARED_CONFIG, train_loader)

# Initialize enhanced model
enhanced_model = EnhancedDeepReservoirModel(
    vocab_size=tokenizer.vocab_size,
    config=enhanced_config
).to(enhanced_config['DEVICE'])

print(f"Enhanced model: {sum(p.numel() for p in enhanced_model.parameters() if p.requires_grad):,} trainable parameters")


# -----------------------------------------------------------------------------
# Train the Enhanced Reservoir Model  
# -----------------------------------------------------------------------------

# Train the enhanced model
training_history = train_optimized_reservoir_model(
    enhanced_model, 
    train_loader, 
    val_loader, 
    enhanced_config,
    tokenizer,
    SHARED_CONFIG
)

# Evaluate reservoir quality periodically
if enhanced_config.get('EVAL_INTERVAL', 0) > 0:
    reservoir_metrics = evaluate_reservoir_quality(enhanced_model, val_loader, enhanced_config)
    print(f"Reservoir Quality - Memory Capacity: {reservoir_metrics['memory_capacity']:.4f}, "
          f"State Separation: {reservoir_metrics['state_separation']:.4f}")

# Update visualization to include reservoir metrics
plt.figure(figsize=(20, 5))
plt.subplot(1, 4, 1)
plt.plot(training_history['train_losses'])
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.grid(True)

plt.subplot(1, 4, 2)
plt.plot(training_history['steps'], training_history['val_perplexities'], marker='o')
plt.title('Validation Perplexity')
plt.xlabel('Steps')
plt.ylabel('Perplexity')
plt.grid(True)

plt.subplot(1, 4, 3)
plt.plot(training_history['learning_rates'])
plt.title('Learning Rate Schedule')
plt.xlabel('Batch')
plt.ylabel('Learning Rate')
plt.grid(True)

# Add reservoir quality metrics plot
if 'reservoir_metrics' in locals():
    plt.subplot(1, 4, 4)
    plt.bar(['Memory Capacity', 'State Separation'], 
            [reservoir_metrics['memory_capacity'], reservoir_metrics['state_separation']])
    plt.title('Reservoir Quality Metrics')
    plt.ylabel('Score')
    plt.grid(True)

plt.tight_layout()
plt.show()

# Save the enhanced model
torch.save(enhanced_model.state_dict(), enhanced_config['SAVE_PATH'])


print("=" * 80)
print("Model Comparison - Text Generation")
print("=" * 80)

prompt = "Once upon a time, there was a little"
print(f"Prompt: '{prompt}'\n")

# Generate from Enhanced Reservoir model
enhanced_text = generate_from_reservoir(
    enhanced_model, 
    prompt, 
    SHARED_CONFIG['MAX_OUT_TOKENS'], 
    enhanced_config,
    tokenizer
)
print("Enhanced Reservoir Model Output:")
print("-" * 50)
print(enhanced_text)
print("\n")
