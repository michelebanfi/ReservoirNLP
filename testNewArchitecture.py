import torch
import torch.nn as nn
import numpy as np
import math
from scipy.sparse import lil_matrix, eye, csr_matrix, tril
from scipy.sparse.linalg import eigs
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
import matplotlib.pyplot as plt
import os

# --- Step 1: Helper functions for the Reservoir Matrix (with Causal Mask) ---

def _base_pattern_by_rule(rule: str, base_dim: int | None = None, density: float = 0.5) -> np.ndarray:
    """Return a base binary pattern according to the requested rule.
    Rules:
      - 'cross8': 3x3 with hole center, 8 ones (original default)
      - 'ring4': 3x3 ring with 4 ones
      - 'checker': 3x3 checkerboard-like
      - 'diag': diagonals set to 1
      - 'sierpinski': 2x2 [[1,1],[1,0]]
      - 'random': random base_dim x base_dim with given density
    """
    rule = (rule or 'cross8').lower()
    if rule == 'sierpinski':
        return np.array([[1, 1], [1, 0]], dtype=int)
    if rule == 'cross8':
        return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    if rule == 'ring4':
        return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=int)
    if rule == 'checker':
        return np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=int)
    if rule == 'diag':
        n = base_dim or 3
        eye1 = np.eye(n, dtype=int)
        eye2 = np.fliplr(eye1)
        patt = np.clip(eye1 + eye2, 0, 1)
        # Optional: zero center for odd n
        if n % 2 == 1:
            patt[n//2, n//2] = 0
        return patt
    if rule == 'random':
        n = base_dim or 3
        rng = np.random.default_rng(42)
        patt = (rng.random((n, n)) < density).astype(int)
        # Ensure not all-zero
        if patt.sum() == 0:
            patt[rng.integers(0, n), rng.integers(0, n)] = 1
        # Encourage a hole in the center if odd
        if n % 2 == 1:
            patt[n//2, n//2] = 0
        return patt
    # Fallback
    return np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)


def generate_fractal_matrix(
    dim: int,
    base_pattern: np.ndarray | None = None,
    causal: bool = False,
    *,
    rule: str = 'cross8',
    base_dim: int | None = None,
    density: float = 0.5,
) -> csr_matrix:
    """Generates a sparse matrix with a recursive fractal pattern, optionally causal.
    You can either pass a concrete base_pattern, or select a `rule`.
    """
    if base_pattern is None:
        base_pattern = _base_pattern_by_rule(rule, base_dim=base_dim, density=density)

    base_dim_local = base_pattern.shape[0]
    if dim == 1:
        return eye(1, format='csr')

    power = math.ceil(math.log(dim, base_dim_local))
    actual_dim = base_dim_local ** power

    # Recursive build
    sub_dim = actual_dim // base_dim_local
    smaller_matrix = generate_fractal_matrix(
        sub_dim, base_pattern=base_pattern, causal=causal, rule=rule, base_dim=base_dim, density=density
    )

    new_matrix = lil_matrix((actual_dim, actual_dim))
    for i in range(base_dim_local):
        for j in range(base_dim_local):
            if base_pattern[i, j] == 1:
                start_row, end_row = i * sub_dim, (i + 1) * sub_dim
                start_col, end_col = j * sub_dim, (j + 1) * sub_dim
                new_matrix[start_row:end_row, start_col:end_col] = smaller_matrix

    # Apply causal mask if requested
    if causal:
        new_matrix = tril(new_matrix, k=-1)  # strictly lower triangular

    return new_matrix.tocsr()[:dim, :dim]

def normalize_spectral_radius(matrix, target_radius=0.99):
    """Normalizes the matrix to have a specific spectral radius."""
    try:
        eigenvals = eigs(matrix.asfptype(), k=1, which='LM', return_eigenvectors=False)
        spectral_radius = np.abs(eigenvals[0])
        if spectral_radius > 1e-6:
            matrix = matrix * (target_radius / spectral_radius)
        return matrix
    except:
        return matrix

def to_torch_sparse(scipy_matrix):
    """Converts a Scipy sparse matrix to a PyTorch sparse COO tensor."""
    coo = scipy_matrix.tocoo()
    values = torch.FloatTensor(coo.data)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))
    shape = torch.Size(coo.shape)
    return torch.sparse_coo_tensor(indices, values, shape)

# --- Step 2: Model Components (PositionalEncoding, ReservoirLayer) ---
# (These are mostly unchanged from the previous version)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class ReservoirLayer(nn.Module):
    def __init__(self, d_model, max_seq_len, activation=nn.GELU(), spectral_radius=1.1, dropout=0.1, num_heads=4,
                 head_rules: list[str] | None = None, use_gating=False, use_head_projs=False, use_hybrid_mode=False, num_attn_heads=4):
        super(ReservoirLayer, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Generate separate fractal matrices for each head (allow different rules per head)
        default_rules = ['cross8', 'ring4', 'checker', 'diag']
        if head_rules is None:
            head_rules = [default_rules[i % len(default_rules)] for i in range(num_heads)]
        self._head_rules = head_rules
        self.scipy_W_res_heads = [
            normalize_spectral_radius(
                generate_fractal_matrix(max_seq_len, causal=True, rule=head_rules[i % len(head_rules)]), spectral_radius
            )
            for i in range(num_heads)
        ]
        
        self.activation = activation
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feature 1: Gating mechanism
        self.use_gating = use_gating
        if self.use_gating:
            self.gate_proj = nn.Linear(d_model, d_model)
        
        # Feature 2: Learnable head projections
        self.use_head_projs = use_head_projs
        if self.use_head_projs:
            self.head_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.num_heads)])
        
        # Feature 3: Hybrid mode
        self.use_hybrid_mode = use_hybrid_mode
        if self.use_hybrid_mode:
            self.attention = nn.MultiheadAttention(d_model, num_attn_heads, dropout=dropout, batch_first=False)
            self.out_proj = nn.Linear(d_model * 2, d_model)
        else:
            self.out_proj = nn.Linear(d_model * num_heads, d_model)

    def forward(self, x):
        seq_len, batch_size, d_model = x.shape
        
        attn_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        
        if not self.use_hybrid_mode:
            # Process each head
            head_outputs = []
            for head in range(self.num_heads):
                sub_matrix_scipy = self.scipy_W_res_heads[head][:seq_len, :seq_len]
                w_res_sub = to_torch_sparse(sub_matrix_scipy).to(x.device)
                
                # Apply reservoir for this head
                res_outputs = []
                for i in range(batch_size):
                    head_input = x[:, i, :]
                    if self.use_head_projs:
                        head_input = self.head_projs[head](head_input)
                    res_outputs.append(torch.sparse.mm(w_res_sub, head_input))
                x_res_head = torch.stack(res_outputs).permute(1, 0, 2)
                x_res_head = self.activation(x_res_head)
                head_outputs.append(x_res_head)
            
            # Concatenate heads
            x_res = torch.cat(head_outputs, dim=-1)  # (seq_len, batch_size, d_model * num_heads)
            x_res = self.out_proj(x_res)  # Project back to d_model
        else:
            # Hybrid mode: average reservoir heads + attention
            accumulated_res = 0
            for head in range(self.num_heads):
                sub_matrix_scipy = self.scipy_W_res_heads[head][:seq_len, :seq_len]
                w_res_sub = to_torch_sparse(sub_matrix_scipy).to(x.device)
                
                res_outputs = []
                for i in range(batch_size):
                    head_input = x[:, i, :]
                    if self.use_head_projs:
                        head_input = self.head_projs[head](head_input)
                    res_outputs.append(torch.sparse.mm(w_res_sub, head_input))
                x_res_head = torch.stack(res_outputs).permute(1, 0, 2)
                accumulated_res += x_res_head
            x_res = self.activation(accumulated_res / self.num_heads)
            
            x_attn, _ = self.attention(x, x, x, attn_mask=attn_mask)
            combined_output = torch.cat([x_res, x_attn], dim=-1)
            x_res = self.out_proj(combined_output)
        
        # Gating mechanism
        if self.use_gating:
            g = torch.sigmoid(self.gate_proj(x))
            x = self.norm1(x * (1 - g) + x_res * g)
        else:
            x = self.norm1(x + x_res)
        
        x_mlp = self.mlp(x)
        x = self.norm2(x + x_mlp)
        return x

# --- Step 3: The Causal Reservoir Transformer Model ---

class ReservoirLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, max_seq_len, dropout=0.1, num_heads=4,
                 head_rules: list[str] | None = None, use_gating=False, use_head_projs=False, use_hybrid_mode=False, num_attn_heads=4):
        super(ReservoirLM, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # We make this non-trainable as per the original idea
        # self.embedding.weight.requires_grad = False  # Commented out to allow training
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList([
            ReservoirLayer(d_model, max_seq_len, dropout=dropout, num_heads=num_heads, head_rules=head_rules,
                           use_gating=use_gating, use_head_projs=use_head_projs, use_hybrid_mode=use_hybrid_mode, num_attn_heads=num_attn_heads)
            for _ in range(num_layers)
        ])
        # Additional trainable projection after reservoir layers
        self.additional_proj = nn.Linear(d_model, d_model)
        # NEW: The head now maps to the entire vocabulary
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, src):
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x)
        x = self.additional_proj(x)  # Additional trainable projection
        output = self.head(x)
        return output

# --- Step 4: Training Script with Hugging Face Datasets ---

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import DataLoader

# Hyperparameters
VOCAB_SIZE = 10000
D_MODEL = 512
NUM_LAYERS = 3
MAX_SEQ_LEN = 512
BATCH_SIZE = 8
EPOCHS = 10
LR = 0.001

# 1. Load Dataset from WikiText-103
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = dataset['train']
val_dataset = dataset['test']

# For tokenizer training, use a subset
training_texts = train_dataset[:10000]["text"]  # Use first 10000 for training tokenizer

# 2. Train a Tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]"])

def get_training_corpus():
    for text in training_texts:
        yield text

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
pad_token_id = tokenizer.token_to_id("[PAD]")

# 3. Prepare DataLoader
def collate_fn(batch):
    # Flatten, tokenize, and then chunk the text
    full_text = " ".join([item['text'] for item in batch if item['text']])
    if not full_text: return None, None
    
    token_ids = tokenizer.encode(full_text).ids
    chunks = [
        token_ids[i : i + MAX_SEQ_LEN + 1] 
        for i in range(0, len(token_ids), MAX_SEQ_LEN + 1)
        if len(token_ids[i : i + MAX_SEQ_LEN + 1]) == MAX_SEQ_LEN + 1
    ]
    
    if not chunks: return None, None
    
    inputs = torch.tensor([chunk[:-1] for chunk in chunks], dtype=torch.long)
    targets = torch.tensor([chunk[1:] for chunk in chunks], dtype=torch.long)
    
    # We need (seq_len, batch_size)
    return inputs.T, targets.T

train_loader = DataLoader([{'text': t} for t in train_dataset['text'][:50000]], batch_size=BATCH_SIZE*5, collate_fn=collate_fn)  # Use first 50000 samples for training
val_loader = DataLoader([{'text': t} for t in val_dataset['text'][:2000]], batch_size=BATCH_SIZE*5, collate_fn=collate_fn)  # Small val set

# 4. Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ReservoirLM(VOCAB_SIZE, D_MODEL, NUM_LAYERS, MAX_SEQ_LEN,
                    use_gating=True, use_head_projs=True, use_hybrid_mode=True, num_attn_heads=4
                    ).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
warmup_epochs = int(0.1 * EPOCHS)  # 10% warmup
scheduler = SequentialLR(optimizer, [
    LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs),
    CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs)
], milestones=[warmup_epochs])
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

# Visualize reservoir matrix
def plot_reservoir_matrix(matrix, title="Reservoir Matrix"):
    plt.figure(figsize=(8, 8))
    plt.imshow(matrix.toarray(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title(title)
    plt.savefig(f'output/{title.replace(" ", "_")}.png')
    plt.close()
    print(f"Reservoir plot saved to output/{title.replace(' ', '_')}.png")

# Ensure output dir exists before plotting
os.makedirs('output', exist_ok=True)

# Plot reservoir matrices from the first layer (all heads if available)
if hasattr(model.layers[0], 'scipy_W_res_heads'):
    first_layer = model.layers[0]
    for h, mat in enumerate(first_layer.scipy_W_res_heads):
        rule = getattr(first_layer, '_head_rules', None)
        title = f"Reservoir Matrix Head {h}"
        if rule and h < len(rule):
            title += f" ({rule[h]})"
        plot_reservoir_matrix(mat, title)
else:
    # Fallback if single head
    plot_reservoir_matrix(model.layers[0].scipy_W_res, "Reservoir Matrix")

print(f"Starting training on {device}...")
if torch.cuda.is_available():
    print(f"Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

train_losses = []
learning_rates = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        if inputs is None: continue
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs) # Shape: (seq_len, batch_size, vocab_size)
        
        # Reshape for loss calculation
        loss = criterion(outputs.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step()
    epoch_loss = total_loss / len(train_loader)
    train_losses.append(epoch_loss)
    learning_rates.append(scheduler.get_last_lr()[0])
    if torch.cuda.is_available():
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# Compute validation perplexity
model.eval()
total_val_loss = 0
with torch.no_grad():
    for inputs, targets in val_loader:
        if inputs is None: continue
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        total_val_loss += loss.item()
val_perplexity = math.exp(total_val_loss / len(val_loader))
print(f"Validation Perplexity: {val_perplexity:.4f}")
if torch.cuda.is_available():
    print(f"Final GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

# Plotting
# already ensured earlier

epochs = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, learning_rates, label='Learning Rate')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()

plt.tight_layout()
plt.savefig('output/training_plots.png')
print("Plots saved to output/training_plots.png")
val_loader = DataLoader([{'text': t} for t in val_dataset['text']], batch_size=BATCH_SIZE*5, collate_fn=collate_fn)

# 5. Test Generation
def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=0.8):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    for _ in range(max_new_tokens):
        input_tokens = tokens[-MAX_SEQ_LEN:]  # Take last MAX_SEQ_LEN tokens
        input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(1).to(device)  # (seq_len, 1)
        
        with torch.no_grad():
            output = model(input_tensor)  # (seq_len, 1, vocab_size)
        
        next_token_logits = output[-1, 0, :]  # Logits for the last token
        next_token_logits = next_token_logits / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        
        tokens.append(next_token)
        
        # Stop if we generate PAD or EOS if defined
        if next_token == pad_token_id:
            break
    
    generated_text = tokenizer.decode(tokens)
    return generated_text

# Test the model
prompt = "Alice was"
generated = generate_text(model, tokenizer, prompt, max_new_tokens=100)
print(f"\nPrompt: {prompt}")
print(f"Generated: {generated}")