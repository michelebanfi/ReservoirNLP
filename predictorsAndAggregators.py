import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

# --- Configuration ---
# Set to True for quick local testing, False for full training
LOCAL_TESTING = False

@dataclass
class TrainConfig:
    VOCAB_SIZE: int
    MAX_SEQ_LEN: int
    D_MODEL: int
    NUM_LAYERS: int
    NUM_HEADS: int
    BATCH_SIZE: int
    EPOCHS: int
    LR: float
    WEIGHT_DECAY: float
    NUM_PREDICTORS: int
    GUMBEL_TAU_START: float
    GUMBEL_TAU_END: float
    WARMUP_EPOCHS: int
    DIVERSITY_LOSS_WEIGHT: float
    CONSISTENCY_LOSS_WEIGHT: float

# Local (fast) config
LOCAL_CONFIG = TrainConfig(
    VOCAB_SIZE=1000,
    MAX_SEQ_LEN=32,
    D_MODEL=128,
    NUM_LAYERS=2,
    NUM_HEADS=2,
    BATCH_SIZE=8,
    EPOCHS=1,
    LR=1e-3,
    WEIGHT_DECAY=1e-4,
    NUM_PREDICTORS=2,
    GUMBEL_TAU_START=2.0,
    GUMBEL_TAU_END=0.5,
    WARMUP_EPOCHS=1,
    DIVERSITY_LOSS_WEIGHT=0.1,
    CONSISTENCY_LOSS_WEIGHT=0.05,
)

# Full (heavier) config – adjust as needed for your machine
FULL_CONFIG = TrainConfig(
    VOCAB_SIZE=16000,
    MAX_SEQ_LEN=128,
    D_MODEL=256,
    NUM_LAYERS=2,
    NUM_HEADS=4,
    BATCH_SIZE=16,
    EPOCHS=10,
    LR=5e-4,
    WEIGHT_DECAY=0.01,
    NUM_PREDICTORS=3,
    GUMBEL_TAU_START=1.5,
    GUMBEL_TAU_END=0.5,
    WARMUP_EPOCHS=2,
    DIVERSITY_LOSS_WEIGHT=0.05,
    CONSISTENCY_LOSS_WEIGHT=0.05,
)

# Select config based on LOCAL_TESTING
_CFG = LOCAL_CONFIG if LOCAL_TESTING else FULL_CONFIG

# Expose config values as module-level constants (rest of code stays the same)
VOCAB_SIZE = _CFG.VOCAB_SIZE
MAX_SEQ_LEN = _CFG.MAX_SEQ_LEN
D_MODEL = _CFG.D_MODEL
NUM_LAYERS = _CFG.NUM_LAYERS
NUM_HEADS = _CFG.NUM_HEADS
BATCH_SIZE = _CFG.BATCH_SIZE
EPOCHS = _CFG.EPOCHS
LR = _CFG.LR
WEIGHT_DECAY = _CFG.WEIGHT_DECAY
NUM_PREDICTORS = _CFG.NUM_PREDICTORS
GUMBEL_TAU_START = _CFG.GUMBEL_TAU_START
GUMBEL_TAU_END = _CFG.GUMBEL_TAU_END
WARMUP_EPOCHS = _CFG.WARMUP_EPOCHS
DIVERSITY_LOSS_WEIGHT = _CFG.DIVERSITY_LOSS_WEIGHT
CONSISTENCY_LOSS_WEIGHT = _CFG.CONSISTENCY_LOSS_WEIGHT

# --- More Diverse Transformer Architectures ---
class DiverseTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len, variant='standard'):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        if variant == 'deep_narrow':
            # More layers, fewer heads
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=max(1, num_heads//2), 
                                                     batch_first=True, dropout=0.15)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers+2)
        elif variant == 'wide_shallow':
            # Fewer layers, more heads
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads*2, 
                                                     batch_first=True, dropout=0.05)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, num_layers-1))
        elif variant == 'regularized':
            # Standard but with heavy regularization
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, 
                                                     batch_first=True, dropout=0.25)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        else:  # standard
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(d_model, vocab_size)
        self.variant = variant

    def forward(self, src):
        seq_len = src.shape[1]
        x = self.embedding(src) + self.pos_encoder[:, :seq_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(src.device)
        output = self.transformer_encoder(x, mask=mask)
        return self.head(output)

# --- The Unified End-to-End Model ---
class PredictorAggregatorLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len, num_predictors):
        super().__init__()
        self.num_predictors = num_predictors
        self.vocab_size = vocab_size
        
        # 1. Create diverse Predictor models with different architectures
        variants = ['standard', 'deep_narrow', 'wide_shallow', 'regularized']
        self.predictors = nn.ModuleList([
            DiverseTransformerLM(vocab_size, d_model, num_heads, num_layers, max_len, 
                               variant=variants[i % len(variants)])
            for i in range(num_predictors)
        ])
        
        # 2. Create the Aggregator model (standard architecture)
        # IMPORTANT: keep aggregator sequence aligned to original time steps
        aggregator_max_len = (max_len - 1)
        self.aggregator = DiverseTransformerLM(vocab_size, d_model, num_heads, num_layers, 
                                             aggregator_max_len, variant='standard')
        
        # 3. Improved gating mechanism with attention-like scoring
        self.predictor_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),  # Added normalization for stability
            nn.Linear(d_model // 2, 1)
        )
        
        # Normalizations to stabilize scales
        self.pre_gate_norm = nn.LayerNorm(d_model)
        self.post_fuse_norm = nn.LayerNorm(d_model)
        self.gate_dropout = nn.Dropout(0.1)
        
        # Added temperature parameter for gating
        self.gate_temperature = nn.Parameter(torch.ones(1) * 1.0)
        
        # 4. Add additional MLP for refining aggregated embeddings
        self.refine_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, src, gumbel_tau=1.0, return_aux_info=False):
        monologue_embeddings = []
        predictor_outputs = []
        raw_predictor_outputs = []
        
        for i, predictor in enumerate(self.predictors):
            # Get logits from each predictor
            predictor_logits = predictor(src)
            # Clamp logits to avoid extreme magnitudes
            predictor_logits = torch.clamp(predictor_logits, min=-20.0, max=20.0)
            raw_predictor_outputs.append(predictor_logits)
            
            # Use logits directly as predictor outputs
            batch_size, seq_len, vocab_size = predictor_logits.shape
            predictor_outputs.append(predictor_logits)
            
            # IMPROVED GUMBEL SOFTMAX: Gradually transition from soft to harder samples
            hard = not self.training or gumbel_tau < 0.8  # Use hard samples at lower temperatures
            
            differentiable_one_hot = F.gumbel_softmax(
                predictor_logits, tau=gumbel_tau, hard=hard
            )
            
            # Convert the one-hot vector into a differentiable embedding
            soft_embeddings = torch.matmul(differentiable_one_hot, self.aggregator.embedding.weight)
            monologue_embeddings.append(soft_embeddings)
            
        # Fuse predictor embeddings with improved gating
        monologue_stack = torch.stack(monologue_embeddings, dim=1)
        monologue_stack = self.pre_gate_norm(monologue_stack)
        
        # Improved gating scores with temperature-controlled sharpness
        gate_scores = self.predictor_gate(monologue_stack)
        # Use learned temperature parameter for adaptive sharpness
        gate_scores = gate_scores / (self.gate_temperature + 0.1)  # Add 0.1 for numerical stability
        gate_weights = torch.softmax(gate_scores, dim=1)  # softmax across predictors
        gate_weights = self.gate_dropout(gate_weights)
        
        # Weighted sum across predictor dim -> [B, S, D]
        aggregator_input_embeddings = (gate_weights * monologue_stack).sum(dim=1)
        
        # NEW: Refine aggregated embeddings with MLP for better representation
        aggregator_input_embeddings = self.refine_mlp(aggregator_input_embeddings)
        
        # Post-fusion normalization
        aggregator_input_embeddings = self.post_fuse_norm(aggregator_input_embeddings)
        
        # --- Manually run the Aggregator's forward pass with our soft embeddings ---
        agg_seq_len = aggregator_input_embeddings.shape[1]
        x = aggregator_input_embeddings + self.aggregator.pos_encoder[:, :agg_seq_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(agg_seq_len).to(src.device)
        aggregator_output = self.aggregator.transformer_encoder(x, mask=mask)
        final_logits = self.aggregator.head(aggregator_output)
        
        if return_aux_info:
            return final_logits, predictor_outputs, monologue_embeddings, raw_predictor_outputs, gate_weights
        return final_logits

    def compute_auxiliary_losses(self, predictor_outputs, monologue_embeddings, targets, raw_predictor_outputs, gate_weights=None):
        """
        Stable, non-negative auxiliary losses to guide training:
        - diversity: average pairwise cosine similarity between predictor embeddings (>=0)
        - smoothness: temporal smoothness of embeddings (>=0)
        - consistency: disagreement rate among confident predictors (in [0,1])
        - information: match position-wise entropy schedule (>=0)
        - gate_diversity: encourage diverse gate usage (in [0,1])
        """
        aux_losses = {}
        device = predictor_outputs[0].device if predictor_outputs else targets.device
        eps = 1e-8

        # 1) Diversity loss (non-negative): minimize average pairwise cosine similarity
        if len(monologue_embeddings) > 1:
            mean_embs = []
            for emb in monologue_embeddings:  # each: [B, S, D]
                if emb is None:
                    continue
                m = emb.mean(dim=(0, 1), keepdim=False)  # [D]
                m = F.normalize(m, dim=-1)
                mean_embs.append(m)
            if len(mean_embs) > 1:
                sims = []
                for i in range(len(mean_embs)):
                    for j in range(i + 1, len(mean_embs)):
                        sim = torch.sum(mean_embs[i] * mean_embs[j])  # cosine because normalized
                        sims.append((sim + 1.0) * 0.5)  # map to [0,1]
                aux_losses['diversity'] = torch.stack(sims).mean() if sims else torch.tensor(0.0, device=device)
            else:
                aux_losses['diversity'] = torch.tensor(0.0, device=device)
        else:
            aux_losses['diversity'] = torch.tensor(0.0, device=device)

        # 2) Smoothness loss
        smooth_vals = []
        for emb in monologue_embeddings:
            if emb is None or emb.shape[1] < 2:
                continue
            deltas = emb[:, 1:, :] - emb[:, :-1, :]
            smooth_vals.append(deltas.pow(2).mean())
        aux_losses['smoothness'] = torch.stack(smooth_vals).mean() if smooth_vals else torch.tensor(0.0, device=device)

        # 3) Consistency loss
        if len(raw_predictor_outputs) > 1:
            top_tokens = []
            top_probs = []
            for logits in raw_predictor_outputs:  # [B, S, V]
                probs = F.softmax(logits, dim=-1)
                pvals, tidx = probs.max(dim=-1)  # [B, S]
                top_tokens.append(tidx)
                top_probs.append(pvals)
            top_tokens = torch.stack(top_tokens, dim=0)  # [P, B, S]
            top_probs = torch.stack(top_probs, dim=0)    # [P, B, S]
            conf_mask = top_probs >= 0.6
            P = top_tokens.shape[0]
            pairs = []
            for i in range(P):
                for j in range(i + 1, P):
                    valid = conf_mask[i] & conf_mask[j] & (targets != pad_token_id)
                    if valid.any():
                        agree = (top_tokens[i] == top_tokens[j]) & valid
                        disagree_rate = 1.0 - (agree.sum().float() / (valid.sum().float() + eps))
                        pairs.append(disagree_rate)
            aux_losses['consistency'] = torch.stack(pairs).mean() if pairs else torch.tensor(0.0, device=device)
        else:
            aux_losses['consistency'] = torch.tensor(0.0, device=device)

        # 4) Information loss (position-wise entropy schedule)
        info_loss = 0.0
        valid_predictors = 0
        for pred_logits in raw_predictor_outputs:
            probs = F.softmax(pred_logits, dim=-1)
            entropy = -(probs * torch.log(torch.clamp(probs, min=eps))).sum(dim=-1)  # [B, S]
            seq_len = entropy.shape[1]
            target_entropy = torch.linspace(2.0, 0.5, seq_len, device=entropy.device)
            seq_mask = (targets != pad_token_id)
            masked_entropy = entropy * seq_mask.float()
            avg_entropy_per_pos = masked_entropy.sum(0) / (seq_mask.sum(0).float() + eps)
            pos_info_loss = F.mse_loss(avg_entropy_per_pos, target_entropy)
            info_loss += pos_info_loss
            valid_predictors += 1
        aux_losses['information'] = info_loss / valid_predictors if valid_predictors > 0 else torch.tensor(0.0, device=device)

        # 5) Gate diversity
        if gate_weights is not None:
            # gate_weights: [B, P, S, 1]
            avg_gate = gate_weights.mean(dim=(0, 2, 3))  # [P]
            gate_entropy = -(avg_gate * torch.log(torch.clamp(avg_gate, min=eps))).sum()
            max_entropy = math.log(float(self.num_predictors)) if self.num_predictors > 1 else 1.0
            aux_losses['gate_diversity'] = 1.0 - (gate_entropy / (max_entropy + eps))

        return aux_losses

# --- Data Preparation (Same as before) ---
print("## Step 0: Preparing Dataset and Tokenizer ##")
# Choose dataset size based on testing mode
if LOCAL_TESTING:
    print("LOCAL TESTING MODE: Using minimal dataset")
    full_dataset = load_dataset("roneneldan/TinyStories", split="train[:500]")  # Very small for testing
else:
    full_dataset = load_dataset("roneneldan/TinyStories", split="train[:20000]")
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]", "[PAD]"])
tokenizer.train_from_iterator(full_dataset['text'], trainer=trainer)
pad_token_id = tokenizer.token_to_id("[PAD]")

def prepare_data(dataset, tokenizer, max_len):
    all_token_ids = []
    for text in tqdm(dataset['text'], desc="Tokenizing"):
        all_token_ids.extend(tokenizer.encode(text).ids)
    chunks = [all_token_ids[i:i + max_len] for i in range(0, len(all_token_ids), max_len)]
    chunks = [chunk for chunk in chunks if len(chunk) == max_len]
    data_tensor = torch.tensor(chunks, dtype=torch.long)
    inputs = data_tensor[:, :-1]
    targets = data_tensor[:, 1:]
    return TensorDataset(inputs, targets)

processed_dataset = prepare_data(full_dataset, tokenizer, MAX_SEQ_LEN)
loader = DataLoader(processed_dataset, batch_size=BATCH_SIZE, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models", exist_ok=True)


# --- Unified Training Loop ---
print("\n## Starting End-to-End Training ##")
model = PredictorAggregatorLM(VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, MAX_SEQ_LEN, NUM_PREDICTORS).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# Use a better loss function that handles padding properly
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction='mean', label_smoothing=0.1)  # Added label smoothing

# Add cosine annealing scheduler with longer cycle
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-WARMUP_EPOCHS, eta_min=LR*0.01)

def get_current_gumbel_tau(epoch, total_epochs):
    """Improved annealing schedule with slower decay"""
    progress = min(epoch / max(1, total_epochs - WARMUP_EPOCHS), 1.0)
    # Slower annealing schedule - stay at higher temperatures longer
    return GUMBEL_TAU_START * (1 - progress**2) + GUMBEL_TAU_END * progress**2

def get_warmup_lr(epoch, base_lr, warmup_epochs):
    """Linear warmup for learning rate"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr

# Curriculum learning - start with shorter sequences
def get_curriculum_mask(inputs, targets, epoch, total_epochs):
    """Create a curriculum mask that focuses on progressively longer sequences"""
    if epoch < total_epochs // 3:
        # First third of training: focus on first 1/3 of sequence
        seq_len = inputs.size(1)
        curriculum_len = max(seq_len // 3, 1)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, :curriculum_len] = True
        return mask
    elif epoch < 2 * (total_epochs // 3):
        # Second third: focus on first 2/3 of sequence
        seq_len = inputs.size(1)
        curriculum_len = max(2 * (seq_len // 3), 1)
        mask = torch.zeros_like(targets, dtype=torch.bool)
        mask[:, :curriculum_len] = True
        return mask
    else:
        # Final third: use full sequence
        return torch.ones_like(targets, dtype=torch.bool)

print(f"Training on {len(loader)} batches")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
if LOCAL_TESTING:
    print("🧪 LOCAL TESTING MODE: Reduced model size and dataset for quick validation")
    print(f"   - Model size: ~{sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    print(f"   - Dataset: {len(processed_dataset)} samples")
    print(f"   - Sequence length: {MAX_SEQ_LEN}, Model dim: {D_MODEL}")
else:
    print("🚀 FULL TRAINING MODE: Large model and dataset")

# Track training metrics
train_losses = []
learning_rates = []
gumbel_taus = []
aux_losses_history = {
    'diversity': [], 'smoothness': [], 'information': [], 'consistency': [], 'gate_diversity': []
}

# Add gradient accumulation for stability
GRAD_ACCUM_STEPS = 4 if not LOCAL_TESTING else 1

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    epoch_aux_losses = {
        'diversity': 0.0, 'smoothness': 0.0, 'information': 0.0, 
        'consistency': 0.0, 'gate_diversity': 0.0
    }
    num_batches = 0
    skipped_batches = 0
    
    # Get current Gumbel temperature (improved annealing)
    current_tau = get_current_gumbel_tau(epoch, EPOCHS)
    gumbel_taus.append(current_tau)
    
    # Apply learning rate warmup
    if epoch < WARMUP_EPOCHS:
        current_lr = get_warmup_lr(epoch, LR, WARMUP_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    
    # Set model to training mode
    model.train()
    
    # Reset gradients at the beginning of each epoch
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Get curriculum mask for this epoch
        curriculum_mask = get_curriculum_mask(inputs, targets, epoch, EPOCHS).to(device)
        
        # Forward pass with auxiliary information
        final_logits, predictor_outputs, monologue_embeddings, raw_predictor_outputs, gate_weights = model(
            inputs, gumbel_tau=current_tau, return_aux_info=True
        )
        
        # Sanity check logits
        if not torch.isfinite(final_logits).all():
            print(f"Non-finite logits at batch {batch_idx}; skipping batch")
            skipped_batches += 1
            continue
        
        # Apply curriculum mask to focus on specific parts of sequence
        masked_targets = targets.clone()
        masked_targets[~curriculum_mask] = pad_token_id  # Treat non-curriculum positions as padding
        
        # Main loss - CrossEntropyLoss handles flattening internally
        main_loss = criterion(final_logits.view(-1, VOCAB_SIZE), masked_targets.view(-1))
        
        # Auxiliary losses with improved computation
        aux_losses = model.compute_auxiliary_losses(
            predictor_outputs, monologue_embeddings, masked_targets, 
            raw_predictor_outputs, gate_weights
        )
        
        # Dynamic weighting of auxiliary losses based on training progress
        # Start with low weights and gradually increase
        progress = epoch / EPOCHS
        diversity_weight = DIVERSITY_LOSS_WEIGHT * min(1.0, progress * 2)  # Ramp up over first half
        consistency_weight = CONSISTENCY_LOSS_WEIGHT * min(1.0, progress * 3)  # Ramp up over first third
        
        # Combine losses with adaptive weights
        total_loss = main_loss
        total_loss += diversity_weight * aux_losses.get('diversity', 0.0)
        total_loss += 0.1 * aux_losses.get('smoothness', 0.0)
        total_loss += 0.05 * aux_losses.get('information', 0.0)
        total_loss += consistency_weight * aux_losses.get('consistency', 0.0)
        total_loss += 0.1 * aux_losses.get('gate_diversity', 0.0)  # Add new gate diversity loss
        
        # Scale loss for gradient accumulation
        total_loss = total_loss / GRAD_ACCUM_STEPS
        
        # Check for NaN/Inf
        if not torch.isfinite(total_loss):
            print(f"NaN/Inf detected in loss at batch {batch_idx}; skipping batch")
            skipped_batches += 1
            continue
        
        # Backward pass with gradient accumulation
        total_loss.backward()
        
        # Only step optimizer and zero gradients after accumulation
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0 or (batch_idx + 1) == len(loader):
            # Gradient clipping
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(torch.tensor(float(total_norm))):
                print(f"Non-finite grad norm ({total_norm}) at batch {batch_idx}; skipping optimizer.step")
                optimizer.zero_grad(set_to_none=True)
                skipped_batches += 1
                continue
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Track losses
        epoch_loss += main_loss.item() * GRAD_ACCUM_STEPS  # Adjust for gradient accumulation
        num_batches += 1
        
        # Track auxiliary losses
        for loss_name, loss_value in aux_losses.items():
            loss_val = loss_value.item() if torch.is_tensor(loss_value) else loss_value
            if not math.isnan(loss_val) and math.isfinite(loss_val):
                epoch_aux_losses[loss_name] += loss_val
        
        # Print loss every 25 batches
        if batch_idx % 25 == 0:
            print(f"Batch {batch_idx}, Main Loss: {main_loss.item():.4f}, Total Loss: {total_loss.item()*GRAD_ACCUM_STEPS:.4f}, Tau: {current_tau:.3f}")
            aux_str = []
            for loss_name, loss_value in aux_losses.items():
                val = loss_value.item() if torch.is_tensor(loss_value) else loss_value
                aux_str.append(f"{loss_name.capitalize()[:3]}={val:.3f}")
            print(f"  Aux: {', '.join(aux_str)}")
    
    # Guard against zero valid batches
    if num_batches == 0:
        print(f"Epoch {epoch+1}: all {skipped_batches} batches skipped due to non-finite losses.")
        avg_loss = float('nan')
        current_lr = optimizer.param_groups[0]['lr']
        # Record placeholders to keep downstream plotting consistent
        train_losses.append(avg_loss)
        learning_rates.append(current_lr)
        for loss_name in epoch_aux_losses:
            aux_losses_history[loss_name].append(float('nan'))
        # Reduce LR to attempt recovery next epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(param_group['lr'] * 0.5, LR * 0.1)
        print(f"  Adjusted LR to {optimizer.param_groups[0]['lr']:.6f} for recovery.")
        continue
    
    avg_loss = epoch_loss / num_batches
    current_lr = optimizer.param_groups[0]['lr']
    
    # Average auxiliary losses
    for loss_name in epoch_aux_losses:
        epoch_aux_losses[loss_name] /= num_batches
        aux_losses_history[loss_name].append(epoch_aux_losses[loss_name])
    
    print(f"Epoch {epoch+1} completed. Main Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Tau: {current_tau:.3f}")
    print(f"  Aux Losses - Diversity: {epoch_aux_losses['diversity']:.4f}, Smoothness: {epoch_aux_losses['smoothness']:.4f}, Info: {epoch_aux_losses['information']:.4f}, Consistency: {epoch_aux_losses['consistency']:.4f}")
    
    # Track metrics
    train_losses.append(avg_loss)
    learning_rates.append(current_lr)
    
    # Step the scheduler (only after warmup)
    if epoch >= WARMUP_EPOCHS:
        scheduler.step()

torch.save(model.state_dict(), "models/end_to_end_model.pt")
print("End-to-end model trained and saved successfully!")

# --- Enhanced Text Generation with Predictor Outputs ---
print("\n## Testing Enhanced Text Generation with Predictor Outputs ##")

def generate_text_with_predictors(model, tokenizer, prompt, max_length=20, temperature=1.0, top_k=40, top_p=0.92):
    """
    Generate text while collecting per-step gate weights and per-predictor outputs.
    Returns a dict with:
      - generated_text: final aggregated text
      - predictor_outputs: list of (text, avg_confidence) for each predictor
      - gate_weights: list[step] -> list[num_predictors] of weights
    """
    model.eval()

    # Tokenize prompt
    tokens = tokenizer.encode(prompt).ids if prompt and prompt.strip() else [tokenizer.token_to_id("[UNK]")]
    if len(tokens) >= MAX_SEQ_LEN - 1:
        tokens = tokens[:MAX_SEQ_LEN - 1]

    generated_tokens = tokens.copy()
    # For each predictor, store tokens and confidences per step
    per_pred_tokens = [[] for _ in range(len(model.predictors))]
    per_pred_confidences = [[] for _ in range(len(model.predictors))]
    gate_weights_steps = []

    with torch.no_grad():
        for _ in range(max_length):
            input_tokens = generated_tokens[-MAX_SEQ_LEN+1:] if len(generated_tokens) >= MAX_SEQ_LEN-1 else generated_tokens
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
            # Right pad to expected length
            if input_tensor.shape[1] < MAX_SEQ_LEN - 1:
                padding = torch.full((1, MAX_SEQ_LEN - 1 - input_tensor.shape[1]), pad_token_id, dtype=torch.long, device=device)
                input_tensor = torch.cat([input_tensor, padding], dim=1)

            # Collect per-predictor next-token and confidence
            for i, predictor in enumerate(model.predictors):
                pred_logits = predictor(input_tensor)  # [1, S, V]
                pos = len(input_tokens) - 1
                pred_probs = F.softmax(pred_logits[0, pos, :], dim=-1)
                pred_conf, pred_tok = torch.max(pred_probs, dim=-1)
                per_pred_tokens[i].append(int(pred_tok.item()))
                per_pred_confidences[i].append(float(pred_conf.item()))

            # Aggregated output and gate weights
            output, _, _, _, gw = model(input_tensor, return_aux_info=True)
            pos = len(input_tokens) - 1
            # gw shape expected [B, P, S, 1]
            if gw is not None:
                step_weights = gw[0, :, pos, 0].detach().cpu().tolist()
                gate_weights_steps.append(step_weights)

            # Get last position logits and sample
            last_token_logits = output[0, pos, :] / max(temperature, 1e-3)
            last_token_logits = torch.clamp(last_token_logits, -20.0, 20.0)

            if top_k and top_k > 0:
                kth = torch.topk(last_token_logits, min(top_k, last_token_logits.size(-1)))[0][-1]
                last_token_logits[last_token_logits < kth] = float('-inf')

            if top_p and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                last_token_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(last_token_logits, dim=-1)
            next_token = int(torch.argmax(last_token_logits).item()) if (not torch.isfinite(probs).all() or probs.sum() <= 0) else int(torch.multinomial(probs, 1).item())
            if next_token == pad_token_id:
                break
            generated_tokens.append(next_token)

    # Build outputs
    try:
        aggregated_text = tokenizer.decode(generated_tokens)
    except Exception:
        aggregated_text = f"Generated tokens: {generated_tokens}"

    predictor_outputs = []
    for i in range(len(model.predictors)):
        try:
            pred_text = tokenizer.decode(tokens + per_pred_tokens[i])
        except Exception:
            pred_text = f"Tokens: {tokens + per_pred_tokens[i]}"
        avg_conf = float(np.mean(per_pred_confidences[i])) if per_pred_confidences[i] else 0.0
        predictor_outputs.append((pred_text, avg_conf))

    return {
        'generated_text': aggregated_text,
        'predictor_outputs': predictor_outputs,
        'gate_weights': gate_weights_steps,
    }

def generate_text_with_predictor_outputs(model, tokenizer, prompt, max_length=20, temperature=1.0, top_k=40, top_p=0.92):
    """Text generation with visibility into each predictor's outputs"""
    model.eval()
    
    # Tokenize the prompt
    if prompt.strip():
        tokens = tokenizer.encode(prompt).ids
    else:
        tokens = [tokenizer.token_to_id("[UNK]")]  # Start with unknown token if no prompt
    
    # Ensure we don't exceed max sequence length
    if len(tokens) >= MAX_SEQ_LEN - 1:
        tokens = tokens[:MAX_SEQ_LEN - 1]
    
    generated_tokens = tokens.copy()
    predictor_token_history = [[] for _ in range(len(model.predictors))]
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input (pad if necessary)
            input_tokens = generated_tokens[-MAX_SEQ_LEN+1:] if len(generated_tokens) >= MAX_SEQ_LEN-1 else generated_tokens
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
            
            # Pad to match expected input length
            if input_tensor.shape[1] < MAX_SEQ_LEN - 1:
                padding = torch.full((1, MAX_SEQ_LEN - 1 - input_tensor.shape[1]), pad_token_id, dtype=torch.long).to(device)
                input_tensor = torch.cat([input_tensor, padding], dim=1)
            
            # NEW: Get each predictor's output and the final output
            predictor_outputs = []
            for i, predictor in enumerate(model.predictors):
                pred_logits = predictor(input_tensor)
                pred_token = torch.argmax(pred_logits[0, len(input_tokens)-1, :]).item()
                predictor_outputs.append(pred_token)
                predictor_token_history[i].append(pred_token)
            
            # Generate aggregated prediction
            output, _, _, _, gate_weights = model(input_tensor, return_aux_info=True)
            
            # Get the last token's logits
            last_token_logits = output[0, len(input_tokens)-1, :] / max(temperature, 1e-3)
            
            # Clamp logits to avoid numerical issues
            last_token_logits = torch.clamp(last_token_logits, -20.0, 20.0)
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = last_token_logits < torch.topk(last_token_logits, top_k)[0][-1]
                last_token_logits[indices_to_remove] = float('-inf')
                
            # Apply nucleus (top-p) filtering
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(last_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                last_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the distribution
            probs = F.softmax(last_token_logits, dim=-1)
            if not torch.isfinite(probs).all() or probs.sum() <= 0:
                # Fallback to argmax
                next_token = torch.argmax(last_token_logits).item()
            else:
                next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Stop if we hit padding token
            if next_token == pad_token_id:
                break
                
            generated_tokens.append(next_token)
    
    # Decode the generated tokens
    try:
        generated_text = tokenizer.decode(generated_tokens)
        
        # NEW: Decode predictor outputs
        predictor_texts = []
        for i, pred_tokens in enumerate(predictor_token_history):
            # Combine original prompt tokens with predictor's generated tokens
            full_pred_sequence = tokens + pred_tokens
            pred_text = tokenizer.decode(full_pred_sequence)
            predictor_texts.append(pred_text)
        
        return generated_text, predictor_texts
    except Exception as e:
        return f"Generated tokens: {generated_tokens}", [f"Error decoding predictor {i}" for i in range(len(model.predictors))]

# Test with a few different prompts
test_prompts = [
    "Once upon a time",
    "The little girl",
    "In the forest",
    ""  # Empty prompt to see what the model generates from scratch
]

for i, prompt in enumerate(test_prompts):
    print(f"\nTest {i+1}: Prompt: '{prompt}'")
    result = generate_text_with_predictors(model, tokenizer, prompt, max_length=15, temperature=0.8)
    print(f"Final Aggregated Output: {result['generated_text']}")

    print("Individual Predictor Outputs:")
    for p_idx, (p_text, p_conf) in enumerate(result['predictor_outputs']):
        variant = ['standard', 'deep_narrow', 'wide_shallow', 'regularized'][p_idx % 4]
        print(f"  Predictor {p_idx+1} ({variant}), Confidence: {p_conf:.3f}")
        print(f"    {p_text}")

    if result['gate_weights']:
        print("\nAggregator Weights by Position:")
        for pos, weights in enumerate(result['gate_weights']):
            formatted_weights = [f"{w:.3f}" for w in weights]
            print(f"  Pos {pos+1}: {formatted_weights}")
    
    print(f"{'='*50}")

print("\nEnhanced text generation test completed!")

# --- Visualize Predictor Contributions ---
print("\n## Creating Predictor Contribution Visualizations ##")
os.makedirs("output", exist_ok=True)

# Create visualizations for the last generated text
prompt_idx = len(test_prompts) - 1
vis_prompt = test_prompts[prompt_idx] if test_prompts[prompt_idx] else "[empty prompt]"
result = generate_text_with_predictors(model, tokenizer, test_prompts[prompt_idx], max_length=30, temperature=0.8)

if result['gate_weights']:
    # Plot gate weights over generation steps
    plt.figure(figsize=(12, 6))
    weights_array = np.array(result['gate_weights'])
    steps = range(weights_array.shape[0])
    for p_idx in range(weights_array.shape[1]):
        variant = ['standard', 'deep_narrow', 'wide_shallow', 'regularized'][p_idx % 4]
        plt.plot(steps, weights_array[:, p_idx], 
                 label=f"Predictor {p_idx+1} ({variant})", 
                 marker='o', linewidth=2)

    plt.title(f"Predictor Contributions During Generation\nPrompt: '{vis_prompt}'")
    plt.xlabel("Generation Step")
    plt.ylabel("Gate Weight")
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    plt.savefig('output/predictor_contributions.png', dpi=300, bbox_inches='tight')
    print("Predictor contribution visualization saved to output/predictor_contributions.png")
else:
    print("No gate weights collected during generation; skipping predictor contribution plot.")

# --- Plotting Training Metrics ---
print("\n## Generating Training Plots ##")

# Create a comprehensive training dashboard
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

# Plot 1: Loss over time
axes[0, 0].plot(range(1, len(train_losses) + 1), train_losses, 'b-', linewidth=2, label='Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss Over Time')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Plot 2: Learning Rate Schedule
axes[0, 1].plot(range(1, len(learning_rates) + 1), learning_rates, 'r-', linewidth=2, label='Learning Rate')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Learning Rate')
axes[0, 1].set_title('Learning Rate Schedule (Warmup + Cosine Annealing)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Plot 3: Gumbel Temperature
axes[0, 2].plot(range(1, len(gumbel_taus) + 1), gumbel_taus, 'g-', linewidth=2, label='Gumbel Temperature')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Temperature')
axes[0, 2].set_title('Gumbel-Softmax Temperature Annealing')
axes[0, 2].grid(True, alpha=0.3)
axes[0, 2].legend()

# Plot 4: All auxiliary losses together
colors = ['purple', 'orange', 'brown', 'pink']
loss_names = ['diversity', 'smoothness', 'information', 'consistency']
for loss_name, color in zip(loss_names, colors):
    if aux_losses_history[loss_name]:
        axes[0, 3].plot(range(1, len(aux_losses_history[loss_name]) + 1), 
                       aux_losses_history[loss_name], color=color, linewidth=2, 
                       label=f'{loss_name.title()}')
axes[0, 3].set_xlabel('Epoch')
axes[0, 3].set_ylabel('Loss Value')
axes[0, 3].set_title('All Auxiliary Losses')
axes[0, 3].grid(True, alpha=0.3)
axes[0, 3].legend()

# Plot 5-8: Individual auxiliary losses
for i, (loss_name, color) in enumerate(zip(loss_names, colors)):
    if aux_losses_history[loss_name]:
        axes[1, i].plot(range(1, len(aux_losses_history[loss_name]) + 1), 
                       aux_losses_history[loss_name], color=color, linewidth=2, 
                       label=f'{loss_name.title()} Loss')
        axes[1, i].set_xlabel('Epoch')
        axes[1, i].set_ylabel('Loss Value')
        axes[1, i].set_title(f'{loss_name.title()} Auxiliary Loss')
        axes[1, i].grid(True, alpha=0.3)
        axes[1, i].legend()

plt.tight_layout()
plt.savefig('output/training_dashboard.png', dpi=300, bbox_inches='tight')
print("Training dashboard saved to output/training_dashboard.png")

# --- Analysis of Loss Plateau ---
print("\n## Loss Analysis ##")
if len(train_losses) >= 3:
    loss_improvement = train_losses[0] - train_losses[-1]
    recent_variance = torch.var(torch.tensor(train_losses[-3:])).item() if len(train_losses) >= 3 else 0
    
    print(f"Initial Loss: {train_losses[0]:.4f}")
    print(f"Final Loss: {train_losses[-1]:.4f}")
    print(f"Total Improvement: {loss_improvement:.4f}")
    print(f"Recent Loss Variance (last 3 epochs): {recent_variance:.6f}")
    
    if recent_variance < 0.01 and loss_improvement < 0.5:
        print("\n🚨 POTENTIAL ISSUES DETECTED:")
        print("1. Loss plateau detected - consider:")
        print("   - Increasing learning rate")
        print("   - Adding gradient clipping")
        print("   - Reducing Gumbel temperature over time")
        print("   - Adding weight decay for regularization")
        print("   - Using learning rate warmup")
        print("2. The predictor-aggregator architecture may need:")
        print("   - Different predictor architectures for diversity")
        print("   - Auxiliary losses on individual predictors")
        print("   - Temperature annealing for Gumbel-Softmax")

plt.close('all')  # Clean up matplotlib

print("\n## Enhanced Generation Demo Complete ##")
print(f"- Visualized contributions from {model.num_predictors} different predictors")
print("- Individual predictor outputs now shown during text generation")
print("- Gate weight visualization shows which predictor influences each token")
