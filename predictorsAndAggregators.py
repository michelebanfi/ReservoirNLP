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

# --- Configuration ---
VOCAB_SIZE = 5000  # Reduced from 4000
MAX_SEQ_LEN = 128   # Reduced from 32
D_MODEL = 1024      # Reduced from 64
NUM_LAYERS = 4     # Reduced from 2
NUM_HEADS = 8
BATCH_SIZE = 32   # Reduced from 32
EPOCHS = 10
LR = 0.001         # Increased back for better progress
WEIGHT_DECAY = 1e-4  # Add weight decay for regularization
NUM_PREDICTORS = 3   # More predictors for better diversity
GUMBEL_TAU_START = 5.0  # Start much higher for more exploration
GUMBEL_TAU_END = 0.5    # End higher to maintain differentiability
WARMUP_EPOCHS = 3       # Longer warmup
DIVERSITY_LOSS_WEIGHT = 0.5  # Stronger diversity pressure
CONSISTENCY_LOSS_WEIGHT = 0.2  # New loss for consistency

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
        aggregator_max_len = num_predictors * (max_len - 1)
        self.aggregator = DiverseTransformerLM(vocab_size, d_model, num_heads, num_layers, 
                                             aggregator_max_len, variant='standard')
        
        # 3. Add prediction head diversity
        self.predictor_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(d_model // 2, d_model)
            ) for _ in range(num_predictors)
        ])

    def forward(self, src, gumbel_tau=1.0, return_aux_info=False):
        monologue_embeddings = []
        predictor_outputs = []
        raw_predictor_outputs = []
        
        for i, predictor in enumerate(self.predictors):
            # Get logits from each predictor
            predictor_logits = predictor(src)
            raw_predictor_outputs.append(predictor_logits)
            
            # Apply predictor-specific projection for diversity
            batch_size, seq_len, vocab_size = predictor_logits.shape
            projected_logits = predictor_logits.view(-1, vocab_size)
            projected_logits = projected_logits.view(batch_size, seq_len, vocab_size)
            predictor_outputs.append(projected_logits)
            
            # ** THE KEY STEP **
            # Use Gumbel-Softmax with dynamic temperature
            differentiable_one_hot = F.gumbel_softmax(projected_logits, tau=gumbel_tau, hard=True)
            
            # Convert the one-hot vector into a differentiable embedding
            # This is a differentiable equivalent of an embedding lookup
            soft_embeddings = torch.matmul(differentiable_one_hot, self.aggregator.embedding.weight)
            monologue_embeddings.append(soft_embeddings)
            
        # Concatenate the "internal monologue" embeddings along the sequence dimension
        aggregator_input_embeddings = torch.cat(monologue_embeddings, dim=1)
        
        # --- Manually run the Aggregator's forward pass with our soft embeddings ---
        agg_seq_len = aggregator_input_embeddings.shape[1]
        x = aggregator_input_embeddings + self.aggregator.pos_encoder[:, :agg_seq_len, :]
        mask = nn.Transformer.generate_square_subsequent_mask(agg_seq_len).to(src.device)
        aggregator_output = self.aggregator.transformer_encoder(x, mask=mask)
        final_logits = self.aggregator.head(aggregator_output)
        
        if return_aux_info:
            return final_logits, predictor_outputs, monologue_embeddings, raw_predictor_outputs
        return final_logits
    
    def compute_auxiliary_losses(self, predictor_outputs, monologue_embeddings, targets, raw_predictor_outputs):
        """
        Improved auxiliary losses with better numerical stability
        """
        aux_losses = {}
        
        # 1. ENHANCED DIVERSITY LOSS: Multiple measures of diversity
        if len(predictor_outputs) > 1:
            diversity_loss = 0
            total_pairs = 0
            
            for i in range(len(predictor_outputs)):
                for j in range(i + 1, len(predictor_outputs)):
                    # L2 distance between probability distributions (better than cosine)
                    prob_i = F.softmax(predictor_outputs[i], dim=-1)
                    prob_j = F.softmax(predictor_outputs[j], dim=-1)
                    
                    # Jensen-Shannon divergence for better diversity measure
                    m = 0.5 * (prob_i + prob_j)
                    js_div = 0.5 * F.kl_div(F.log_softmax(predictor_outputs[i], dim=-1), m, reduction='none').sum(-1).mean()
                    js_div += 0.5 * F.kl_div(F.log_softmax(predictor_outputs[j], dim=-1), m, reduction='none').sum(-1).mean()
                    
                    # We want to MAXIMIZE diversity, so minimize negative JS divergence
                    diversity_loss += -js_div
                    total_pairs += 1
            
            aux_losses['diversity'] = diversity_loss / total_pairs if total_pairs > 0 else torch.tensor(0.0)
        
        # 2. IMPROVED SMOOTHNESS LOSS: Temporal consistency
        smoothness_loss = 0
        for emb in monologue_embeddings:
            if emb.shape[1] > 1:
                # Cosine similarity between consecutive embeddings (should be high)
                curr_emb = emb[:, :-1, :].reshape(-1, emb.shape[-1])
                next_emb = emb[:, 1:, :].reshape(-1, emb.shape[-1])
                
                cos_sim = F.cosine_similarity(curr_emb, next_emb, dim=-1)
                # Penalize low similarity (we want smooth transitions)
                smoothness_loss += (1 - cos_sim).mean()
        
        aux_losses['smoothness'] = smoothness_loss / len(monologue_embeddings) if len(monologue_embeddings) > 0 else torch.tensor(0.0)
        
        # 3. CONSISTENCY LOSS: Predictors should agree on confident predictions
        if len(raw_predictor_outputs) > 1:
            consistency_loss = 0
            for i in range(len(raw_predictor_outputs)):
                for j in range(i + 1, len(raw_predictor_outputs)):
                    # Get confidence masks (high-confidence predictions)
                    conf_i = F.softmax(raw_predictor_outputs[i], dim=-1).max(dim=-1)[0]
                    conf_j = F.softmax(raw_predictor_outputs[j], dim=-1).max(dim=-1)[0]
                    
                    # Where both are confident, they should agree
                    high_conf_mask = (conf_i > 0.7) & (conf_j > 0.7)
                    
                    if high_conf_mask.sum() > 0:
                        pred_i = raw_predictor_outputs[i].argmax(dim=-1)
                        pred_j = raw_predictor_outputs[j].argmax(dim=-1)
                        
                        # Agreement loss (should be low when predictions match)
                        agreement = (pred_i == pred_j).float()
                        consistency_loss += (1 - agreement[high_conf_mask]).mean()
            
            aux_losses['consistency'] = consistency_loss / (len(raw_predictor_outputs) * (len(raw_predictor_outputs) - 1) / 2)
        else:
            aux_losses['consistency'] = torch.tensor(0.0)
        
        # 4. INFORMATION PRESERVATION (improved)
        info_loss = 0
        for pred_logits in raw_predictor_outputs:
            # Encourage high confidence where it makes sense, but not everywhere
            probs = F.softmax(pred_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            
            # Target entropy: higher at beginning (more uncertainty), lower at end
            seq_len = entropy.shape[1]
            target_entropy = torch.linspace(2.0, 0.5, seq_len).to(entropy.device)
            
            # L2 loss between actual and target entropy
            entropy_loss = F.mse_loss(entropy.mean(0), target_entropy)
            info_loss += entropy_loss
        
        aux_losses['information'] = info_loss / len(raw_predictor_outputs)
        
        return aux_losses

# --- Data Preparation (Same as before) ---
print("## Step 0: Preparing Dataset and Tokenizer ##")
# Using a much smaller portion for testing (first 1000 samples)
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
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)  # Use AdamW for weight decay
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

# Add cosine annealing scheduler (will start after warmup)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-WARMUP_EPOCHS, eta_min=LR*0.01)

def get_current_gumbel_tau(epoch, total_epochs):
    """Linearly anneal Gumbel temperature from start to end"""
    progress = epoch / total_epochs
    return GUMBEL_TAU_START * (1 - progress) + GUMBEL_TAU_END * progress

def get_warmup_lr(epoch, base_lr, warmup_epochs):
    """Linear warmup for learning rate"""
    if epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    return base_lr

print(f"Training on {len(loader)} batches")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Track training metrics
train_losses = []
learning_rates = []
gumbel_taus = []
aux_losses_history = {'diversity': [], 'smoothness': [], 'information': [], 'consistency': []}

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    epoch_aux_losses = {'diversity': 0.0, 'smoothness': 0.0, 'information': 0.0, 'consistency': 0.0}
    num_batches = 0
    
    # Get current Gumbel temperature (slower annealing)
    current_tau = get_current_gumbel_tau(epoch, EPOCHS)
    gumbel_taus.append(current_tau)
    
    # Apply learning rate warmup
    if epoch < WARMUP_EPOCHS:
        current_lr = get_warmup_lr(epoch, LR, WARMUP_EPOCHS)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with auxiliary information
        final_logits, predictor_outputs, monologue_embeddings, raw_predictor_outputs = model(
            inputs, gumbel_tau=current_tau, return_aux_info=True
        )
        
        # Main loss
        main_loss = criterion(final_logits[:, :targets.shape[1], :].reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        
        # Auxiliary losses with improved computation
        aux_losses = model.compute_auxiliary_losses(predictor_outputs, monologue_embeddings, targets, raw_predictor_outputs)
        
        # Combine losses with adaptive weights
        total_loss = main_loss
        total_loss += DIVERSITY_LOSS_WEIGHT * aux_losses['diversity']
        total_loss += 0.1 * aux_losses['smoothness']  # Lower weight for smoothness
        total_loss += 0.05 * aux_losses['information']  # Lower weight for information
        total_loss += CONSISTENCY_LOSS_WEIGHT * aux_losses['consistency']  # New consistency loss
        
        # Check for NaN
        if torch.isnan(total_loss):
            print(f"NaN detected in loss at batch {batch_idx}")
            continue
        
        total_loss.backward()
        
        # Gradient clipping (stronger)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        optimizer.step()
        
        epoch_loss += main_loss.item()
        num_batches += 1
        
        # Track auxiliary losses
        for loss_name, loss_value in aux_losses.items():
            if not torch.isnan(loss_value):
                epoch_aux_losses[loss_name] += loss_value.item()
        
        # Print loss every 50 batches for monitoring
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}, Main Loss: {main_loss.item():.4f}, Total Loss: {total_loss.item():.4f}, Tau: {current_tau:.3f}")
            print(f"  Aux: Div={aux_losses['diversity'].item():.3f}, Smooth={aux_losses['smoothness'].item():.3f}, Info={aux_losses['information'].item():.3f}, Cons={aux_losses['consistency'].item():.3f}")
    
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

# --- Simple Text Generation Test ---
print("\n## Testing Text Generation ##")

def generate_text(model, tokenizer, prompt, max_length=20, temperature=1.0):
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
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input (pad if necessary)
            input_tokens = generated_tokens[-MAX_SEQ_LEN+1:] if len(generated_tokens) >= MAX_SEQ_LEN-1 else generated_tokens
            input_tensor = torch.tensor([input_tokens], dtype=torch.long).to(device)
            
            # Pad to match expected input length
            if input_tensor.shape[1] < MAX_SEQ_LEN - 1:
                padding = torch.full((1, MAX_SEQ_LEN - 1 - input_tensor.shape[1]), pad_token_id, dtype=torch.long).to(device)
                input_tensor = torch.cat([input_tensor, padding], dim=1)
            
            # Generate prediction
            output = model(input_tensor)
            
            # Get the last token's logits
            last_token_logits = output[0, len(input_tokens)-1, :] / temperature
            
            # Sample from the distribution
            probs = torch.softmax(last_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Stop if we hit padding token
            if next_token == pad_token_id:
                break
                
            generated_tokens.append(next_token)
    
    # Decode the generated tokens
    try:
        generated_text = tokenizer.decode(generated_tokens)
        return generated_text
    except:
        return f"Generated tokens: {generated_tokens}"

# Test with a few different prompts
test_prompts = [
    "Once upon a time",
    "The little girl",
    "In the forest",
    ""  # Empty prompt to see what the model generates from scratch
]

for i, prompt in enumerate(test_prompts):
    print(f"\nTest {i+1}: Prompt: '{prompt}'")
    generated = generate_text(model, tokenizer, prompt, max_length=15, temperature=0.8)
    print(f"Generated: {generated}")

print("\nText generation test completed!")

# --- Plotting Training Metrics ---
print("\n## Generating Training Plots ##")
os.makedirs("output", exist_ok=True)

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