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
LR = 0.0005        # Reduced from 0.001 for better stability
NUM_PREDICTORS = 2
GUMBEL_TAU_START = 2.0  # Start higher for exploration
GUMBEL_TAU_END = 0.1    # End lower for exploitation
WARMUP_EPOCHS = 2       # Learning rate warmup
DIVERSITY_LOSS_WEIGHT = 0.1  # Weight for predictor diversity loss

# --- Generic Transformer for building blocks ---
class SimpleTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        # This forward is for a standard LM (used by Predictors)
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
        
        # 1. Create diverse Predictor models with different dropout rates
        self.predictors = nn.ModuleList([
            SimpleTransformerLM(vocab_size, d_model, num_heads, num_layers, max_len)
            for _ in range(num_predictors)
        ])
        
        # Apply different dropout rates to encourage diversity
        for i, predictor in enumerate(self.predictors):
            dropout_rate = 0.1 + i * 0.05  # Different dropout for each predictor
            for layer in predictor.transformer_encoder.layers:
                layer.dropout.p = dropout_rate
        
        # 2. Create the Aggregator model
        aggregator_max_len = num_predictors * (max_len - 1)
        self.aggregator = SimpleTransformerLM(vocab_size, d_model, num_heads, num_layers, aggregator_max_len)

    def forward(self, src, gumbel_tau=1.0, return_aux_info=False):
        monologue_embeddings = []
        predictor_outputs = []
        
        for predictor in self.predictors:
            # Get logits from each predictor
            predictor_logits = predictor(src)
            predictor_outputs.append(predictor_logits)
            
            # ** THE KEY STEP **
            # Use Gumbel-Softmax with dynamic temperature
            differentiable_one_hot = F.gumbel_softmax(predictor_logits, tau=gumbel_tau, hard=True)
            
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
            return final_logits, predictor_outputs, monologue_embeddings
        return final_logits
    
    def compute_auxiliary_losses(self, predictor_outputs, monologue_embeddings, targets):
        """
        Compute auxiliary losses to help predictors prepare better tokens for the aggregator
        """
        aux_losses = {}
        
        # 1. DIVERSITY LOSS: Encourage predictors to produce different outputs
        # This helps each predictor specialize in different aspects
        if len(predictor_outputs) > 1:
            diversity_loss = 0
            for i in range(len(predictor_outputs)):
                for j in range(i + 1, len(predictor_outputs)):
                    # Compute similarity between predictor outputs
                    sim = F.cosine_similarity(
                        predictor_outputs[i].view(-1, self.vocab_size),
                        predictor_outputs[j].view(-1, self.vocab_size),
                        dim=1
                    ).mean()
                    # Penalize high similarity (we want diversity)
                    diversity_loss += sim
            aux_losses['diversity'] = diversity_loss / (len(predictor_outputs) * (len(predictor_outputs) - 1) / 2)
        
        # 2. EMBEDDING SMOOTHNESS LOSS: Encourage smooth transitions in embeddings
        # This helps the aggregator receive more coherent information
        smoothness_loss = 0
        for emb in monologue_embeddings:
            # Compute differences between consecutive embeddings
            if emb.shape[1] > 1:
                diff = emb[:, 1:, :] - emb[:, :-1, :]
                smoothness_loss += torch.norm(diff, dim=-1).mean()
        aux_losses['smoothness'] = smoothness_loss / len(monologue_embeddings)
        
        # 3. INFORMATION PRESERVATION LOSS: Ensure predictors don't lose important info
        # Compare predictor attention patterns to encourage information retention
        info_loss = 0
        for pred_logits in predictor_outputs:
            # Encourage high confidence (low entropy) where it makes sense
            probs = F.softmax(pred_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
            # Penalize too much uncertainty in early positions (where context is clear)
            seq_len = entropy.shape[1]
            position_weights = torch.linspace(1.0, 0.1, seq_len).to(entropy.device)
            weighted_entropy = (entropy * position_weights).mean()
            info_loss += weighted_entropy
        aux_losses['information'] = info_loss / len(predictor_outputs)
        
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
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

# Add cosine annealing scheduler (will start after warmup)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS-WARMUP_EPOCHS, eta_min=LR*0.1)

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
aux_losses_history = {'diversity': [], 'smoothness': [], 'information': []}

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    epoch_aux_losses = {'diversity': 0.0, 'smoothness': 0.0, 'information': 0.0}
    num_batches = 0
    
    # Get current Gumbel temperature
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
        final_logits, predictor_outputs, monologue_embeddings = model(
            inputs, gumbel_tau=current_tau, return_aux_info=True
        )
        
        # Main loss
        main_loss = criterion(final_logits[:, :targets.shape[1], :].reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        
        # Auxiliary losses
        aux_losses = model.compute_auxiliary_losses(predictor_outputs, monologue_embeddings, targets)
        
        # Combine losses
        total_loss = main_loss
        for loss_name, loss_value in aux_losses.items():
            total_loss += DIVERSITY_LOSS_WEIGHT * loss_value
            epoch_aux_losses[loss_name] += loss_value.item()
        
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += main_loss.item()
        num_batches += 1
        
        # Print loss every 50 batches for monitoring
        if batch_idx % 50 == 0:
            print(f"Batch {batch_idx}, Main Loss: {main_loss.item():.4f}, Total Loss: {total_loss.item():.4f}, Tau: {current_tau:.3f}")
    
    avg_loss = epoch_loss / num_batches
    current_lr = optimizer.param_groups[0]['lr']
    
    # Average auxiliary losses
    for loss_name in epoch_aux_losses:
        epoch_aux_losses[loss_name] /= num_batches
        aux_losses_history[loss_name].append(epoch_aux_losses[loss_name])
    
    print(f"Epoch {epoch+1} completed. Main Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Tau: {current_tau:.3f}")
    print(f"  Aux Losses - Diversity: {epoch_aux_losses['diversity']:.4f}, Smoothness: {epoch_aux_losses['smoothness']:.4f}, Info: {epoch_aux_losses['information']:.4f}")
    
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
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

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

# Plot 4: Auxiliary Losses
colors = ['purple', 'orange', 'brown']
loss_names = ['diversity', 'smoothness', 'information']
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