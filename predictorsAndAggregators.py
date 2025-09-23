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

# --- Configuration ---
VOCAB_SIZE = 2000  # Reduced from 4000
MAX_SEQ_LEN = 16   # Reduced from 32
D_MODEL = 32       # Reduced from 64
NUM_LAYERS = 1     # Reduced from 2
NUM_HEADS = 2
BATCH_SIZE = 16    # Reduced from 32
EPOCHS = 1
LR = 0.001         # Slightly increased for faster convergence
NUM_PREDICTORS = 2
GUMBEL_TAU = 1.0 # Temperature for Gumbel-Softmax

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
        
        # 1. Create the Predictor models
        self.predictors = nn.ModuleList([
            SimpleTransformerLM(vocab_size, d_model, num_heads, num_layers, max_len)
            for _ in range(num_predictors)
        ])
        
        # 2. Create the Aggregator model
        aggregator_max_len = num_predictors * (max_len - 1)
        self.aggregator = SimpleTransformerLM(vocab_size, d_model, num_heads, num_layers, aggregator_max_len)

    def forward(self, src, targets=None):
        monologue_embeddings = []
        
        for predictor in self.predictors:
            # Get logits from each predictor
            predictor_logits = predictor(src)
            
            # ** THE KEY STEP **
            # Use Gumbel-Softmax to get a differentiable one-hot vector
            differentiable_one_hot = F.gumbel_softmax(predictor_logits, tau=GUMBEL_TAU, hard=True)
            
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
        
        return final_logits

# --- Data Preparation (Same as before) ---
print("## Step 0: Preparing Dataset and Tokenizer ##")
# Using a much smaller portion for testing (first 1000 samples)
full_dataset = load_dataset("roneneldan/TinyStories", split="train[:1000]")
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

print(f"Training on {len(loader)} batches")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # The single forward pass handles everything
        final_logits = model(inputs)
        
        # Calculate loss against the original targets
        # Ensure the output is sliced to match the target length
        loss = criterion(final_logits[:, :targets.shape[1], :].reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # Print loss every 10 batches for monitoring
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

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