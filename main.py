import time
import torch
import numpy as np
import torch.nn as nn
import reservoirpy as rpy
import matplotlib.pyplot as plt

from model import DeepReservoirModel
from data_handler import DataHandler

def process_batches(model, data_handler, batch_size, optimizer, criterion, is_training, device):
    """Processes a dataset in batches for one epoch using on-the-fly computation."""
    split = 'train' if is_training else 'val'
    
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    # Use the length of the data tensor directly
    num_batches = len(data_handler._get_split_indices(split)) // (batch_size * data_handler.block_size)
    if num_batches == 0: return float('inf')

    for _ in range(num_batches):
        x, y = data_handler.get_batch(split)
        x, y = x.to(device), y.to(device)

        if is_training:
            optimizer.zero_grad()
            logits = model(x)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), y.view(B * T))
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(x)
                B, T, C = logits.shape
                loss = criterion(logits.view(B * T, C), y.view(B * T))
        
        total_loss += loss.item()
        
    return total_loss / num_batches

# --- generate FUNCTION (CHANGED) ---
def generate(model, data_handler, context_str, max_new_tokens, device):
    """Generates text from the model given a starting context."""
    model.eval()
    
    # Use the handler's encode method directly
    start_indices = data_handler.encode(context_str)
    context = torch.tensor(start_indices, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\n--- Starting Generation from Reservoir Model ---")
    print(f"Prompt: '{context_str}'")
    
    # Generate new tokens autoregressively
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(context)
            logits = logits[:, -1, :] 
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)

    # Use the handler's decode method directly
    token_ids = context.squeeze().tolist()
    generated_text = data_handler.decode(token_ids)
    print("--- Generated Text ---")
    print(generated_text)
    print("------------------------\n")


if __name__ == '__main__':
    # --- Configuration (CHANGED for Tiny Shakespeare) ---
    config = {
        'dataset_name': 'tinyshakespeare', # ADDED: To select the correct dataset
        # 'vocab_size' REMOVED: Determined automatically by the DataHandler
        'embedding_dim': 256,             # REDUCED: More suitable for a smaller vocab
        'num_blocks': 4,
        'reservoirs_per_block': [
            {'name': 'short', 'window_size': 5, 'reservoir_size': 128, 'leaking_rate': 0.3, 'spectral_radius': 0.9},
            {'name': 'long',  'window_size': 10, 'reservoir_size': 256, 'leaking_rate': 0.1, 'spectral_radius': 0.9},
            # Note: Spectral radius > 1.0 can cause instability. Changed 1.1 to 0.95 for stability.
            {'name': 'long_memory', 'window_size': 10, 'reservoir_size': 256, 'leaking_rate': 0.1, 'spectral_radius': 0.95},
            {'name': 'fast_dynamics', 'window_size': 3, 'reservoir_size': 256, 'leaking_rate': 0.7, 'spectral_radius': 0.9},
        ],
        'readout_hidden_size': 128,
        'epochs': 1,
        'batch_size': 64,
        'block_size': 128, 
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    rpy.verbosity(0)
    
    # --- 1. Data Handling ---
    data_handler = DataHandler(config)

    # --- 2. Model Initialization ---
    model = DeepReservoirModel(
        vocab_size=data_handler.get_vocab_size(),
        config=config
    ).to(config['device'])
    
    print(f"Model initialized on {config['device']} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    # --- 3. Training Loop ---
    print("Starting training loop...")
    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        avg_train_loss = process_batches(model, data_handler, config['batch_size'], optimizer, criterion, is_training=True, device=config['device'])
        train_losses.append(avg_train_loss)
        
        avg_val_loss = process_batches(model, data_handler, config['batch_size'], optimizer, criterion, is_training=False, device=config['device'])
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch + 1}/{config['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # --- 4. Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs (Deep Reservoir Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_deep_reservoir_model.png')
    
    # Only try to show the plot if in an interactive environment
    import matplotlib
    if matplotlib.get_backend() != 'agg':
        try:
            plt.show()
        except Exception as e:
            print(f"Note: Could not display plot interactively ({e})")
    
    generate(model, data_handler, context_str="The story begins", max_new_tokens=200, device=config['device'])