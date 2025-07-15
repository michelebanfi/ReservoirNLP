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
    num_batches = len(data_handler._get_split_indices(split)) // (batch_size * data_handler.block_size)
    if num_batches == 0: return float('inf')

    for _ in range(num_batches): # Simplified loop for demonstration
        x, y = data_handler.get_batch(split)
        x, y = x.to(device), y.to(device)

        if is_training:
            optimizer.zero_grad()
            logits = model(x)
            # Reshape for CrossEntropyLoss: (N, C) where N is total tokens
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

if __name__ == '__main__':
    # --- Configuration ---
    config = {
        'max_words': 50000,
        'vocab_size': 1000,
        'embedding_dim': 512,
        'num_blocks': 2,  # Number of repeated Reservoir Blocks
        'reservoirs_per_block': [
            {'name': 'short', 'window_size': 5, 'reservoir_size': 256, 'leaking_rate': 0.3, 'spectral_radius': 0.9},
            {'name': 'long',  'window_size': 10, 'reservoir_size': 256, 'leaking_rate': 0.1, 'spectral_radius': 0.9},
            {'name': 'short', 'window_size': 7, 'reservoir_size': 256, 'leaking_rate': 0.3, 'spectral_radius': 0.9},
            {'name': 'long',  'window_size': 8, 'reservoir_size': 256, 'leaking_rate': 0.1, 'spectral_radius': 0.9},
            {'name': 'fast_dynamics', 'window_size': 6, 'reservoir_size': 256, 'leaking_rate': 0.7, 'spectral_radius': 0.9},
            {'name': 'long_memory', 'window_size': 10, 'reservoir_size': 256, 'leaking_rate': 0.1, 'spectral_radius': 1.1},
            {'name': 'fast_dynamics', 'window_size': 3, 'reservoir_size': 256, 'leaking_rate': 0.7, 'spectral_radius': 0.9},
            {'name': 'long_memory', 'window_size': 15, 'reservoir_size': 256, 'leaking_rate': 0.1, 'spectral_radius': 1.1},
        ],
        'readout_hidden_size': 128,
        'epochs': 20,
        'batch_size': 32,
        'block_size': 64, # Sequence length per batch
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
    plt.title('Training and Validation Loss over Epochs (Deep Model)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_deep_model.png')
    plt.show()