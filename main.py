import time
import torch
import numpy as np
import torch.nn as nn
import reservoirpy as rpy
import matplotlib.pyplot as plt

from model import ReadoutNN
from data_handler import DataAndStateGenerator

def process_batches(model, input_data, targets, batch_size, optimizer, criterion, is_training):
    """Processes a dataset in batches for one epoch."""
    if input_data is None or targets is None:
        return float('inf')
        
    epoch_loss = 0.0
    num_batches = 0
    
    if is_training:
        model.train()
    else:
        model.eval()

    # Create a permutation of the data to shuffle batches each epoch
    permutation = torch.randperm(input_data.size(0))

    for i in range(0, input_data.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        batch_input = input_data[indices]
        batch_targets = targets[indices]
        
        if is_training:
            optimizer.zero_grad()
            output = model(batch_input)
            loss = criterion(output, batch_targets)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                output = model(batch_input)
                loss = criterion(output, batch_targets)
        
        epoch_loss += loss.item()
        num_batches += 1
        
    return epoch_loss / num_batches if num_batches > 0 else float('inf')

if __name__ == '__main__':
    # --- Configuration ---
    # You can now dynamically define any number of reservoirs.
    # Each reservoir is a dictionary with its own hyperparameters.
    config = {
        'max_words': 50000,
        'vocab_size': 10000,
        'embedding_dim': 128,
        'reservoirs': [
            {'name': 'short_window', 'window_size': 5, 'reservoir_size': 256, 'leaking_rate': 0.3, 'spectral_radius': 0.9},
            {'name': 'long_window',  'window_size': 8, 'reservoir_size': 256, 'leaking_rate': 0.3, 'spectral_radius': 0.9},
            # Add more reservoirs here to experiment!
            {'name': 'fast_dynamics', 'window_size': 6, 'reservoir_size': 128, 'leaking_rate': 0.7, 'spectral_radius': 0.9},
             {'name': 'long_memory', 'window_size': 10, 'reservoir_size': 128, 'leaking_rate': 0.1, 'spectral_radius': 1.1},
        ],
        'readout_hidden_size': 128,
        'epochs': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
    }

    rpy.verbosity(0)
    
    # --- 1. One-time Data and State Generation ---
    start_time = time.time()
    data_generator = DataAndStateGenerator(config)
    end_time = time.time()
    print(f"\nData and state generation finished in {(end_time - start_time):.2f} seconds.\n")

    # --- 2. Model Initialization ---
    # The input size for the readout is now dynamically calculated.
    total_reservoir_size = sum(res['reservoir_size'] for res in config['reservoirs'])
    
    readout_model = ReadoutNN(
        input_size=total_reservoir_size,
        hidden_size=config['readout_hidden_size'],
        output_size=data_generator.get_vocab_size()
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(readout_model.parameters(), lr=config['learning_rate'])

    # --- 3. Training Loop ---
    print("Starting training loop...")
    train_losses = []
    val_losses = []

    for epoch in range(config['epochs']):
        # Training
        avg_train_loss = process_batches(readout_model, data_generator.train_states, data_generator.train_targets, config['batch_size'], optimizer, criterion, is_training=True)
        train_losses.append(avg_train_loss)
        
        # Validation
        avg_val_loss = process_batches(readout_model, data_generator.val_states, data_generator.val_targets, config['batch_size'], optimizer, criterion, is_training=False)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch + 1}/{config['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # --- 4. Final Evaluation ---
    avg_test_loss = process_batches(readout_model, data_generator.test_states, data_generator.test_targets, config['batch_size'], optimizer, criterion, is_training=False)
    print(f"\nFinal Test Loss: {avg_test_loss:.4f}")
    print(f"Final Test Perplexity: {np.exp(avg_test_loss):.4f}")

    # --- 5. Plotting ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_validation_loss_dynamic.png')
    plt.show()