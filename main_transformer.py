import time
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

# Note: We import the new TransformerModel
from model_transformer import TransformerModel
from data_handler import DataHandler

def process_batches(model, data_handler, batch_size, optimizer, criterion, is_training, device):
    """Processes a dataset in batches for one epoch using on-the-fly computation."""
    split = 'train' if is_training else 'val'
    
    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0
    # Calculate the number of batches based on total tokens and sequence length
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
            # Clip gradients to prevent exploding gradients, a common practice for Transformers
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(x)
                B, T, C = logits.shape
                loss = criterion(logits.view(B * T, C), y.view(B * T))
        
        total_loss += loss.item()
        
    return total_loss / num_batches

def generate(model, data_handler, context_str, max_new_tokens, device):
    """Generates text from the model given a starting context."""
    model.eval()
    
    # OLD way: tokenizer = data_handler.tokenizer
    # OLD way: start_indices = tokenizer.encode(context_str).ids
    # NEW way: Use the handler's encode method directly
    start_indices = data_handler.encode(context_str)
    context = torch.tensor(start_indices, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\n--- Starting Generation from Transformer Model ---")
    print(f"Prompt: '{context_str}'")
    
    # Generate new tokens autoregressively
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context_cond = context if context.size(1) <= config['block_size'] else context[:, -config['block_size']:]
            logits = model(context_cond)
            logits = logits[:, -1, :]
            probs = torch.nn.functional.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, idx_next), dim=1)

    # OLD way: generated_text = tokenizer.decode(token_ids)
    # NEW way: Use the handler's decode method directly
    token_ids = context.squeeze().tolist()
    generated_text = data_handler.decode(token_ids)
    
    print("--- Generated Text ---")
    print(generated_text)
    print("------------------------\n")

if __name__ == '__main__':
    # --- Configuration for the Transformer Baseline ---
    # These parameters are chosen to be comparable to the reservoir model's complexity
    config = {
        'dataset_name': 'tinyshakespeare', 
        'embedding_dim': 512,   # d_model in Transformer terminology
        'num_blocks': 2,        # Number of Transformer Encoder Layers
        'num_heads': 2,         # Number of attention headss
        'ff_dim': 128,         # Dimension of the feed-forward network (often 4*d_model)
        'dropout': 0.1,
        'epochs': 10,
        'batch_size': 32,
        'block_size': 64,       # Sequence length per batch
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # --- 1. Data Handling (reusing the same DataHandler) ---
    data_handler = DataHandler(config)

    # --- 2. Model Initialization ---
    model = TransformerModel(
        vocab_size=data_handler.get_vocab_size(),
        config=config
    ).to(config['device'])
    
    print(f"Transformer Model initialized on {config['device']} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Adding Cosine Annealing Learning Rate Scheduler
    # T_max is set to the number of epochs, eta_min is the minimum learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'], 
        eta_min=config['learning_rate'] * 0.01  # Minimum LR will be 1% of initial LR
    )

    # --- 3. Training Loop ---
    print("Starting Transformer training loop...")
    train_losses = []
    val_losses = []
    learning_rates = []  # Track learning rates for plotting

    for epoch in range(config['epochs']):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        avg_train_loss = process_batches(model, data_handler, config['batch_size'], optimizer, criterion, is_training=True, device=config['device'])
        train_losses.append(avg_train_loss)
        
        avg_val_loss = process_batches(model, data_handler, config['batch_size'], optimizer, criterion, is_training=False, device=config['device'])
        val_losses.append(avg_val_loss)
        
        # Step the scheduler after each epoch
        if scheduler is not None:
            scheduler.step()
        
        print(f"Epoch {epoch + 1}/{config['epochs']}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

    # --- 4. Plotting ---
    # Plot 1: Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs (Transformer Baseline)')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_transformer_baseline.png')
    
    # Plot 2: Learning Rate Schedule
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config['epochs'] + 1), learning_rates, 'o-')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Annealing Learning Rate Schedule')
    plt.grid(True)
    plt.savefig('transformer_lr_schedule.png')
    
    # Only try to show the plot if in an interactive environment
    import matplotlib
    if matplotlib.get_backend() != 'agg':
        try:
            plt.show()
        except Exception as e:
            print(f"Note: Could not display plot interactively ({e})")
    
    generate(model, data_handler, context_str="The story begins", max_new_tokens=100, device=config['device'])