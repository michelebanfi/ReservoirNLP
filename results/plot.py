import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_metrics(filepath):
    """Plot training and validation metrics from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    epochs = [m["epoch"] for m in data["train"]]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # LM Loss
    lm_loss = [m["lm_loss"] for m in data["train"]]
    axes[0, 0].plot(epochs, lm_loss, marker='o')
    axes[0, 0].set_title("Language Model Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].grid(True)
    
    # Q Loss
    q_loss = [m["q_loss"] for m in data["train"]]
    axes[0, 1].plot(epochs, q_loss, marker='o', color='orange')
    axes[0, 1].set_title("Q Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].grid(True)
    
    # Validation Loss & Accuracy
    val_epochs = [m["epoch"] for m in data["validation"]]
    val_loss = [m["loss"] for m in data["validation"]]
    val_acc = [m["accuracy"] for m in data["validation"]]
    
    axes[1, 0].plot(val_epochs, val_loss, marker='s', color='green')
    axes[1, 0].set_title("Validation Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(val_epochs, val_acc, marker='s', color='red')
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(Path(filepath).parent / "metrics_plot.png", dpi=100)
    plt.show()

if __name__ == "__main__":
    plot_metrics("results/metrics_20251219_103004.json")