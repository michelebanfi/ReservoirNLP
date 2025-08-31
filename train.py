from __future__ import annotations
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import ESNLanguageModel
from config import TrainConfig

def evaluate(model: ESNLanguageModel, loader: DataLoader, device: str) -> float:
    model.eval()
    loss_total = 0.0
    n = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        # evaluate on a small subset to keep training snappy
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            B, T, C = logits.shape
            loss = criterion(logits.view(B*T, C), y.view(B*T))
            loss_total += loss.item()
            n += 1
            if n >= 10:  # tighter cap for speed; configurable if needed
                break
    model.train()
    return loss_total / max(1, n)


def _build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(model: ESNLanguageModel, train_loader: DataLoader, val_loader: DataLoader, device: str,
          epochs: int = 2, lr: float = 3e-3, eval_interval: int = 200, ckpt_path: str | None = None,
          weight_decay: float = 1e-2, label_smoothing: float = 0.0, patience: int | None = None) -> Dict:
    # speed-friendly matmul defaults
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model.to(device)
    # ensure sparse buffer is on device (no-op for dense)
    if hasattr(model, "W_sparse") and model.W_sparse is not None:
        model.W_sparse = model.W_sparse.to(device)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.startswith('cuda') and torch.cuda.is_available()))

    step = 0
    history = {"train_loss": [], "val_loss": [], "steps": []}

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    scheduler = _build_scheduler(optim, warmup_steps=max(10, total_steps // 20), total_steps=total_steps)
    best_val = float('inf')
    bad = 0

    for ep in range(epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.cuda.amp.autocast(enabled=(device.startswith('cuda') and torch.cuda.is_available())):
                logits = model(x)
                B, T, C = logits.shape
                loss = criterion(logits.view(B*T, C), y.view(B*T))
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            step += 1
            if step % 20 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")

            if step % eval_interval == 0:
                val = evaluate(model, val_loader, device)
                history["train_loss"].append(loss.item())
                history["val_loss"].append(val)
                history["steps"].append(step)
                pbar.write(f"step {step}: train {loss.item():.3f}, val {val:.3f}")
                if ckpt_path and val < best_val:
                    torch.save(model.state_dict(), ckpt_path)
                # early stopping logic
                if val < best_val - 1e-4:
                    best_val = val
                    bad = 0
                else:
                    bad += 1
                    if patience is not None and bad >= patience:
                        pbar.write("Early stopping: no improvement")
                        return history
    # Ensure we have a terminal eval in history
    try:
        val = evaluate(model, val_loader, device)
        history["val_loss"].append(val)
        if len(history["train_loss"]) == 0:
            history["train_loss"].append(float('nan'))
        last_step = history["steps"][-1] if history["steps"] else step
        history["steps"].append(last_step)
    except Exception:
        pass
    return history
