
import os
import time
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

from transformers import AutoTokenizer
from tqdm import tqdm

# Import your existing modules
# We assume the script is run from the root, so src. is accessible
from src.config import Config
from src.model import NanoHRMv3
from src.dataset import UnifiedQADataset, get_dataloaders
from src.train import compute_loss, train_step

def _mp_fn(rank, flags):
    """
    Function meant to be spawned on each TPU core.
    """
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # 1. Setup Device
    device = xm.xla_device()
    xm.master_print(f"Process {rank} utilizing device {device}")
    
    # Debug: Check total available devices
    devices = xm.get_xla_supported_devices()
    xm.master_print(f"Total XLA Devices Detected: {len(devices)}")
    xm.master_print(f"Devices: {devices}")
    
    # 2. Config Override for TPU
    class TPUConfig(Config):
        DEVICE = device
        # Ensure batch size is per-core (e.g. 16 per core * 8 cores = 128 global batch size)
        # Adjust if needed. standard batch size 16 per core is usually fine for TPU.
        # Ensure we use a compatible scheduler/optimizer setup if needed.
    
    cfg = TPUConfig()
    
    # 3. Load Tokenizer & Model
    # Only download on master, others wait
    if xm.is_master_ordinal():
        tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
        # Pre-download model to cache
        NanoHRMv3(tokenizer, cfg)
        xm.master_print("Model and Tokenizer loaded/cached on master.")
    
    # Reduce barrier to ensure download is done
    xm.rendezvous('init_download_done')
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    model = NanoHRMv3(tokenizer, cfg).to(device)
    
    # 4. Data Loading with Distributed Sampler
    # We need to reconstruct get_dataloaders logic to insert DistributedSampler
    
    # --- Custom Data setup for Distributed ---
    train_datasets = []
    val_datasets = []
    
    # We'll use a smaller subset or full set defined in Config
    for dataset_name in cfg.DATASETS:
        # Train
        train_ds = UnifiedQADataset(tokenizer, dataset_name, 'train', cfg.SAMPLES_PER_DATASET)
        if len(train_ds) > 0: train_datasets.append(train_ds)
        
        # Val
        val_ds = UnifiedQADataset(tokenizer, dataset_name, 'validation', min(cfg.SAMPLES_PER_DATASET // 10, 500))
        if len(train_ds) > 0: val_datasets.append(val_ds)
            
    if not train_datasets:
        xm.master_print("No training data found!")
        return

    full_train_dataset = torch.utils.data.ConcatDataset(train_datasets)
    full_val_dataset = torch.utils.data.ConcatDataset(val_datasets) if val_datasets else None
    
    # Samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        full_train_dataset,
        num_replicas=xr.world_size(),
        rank=xr.global_ordinal(),
        shuffle=True
    )
    
    train_loader = torch.utils.data.DataLoader(
        full_train_dataset,
        batch_size=cfg.BATCH_SIZE,
        sampler=train_sampler,
        num_workers=2,
        drop_last=True # TPU prefers fixed shapes
    )
    
    # 5. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # 6. Training Loop
    xm.master_print("Starting TPU Training Loop...")
    
    model.train()
    # Initial freeze (curriculum)
    model.freeze_t5()
    
    for epoch in range(cfg.EPOCHS):
        # Curriculum unfreeze
        if epoch == 2:
            model.unfreeze_t5()
            xm.master_print(">>> Unfreezing T5 Backbone <<<")
            
        # Parallel Loader wrapper for efficient data transfer
        para_loader = pl.ParallelLoader(train_loader, [device])
        
        # Track metrics
        epoch_loss = 0.0
        num_steps = 0
        
        # Progress bar only on master
        if xm.is_master_ordinal():
            pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}")
            
        for batch in para_loader.per_device_loader(device):
            optimizer.zero_grad()
            
            # --- Reusing existing train_step logic ---
            # Ideally we'd call `train_step` but it has `loss.backward()` and `optimizer.step()` 
            # embedded which we need to override/wrap for XLA.
            
            # Since train_step in src/train.py does .backward() and .step(), 
            # we need to inline the logic or modify train_step.
            # INLINE adaptation for XLA safety:
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Encode
            memory, src_mask = model.encode(input_ids)
            B, L, D = memory.shape
            zH, zL = model.hrm_core.init_state(B, L, device)
            
            # ACT Logic (Simplified for brevity, matching train_step)
            M_max = cfg.MAX_SEGMENTS
            epsilon = cfg.ACT_EPSILON
            cumulative_halt = torch.zeros(B, device=device)
            halting_weights = []
            zH_states = []
            steps_taken = torch.zeros(B, device=device)
            
            for m in range(M_max):
                # Detach for TBPTT approximation if needed, usually we keep graph
                # src/train.py does zH.detach() except for m=0
                if m > 0:
                    zH_in = zH.detach()
                    zL_in = zL.detach()
                    curr_mem = memory.detach()
                else:
                    zH_in, zL_in, curr_mem = zH, zL, memory
                    
                zH, zL = model.hrm_core.forward_segment(zH_in, zL_in, curr_mem, key_padding_mask=src_mask)
                p_halt = model.hrm_core.get_q_values(zH).squeeze(-1)
                
                remainder = 1.0 - cumulative_halt
                active_mask = (remainder > epsilon).float()
                effective_halt = torch.min(p_halt, remainder) * active_mask
                
                cumulative_halt = cumulative_halt + effective_halt
                steps_taken = steps_taken + active_mask
                
                halting_weights.append(effective_halt)
                zH_states.append(zH)
                
                if (remainder <= epsilon).all():
                    break
            
            # Combine
            weight_stack = torch.stack(halting_weights, dim=0)
            weight_sum = weight_stack.sum(dim=0, keepdim=True).clamp(min=1e-8)
            norm_weights = weight_stack / weight_sum
            
            final_zH = torch.zeros_like(zH_states[0])
            for w, s in zip(norm_weights, zH_states):
                final_zH = final_zH + w.unsqueeze(-1).unsqueeze(-1) * s
                
            enhanced_mem, enhanced_mask = model.prepare_enhanced_memory(memory, final_zH, src_mask)
            logits = model.decode(enhanced_mem, labels, enhanced_mask)
            
            # Loss
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
            lm_loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            ponder_cost = steps_taken.mean() + (1.0 - cumulative_halt).mean()
            total_loss = lm_loss + cfg.ACT_PONDER_COST_TAU * ponder_cost
            
            # Backward
            total_loss.backward()
            
            # Clip Grad
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            
            # XLA Optimizer Step
            xm.optimizer_step(optimizer)
            
            # Metric Aggregation
            loss_item = total_loss.item()
            epoch_loss += loss_item
            num_steps += 1
            
            if xm.is_master_ordinal():
                pbar.update(1)
                pbar.set_postfix({'loss': loss_item})
                
        if xm.is_master_ordinal():
            pbar.close()
            avg_loss = epoch_loss / num_steps
            print(f"Epoch {epoch+1} Mean Loss: {avg_loss:.4f}")
            
            # Save Checkpoint
            if (epoch + 1) % 1 == 0:
                xm.save(model.state_dict(), f"model_tpu_epoch_{epoch+1}.pt")
                print("Model saved.")

if __name__ == '__main__':
    # Configures execution of _mp_fn. 
    # nprocs=1 is recommended for Colab PJRT runtime to avoid "Expected 4 worker addresses" errors.
    # We will log available devices inside the function to verify visibility.
    xmp.spawn(_mp_fn, args=({},), nprocs=1, start_method='fork')
