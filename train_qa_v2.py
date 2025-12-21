"""
ACT-HRM Training V2 - Improved ACT mechanism
Key changes:
1. Better Q-head initialization (neutral, not biased toward halt)
2. Ponder cost to encourage using more steps
3. Proper ACT loss (Graves 2016 style)
4. No warmup freeze - train Q from the start
5. Curriculum: start with easy examples
"""

from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
import math
import psutil
from datetime import datetime
from tqdm import tqdm
import random


class ConfigV2:
    D_MODEL = 512
    N_HEADS = 8
    N_ENCODER_LAYERS = 4
    N_DECODER_LAYERS = 4
    N_HRM_BLOCKS = 3  # Slightly more reasoning blocks
    DROPOUT = 0.1
    
    MAX_ACT_STEPS = 8
    PONDER_COST = 0.01  # Small cost per step to encourage efficiency
    TIME_PENALTY = 0.001  # Penalty for using too many steps
    
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4  # Slightly higher
    WEIGHT_DECAY = 0.01
    EPOCHS = 30  # More epochs
    WARMUP_STEPS = 500
    GRADIENT_CLIP = 1.0
    
    MAX_SRC_LEN = 512
    MAX_TGT_LEN = 64
    TRAIN_SIZE = None
    VAL_SIZE = 1000
    TEST_SIZE = 1000
    NUM_VAL_SAMPLES = 5
    
    TOKENIZER_NAME = "google/flan-t5-base"
    
    RESULTS_DIR = "results"
    MODEL_SAVE_PATH = "models/act_qa_model_v2.pt"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def to_dict(cls):
        result = {}
        for k, v in vars(cls).items():
            if k.startswith('_'):
                continue
            if callable(v) or isinstance(v, classmethod):
                continue
            result[k] = v
        return result


def get_memory_usage():
    mem = {}
    if torch.cuda.is_available():
        mem['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
        mem['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
    mem['cpu_percent'] = psutil.virtual_memory().percent
    mem['cpu_used_gb'] = psutil.virtual_memory().used / 1e9
    return mem


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def save_metrics(metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


class DROPDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_samples=None):
        self.tokenizer = tokenizer
        self.data = load_dataset('ucinlp/drop', split=split)
        
        if max_samples:
            indices = list(range(min(max_samples, len(self.data))))
            self.data = self.data.select(indices)
        
        self.max_src_len = ConfigV2.MAX_SRC_LEN
        self.max_tgt_len = ConfigV2.MAX_TGT_LEN
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        passage = item['passage'][:2000]
        question = item['question']
        answers = item['answers_spans']['spans']
        answer = answers[0] if answers else ""
        
        input_text = f"question: {question} context: {passage}"
        target_text = answer
        
        source = self.tokenizer(
            input_text,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            target_text,
            max_length=self.max_tgt_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Estimate complexity based on passage length and answer type
        complexity = min(len(passage) / 500, 1.0)  # 0-1 scale
        
        return {
            'input_ids': source.input_ids.squeeze(),
            'attention_mask': source.attention_mask.squeeze(),
            'labels': labels,
            'raw_question': question,
            'raw_answer': answer,
            'complexity': complexity
        }


class SQuADDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_samples=None):
        self.tokenizer = tokenizer
        self.data = load_dataset('rajpurkar/squad', split=split)
        
        if max_samples:
            indices = list(range(min(max_samples, len(self.data))))
            self.data = self.data.select(indices)
        
        self.max_src_len = ConfigV2.MAX_SRC_LEN
        self.max_tgt_len = ConfigV2.MAX_TGT_LEN
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context'][:2000]
        question = item['question']
        answers = item['answers']['text']
        answer = answers[0] if answers else ""
        
        input_text = f"question: {question} context: {context}"
        target_text = answer
        
        source = self.tokenizer(
            input_text,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            target_text,
            max_length=self.max_tgt_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        complexity = min(len(context) / 500, 1.0)
        
        return {
            'input_ids': source.input_ids.squeeze(),
            'attention_mask': source.attention_mask.squeeze(),
            'labels': labels,
            'raw_question': question,
            'raw_answer': answer,
            'complexity': complexity
        }


class CombinedQADataset(Dataset):
    def __init__(self, tokenizer, split='train', max_samples=None):
        self.tokenizer = tokenizer
        samples_per_ds = max_samples // 2 if max_samples else None
        
        drop_split = 'train' if split == 'train' else 'validation'
        squad_split = 'train' if split == 'train' else 'validation'
        
        self.drop_data = DROPDataset(tokenizer, drop_split, samples_per_ds)
        self.squad_data = SQuADDataset(tokenizer, squad_split, samples_per_ds)
        self.total_len = len(self.drop_data) + len(self.squad_data)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        if idx < len(self.drop_data):
            return self.drop_data[idx]
        return self.squad_data[idx - len(self.drop_data)]


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HRMBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout
        )
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * 4 * 2 / 3))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, key_padding_mask=None):
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class ACTReasoningCoreV2(nn.Module):
    """
    Improved ACT Reasoning Core with proper halting mechanism
    Based on Graves 2016 "Adaptive Computation Time"
    """
    
    def __init__(self, dim, num_heads, n_blocks=3, dropout=0.1, max_steps=8):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps
        
        # Halting unit - outputs single probability
        # Initialize with slight bias toward continuing (not halting)
        self.halt_unit = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid()
        )
        # Initialize to output ~0.3 initially (continue more often)
        with torch.no_grad():
            self.halt_unit[-2].bias.fill_(-0.5)
        
        # Step embedding
        self.step_embed = nn.Embedding(max_steps + 1, dim)
        
        # Reasoning blocks with depth scaling
        self.blocks = nn.ModuleList([
            HRMBlock(dim, num_heads, dropout) for _ in range(n_blocks)
        ])
        
        # Cross-attention for state-input mixing
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.cross_norm = nn.LayerNorm(dim)
        
        # State transformation
        self.state_proj = nn.Linear(dim, dim)
        self.output_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        
        # Learnable initialization
        self.state_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, ConfigV2.MAX_SRC_LEN, dim) * 0.02)
    
    def forward_step(self, z_state, x_input, mask=None, step=0):
        B, L, D = z_state.shape
        padding_mask = (mask == 0) if mask is not None else None
        
        # Add step embedding
        step_emb = self.step_embed(torch.tensor(step, device=z_state.device))
        z_state = z_state + step_emb.unsqueeze(0).unsqueeze(0)
        
        # Cross-attention: state attends to input
        cross_out, _ = self.cross_attn(z_state, x_input, x_input, key_padding_mask=padding_mask)
        z_fused = self.cross_norm(z_state + cross_out)
        
        # Self-attention reasoning
        for block in self.blocks:
            z_fused = block(z_fused, key_padding_mask=padding_mask)
        
        return z_fused
    
    def get_halt_prob(self, z_state):
        """Get halting probability from [CLS]-like first token"""
        return self.halt_unit(z_state[:, 0, :]).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1), :].unsqueeze(0))


class NanoACTv2(nn.Module):
    """
    Improved ACT model with proper adaptive computation
    """
    
    def __init__(self, tokenizer, config=ConfigV2):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        vocab_size = tokenizer.vocab_size
        
        d_model = config.D_MODEL
        n_heads = config.N_HEADS
        dropout = config.DROPOUT
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=config.MAX_SRC_LEN, dropout=dropout)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.N_ENCODER_LAYERS,
            enable_nested_tensor=False
        )
        
        # ACT Reasoning Core
        self.act_core = ACTReasoningCoreV2(
            dim=d_model, 
            num_heads=n_heads, 
            n_blocks=config.N_HRM_BLOCKS,
            dropout=dropout,
            max_steps=config.MAX_ACT_STEPS
        )
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
            dropout=dropout
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=config.N_DECODER_LAYERS
        )
        
        # Output head
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.embedding.weight  # Tie embeddings
        
        self.config = config
        self.halt_threshold = 1.0 - 1e-3  # Cumulative halt threshold
    
    def forward(self, input_ids, attention_mask=None, labels=None, return_act_stats=False):
        device = input_ids.device
        B = input_ids.size(0)
        
        # Encode input
        src_emb = self.pos_encoder(self.embedding(input_ids))
        src_padding_mask = (input_ids == self.pad_token_id)
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        # ACT Loop with proper halting
        L, D = memory.shape[1], memory.shape[2]
        z = self.act_core.state_init.expand(B, L, D)
        memory_pos = memory + self.act_core.pos_embed[:, :L, :]
        
        # Track ACT variables
        halting_prob = torch.zeros(B, device=device)
        remainders = torch.zeros(B, device=device)
        n_updates = torch.zeros(B, device=device)
        
        # Accumulate weighted states
        accumulated_state = torch.zeros(B, L, D, device=device)
        
        # Track per-step info for loss
        step_halt_probs = []
        step_states = []
        
        for step in range(self.config.MAX_ACT_STEPS):
            # Forward step
            z = self.act_core.forward_step(z, memory_pos, mask=(~src_padding_mask).long(), step=step)
            
            # Get halt probability
            p = self.act_core.get_halt_prob(z)  # [B]
            step_halt_probs.append(p)
            
            # Determine which samples are still running
            still_running = (halting_prob < self.halt_threshold).float()
            
            # Compute new halting probability
            new_halted = (halting_prob + p * still_running > self.halt_threshold).float() * still_running
            still_running_new = still_running - new_halted
            
            # Update remainders for newly halted
            remainders = remainders + new_halted * (1 - halting_prob)
            
            # Update halting probability
            halting_prob = halting_prob + p * still_running
            
            # Compute weights for this step
            update_weights = p * still_running + new_halted * remainders
            
            # Accumulate weighted state
            accumulated_state = accumulated_state + update_weights.view(B, 1, 1) * z
            
            # Track updates
            n_updates = n_updates + still_running
            
            step_states.append(z.clone())
            
            # Check if all samples have halted
            if (halting_prob >= self.halt_threshold).all():
                break
        
        # Handle samples that never halted
        not_halted = (halting_prob < self.halt_threshold).float()
        remainders = remainders + not_halted * (1 - halting_prob)
        accumulated_state = accumulated_state + remainders.view(B, 1, 1) * z
        n_updates = n_updates + not_halted
        
        # Final output
        z_out = self.act_core.output_proj(accumulated_state)
        enhanced_memory = memory + z_out
        
        result = {
            'enhanced_memory': enhanced_memory,
            'n_updates': n_updates,
            'remainders': remainders,
            'step_halt_probs': step_halt_probs,
            'step_states': step_states,
        }
        
        # Compute loss if labels provided
        if labels is not None:
            # Prepare decoder input (shift right)
            decoder_input = labels.clone()
            decoder_input[decoder_input == -100] = self.pad_token_id
            
            sos_token = torch.zeros((B, 1), device=device, dtype=torch.long)
            decoder_input = torch.cat([sos_token, decoder_input[:, :-1]], dim=1)
            
            tgt_emb = self.pos_encoder(self.embedding(decoder_input))
            tgt_padding_mask = (decoder_input == self.pad_token_id)
            tgt_len = decoder_input.size(1)
            tgt_causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device) * float('-inf'), 
                diagonal=1
            )
            
            # Decode
            dec_out = self.decoder(
                tgt=tgt_emb,
                memory=enhanced_memory,
                tgt_mask=tgt_causal_mask,
                tgt_key_padding_mask=tgt_padding_mask,
                memory_key_padding_mask=src_padding_mask
            )
            
            logits = self.lm_head(dec_out)
            
            # Language modeling loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Ponder cost (ACT regularization) - penalize using more steps
            ponder_cost = n_updates.mean() * self.config.PONDER_COST
            
            # Remainder cost - encourage clean halting
            remainder_cost = remainders.mean() * self.config.TIME_PENALTY
            
            # Total loss
            total_loss = lm_loss + ponder_cost + remainder_cost
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct = (preds == labels) & mask
            seq_correct = (correct.sum(dim=-1) == mask.sum(dim=-1)).float()
            
            result['lm_loss'] = lm_loss
            result['ponder_cost'] = ponder_cost
            result['remainder_cost'] = remainder_cost
            result['total_loss'] = total_loss
            result['logits'] = logits
            result['is_correct'] = seq_correct
        
        return result
    
    @torch.no_grad()
    def generate(self, input_text, max_len=64):
        """Greedy decoding with proper ACT"""
        self.eval()
        device = next(self.parameters()).device
        
        inputs = self.tokenizer(
            input_text, 
            return_tensors='pt',
            max_length=self.config.MAX_SRC_LEN,
            truncation=True
        ).to(device)
        input_ids = inputs.input_ids
        
        # Run forward to get enhanced memory
        result = self.forward(input_ids)
        enhanced_memory = result['enhanced_memory']
        n_steps = result['n_updates'].item()
        
        src_padding_mask = (input_ids == self.pad_token_id)
        
        # Greedy decode
        curr_tokens = torch.zeros((1, 1), device=device, dtype=torch.long)
        
        for _ in range(max_len):
            tgt_emb = self.pos_encoder(self.embedding(curr_tokens))
            tgt_len = curr_tokens.size(1)
            tgt_causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool), 
                diagonal=1
            )
            
            dec_out = self.decoder(
                tgt=tgt_emb,
                memory=enhanced_memory,
                tgt_mask=tgt_causal_mask,
                memory_key_padding_mask=src_padding_mask
            )
            
            logits = self.lm_head(dec_out[:, -1, :])
            next_token = logits.argmax(dim=-1).unsqueeze(0)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
        
        generated_text = self.tokenizer.decode(curr_tokens[0], skip_special_tokens=True)
        return generated_text, n_steps


def evaluate(model, dataloader, device, num_samples=5):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_steps = 0
    samples = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            result = model(input_ids, attention_mask, labels=labels)
            
            total_loss += result['lm_loss'].item()
            total_correct += result['is_correct'].sum().item()
            total_samples += len(input_ids)
            total_steps += result['n_updates'].sum().item()
            
            if len(samples) < num_samples:
                for i in range(min(num_samples - len(samples), len(input_ids))):
                    pred_ids = result['logits'][i].argmax(dim=-1)
                    # Truncate at first EOS or PAD token
                    eos_id = model.tokenizer.eos_token_id
                    pad_id = model.tokenizer.pad_token_id
                    truncate_idx = len(pred_ids)
                    for j, tok in enumerate(pred_ids):
                        if tok.item() in (eos_id, pad_id, 0):
                            truncate_idx = j
                            break
                    pred_ids = pred_ids[:truncate_idx]
                    pred_text = model.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    samples.append({
                        'question': batch['raw_question'][i],
                        'expected': batch['raw_answer'][i],
                        'predicted': pred_text,
                        'steps': result['n_updates'][i].item()
                    })
    
    avg_loss = total_loss / len(dataloader) if dataloader else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    avg_steps = total_steps / total_samples if total_samples > 0 else 0
    
    model.train()
    return avg_loss, accuracy, avg_steps, samples


def train(config=ConfigV2):
    print("=" * 60)
    print("ACT-HRM Training V2 - Improved ACT Mechanism")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Config: {config.to_dict()}")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    print(f"   Tokenizer: {config.TOKENIZER_NAME}")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    
    print("\nBuilding model...")
    model = NanoACTv2(tokenizer, config).to(config.DEVICE)
    params = count_parameters(model)
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    
    print("\nLoading datasets...")
    train_dataset = CombinedQADataset(tokenizer, 'train', config.TRAIN_SIZE)
    val_dataset = CombinedQADataset(tokenizer, 'validation', config.VAL_SIZE)
    test_dataset = CombinedQADataset(tokenizer, 'validation', config.TEST_SIZE)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Single optimizer with unified LR
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=0.1
    )
    
    all_metrics = {
        'config': config.to_dict(),
        'params': params,
        'train': [],
        'validation': [],
        'test': None
    }
    
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    print("-" * 60)
    
    for epoch in range(config.EPOCHS):
        model.train()
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'lm_loss': 0,
            'ponder_cost': 0,
            'avg_steps': 0,
            'memory': {}
        }
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", ncols=120)
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            result = model(input_ids, attention_mask, labels=labels)
            
            loss = result['total_loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            
            epoch_metrics['lm_loss'] += result['lm_loss'].item()
            epoch_metrics['ponder_cost'] += result['ponder_cost'].item()
            epoch_metrics['avg_steps'] += result['n_updates'].mean().item()
            
            pbar.set_postfix({
                'lm_loss': f"{result['lm_loss'].item():.3f}",
                'steps': f"{result['n_updates'].mean().item():.1f}"
            })
        
        # Normalize
        n_batches = len(train_loader)
        epoch_metrics['lm_loss'] /= n_batches
        epoch_metrics['ponder_cost'] /= n_batches
        epoch_metrics['avg_steps'] /= n_batches
        epoch_metrics['memory'] = get_memory_usage()
        
        all_metrics['train'].append(epoch_metrics)
        
        # Validation
        val_loss, val_acc, val_steps, val_samples = evaluate(model, val_loader, config.DEVICE, config.NUM_VAL_SAMPLES)
        
        all_metrics['validation'].append({
            'epoch': epoch + 1,
            'loss': val_loss,
            'accuracy': val_acc,
            'avg_steps': val_steps
        })
        
        print(f"\nEpoch {epoch+1}: Train LM Loss={epoch_metrics['lm_loss']:.4f}, "
              f"Avg Steps={epoch_metrics['avg_steps']:.2f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}, Val Steps={val_steps:.2f}")
        
        # Print sample outputs
        print("\nSample outputs:")
        for i, s in enumerate(val_samples[:3]):
            print(f"  Q: {s['question'][:80]}...")
            print(f"  Expected: {s['expected'][:50]}")
            print(f"  Predicted: {s['predicted'][:50]} (steps: {s['steps']:.1f})")
            print()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.to_dict()
            }, config.MODEL_SAVE_PATH)
            print(f"   Saved best model (val_loss={val_loss:.4f})")
    
    # Final test
    print("\n" + "=" * 60)
    print("Final Test Evaluation")
    print("=" * 60)
    
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_steps, test_samples = evaluate(model, test_loader, config.DEVICE, num_samples=10)
    
    all_metrics['test'] = {
        'loss': test_loss,
        'accuracy': test_acc,
        'avg_steps': test_steps
    }
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Avg Steps: {test_steps:.2f}")
    
    print("\nTest Sample Outputs:")
    for i, s in enumerate(test_samples[:5]):
        print(f"  Q: {s['question'][:80]}...")
        print(f"  Expected: {s['expected']}")
        print(f"  Predicted: {s['predicted']} (steps: {s['steps']:.1f})")
        print()
    
    # Save metrics
    save_metrics(all_metrics, f"{config.RESULTS_DIR}/metrics_v2_{timestamp}.json")
    save_metrics(all_metrics, f"{config.RESULTS_DIR}/metrics_v2_latest.json")
    
    print(f"\nTraining complete!")
    print(f"   Results saved to: {config.RESULTS_DIR}/metrics_v2_{timestamp}.json")
    print(f"   Model saved to: {config.MODEL_SAVE_PATH}")
    
    return model, all_metrics


if __name__ == "__main__":
    model, metrics = train()
