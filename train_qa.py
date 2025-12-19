import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from transformers import AutoTokenizer
import random
import math
import numpy as np
import json
import os
from datetime import datetime
from tqdm import tqdm
import psutil
import gc

class Config:
    D_MODEL = 512
    N_HEADS = 8
    N_ENCODER_LAYERS = 4
    N_DECODER_LAYERS = 4
    N_HRM_BLOCKS = 2
    DROPOUT = 0.1
    

    MAX_ACT_STEPS = 8
    HALT_EXPLORATION = 0.2
    
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    EPOCHS = 20
    WARMUP_STEPS = 1000
    GRADIENT_CLIP = 1.0
    
    MAX_SRC_LEN = 512       # Context + Question
    MAX_TGT_LEN = 64        # Answer
    TRAIN_SIZE = None       # None = use all
    VAL_SIZE = 1000         # Validation subset
    TEST_SIZE = 1000        # Test subset
    NUM_VAL_SAMPLES = 5     # Samples to print each epoch
    
    TOKENIZER_NAME = "google/flan-t5-base"  
    
    RESULTS_DIR = "results"
    MODEL_SAVE_PATH = "models/act_qa_model.pt"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() 
                if not k.startswith('_') and k.isupper()}


def get_memory_usage():
    """Get current GPU and CPU memory usage"""
    mem = {}
    if torch.cuda.is_available():
        mem['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
        mem['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
    mem['cpu_percent'] = psutil.virtual_memory().percent
    mem['cpu_used_gb'] = psutil.virtual_memory().used / 1e9
    return mem

def count_parameters(model):
    """Count trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}

def save_metrics(metrics, filepath):
    """Save metrics to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)

def load_metrics(filepath):
    """Load metrics from JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


class DROPDataset(Dataset):
    """DROP Dataset - Discrete Reasoning Over Paragraphs"""
    
    def __init__(self, tokenizer, split='train', max_samples=None):
        self.tokenizer = tokenizer
        self.data = load_dataset('ucinlp/drop', split=split)
        
        if max_samples:
            indices = list(range(min(max_samples, len(self.data))))
            self.data = self.data.select(indices)
        
        self.max_src_len = Config.MAX_SRC_LEN
        self.max_tgt_len = Config.MAX_TGT_LEN
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format: "question: {question} context: {passage}"
        passage = item['passage'][:2000]  # Truncate very long passages
        question = item['question']
        
        # Get first answer span
        answers = item['answers_spans']['spans']
        answer = answers[0] if answers else ""
        
        input_text = f"question: {question} context: {passage}"
        target_text = answer
        
        # Tokenize
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
        
        # Create labels (replace padding with -100)
        labels = target.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source.input_ids.squeeze(),
            'attention_mask': source.attention_mask.squeeze(),
            'labels': labels,
            'raw_question': question,
            'raw_answer': answer
        }

class SQuADDataset(Dataset):
    """SQuAD Dataset - Reading Comprehension"""
    
    def __init__(self, tokenizer, split='train', max_samples=None):
        self.tokenizer = tokenizer
        self.data = load_dataset('rajpurkar/squad', split=split)
        
        if max_samples:
            indices = list(range(min(max_samples, len(self.data))))
            self.data = self.data.select(indices)
        
        self.max_src_len = Config.MAX_SRC_LEN
        self.max_tgt_len = Config.MAX_TGT_LEN
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        context = item['context'][:2000]
        question = item['question']
        
        # Get first answer
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
        
        return {
            'input_ids': source.input_ids.squeeze(),
            'attention_mask': source.attention_mask.squeeze(),
            'labels': labels,
            'raw_question': question,
            'raw_answer': answer
        }

class CombinedQADataset(Dataset):
    """Combined DROP + SQuAD dataset"""
    
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
    """Heterogeneous Reasoning Module Block"""
    
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
        # Pre-norm architecture
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class IterativeRefinementAdapter(nn.Module):
    """Gated adapter for iterative refinement"""
    
    def __init__(self, dim, max_steps=8):
        super().__init__()
        self.input_proj = nn.Linear(dim, dim)
        self.gate_proj = nn.Linear(dim * 2, dim)
        self.step_embed = nn.Embedding(max_steps + 1, dim)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x, step=0):
        B, L, D = x.shape
        step_emb = self.step_embed(torch.tensor(step, device=x.device))
        step_emb = step_emb.unsqueeze(0).unsqueeze(0).expand(B, L, D)
        
        x_proj = self.input_proj(x)
        gate_input = torch.cat([x, step_emb], dim=-1)
        gate = torch.sigmoid(self.gate_proj(gate_input))
        
        return self.norm(gate * x_proj + (1 - gate) * x)

class ACTReasoningCore(nn.Module):
    """Adaptive Computation Time Reasoning Core"""
    
    def __init__(self, dim, num_heads, n_blocks=2, dropout=0.1):
        super().__init__()
        self.dim = dim
        
        # Adapters
        self.input_adapter = IterativeRefinementAdapter(dim, Config.MAX_ACT_STEPS)
        self.output_adapter = nn.Linear(dim, dim)
        nn.init.zeros_(self.output_adapter.weight)
        nn.init.zeros_(self.output_adapter.bias)
        
        # Q-Head for halting decision
        self.q_head = nn.Linear(dim, 2)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)
        
        # Reasoning blocks
        self.blocks = nn.ModuleList([
            HRMBlock(dim, num_heads, dropout) for _ in range(n_blocks)
        ])
        
        # Cross-attention for state-input mixing
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.cross_norm = nn.LayerNorm(dim)
        
        # Learnable state initialization
        self.state_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, Config.MAX_SRC_LEN, dim) * 0.02)
    
    def forward_step(self, z_state, x_input, mask=None, step=0):
        padding_mask = (mask == 0) if mask is not None else None
        
        # Cross-attention: state attends to input
        cross_out, _ = self.cross_attn(z_state, x_input, x_input, key_padding_mask=padding_mask)
        z_fused = self.cross_norm(z_state + cross_out)
        
        # Self-attention reasoning
        for block in self.blocks:
            z_fused = block(z_fused, key_padding_mask=padding_mask)
        
        return z_fused
    
    def predict_q(self, z_state):
        return self.q_head(z_state[:, 0, :])

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

class NanoACTLarge(nn.Module):
    """Larger ACT model for QA tasks"""
    
    def __init__(self, tokenizer, config=Config):
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
        
        # HRM Reasoning Core
        self.hrm = ACTReasoningCore(
            dim=d_model, 
            num_heads=n_heads, 
            n_blocks=config.N_HRM_BLOCKS,
            dropout=dropout
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
        
        # Tie embeddings
        self.lm_head.weight = self.embedding.weight
        
        self.config = config
    
    def forward(self, input_ids, attention_mask=None, labels=None, epoch=0):
        device = input_ids.device
        
        # Encode
        src_emb = self.pos_encoder(self.embedding(input_ids))
        src_padding_mask = (input_ids == self.pad_token_id)
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        # ACT Reasoning Loop
        B, L, D = memory.shape
        z = self.hrm.state_init.expand(B, L, D)
        memory_pos = memory + self.hrm.pos_embed[:, :L, :]
        
        step_outputs = []
        
        if labels is not None:
            # Prepare decoder input (shift right)
            decoder_input = labels.clone()
            decoder_input[decoder_input == -100] = self.pad_token_id
            
            # Shift right
            sos_token = torch.zeros((B, 1), device=device, dtype=torch.long)
            decoder_input = torch.cat([sos_token, decoder_input[:, :-1]], dim=1)
            
            tgt_emb = self.pos_encoder(self.embedding(decoder_input))
            tgt_padding_mask = (decoder_input == self.pad_token_id)
            tgt_len = decoder_input.size(1)
            tgt_causal_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=device) * float('-inf'), 
                diagonal=1
            )
            
            # ACT Loop
            for step in range(Config.MAX_ACT_STEPS):
                x_in = self.hrm.input_adapter(memory_pos, step=step)
                z = self.hrm.forward_step(z, x_in, mask=(~src_padding_mask).long(), step=step)
                q_logits = self.hrm.predict_q(z)
                z_out = self.hrm.output_adapter(z)
                
                # Phase-based residual connection
                if self.training and epoch < 3:
                    enhanced_memory = z_out
                else:
                    enhanced_memory = memory + z_out
                
                # Decode
                dec_out = self.decoder(
                    tgt=tgt_emb,
                    memory=enhanced_memory,
                    tgt_mask=tgt_causal_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
                )
                
                logits = self.lm_head(dec_out)
                
                # Loss
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # Accuracy
                preds = logits.argmax(dim=-1)
                mask = labels != -100
                correct = (preds == labels) & mask
                seq_correct = (correct.sum(dim=-1) == mask.sum(dim=-1)).float()
                
                step_outputs.append({
                    'q_logits': q_logits,
                    'lm_loss': lm_loss,
                    'is_correct': seq_correct,
                    'z_final': enhanced_memory,
                    'logits': logits
                })
        
        return step_outputs
    
    @torch.no_grad()
    def generate(self, input_text, max_len=64):
        """Greedy decoding for inference"""
        self.eval()
        device = next(self.parameters()).device
        
        inputs = self.tokenizer(
            input_text, 
            return_tensors='pt',
            max_length=Config.MAX_SRC_LEN,
            truncation=True
        ).to(device)
        input_ids = inputs.input_ids
        
        # Encode
        src_emb = self.pos_encoder(self.embedding(input_ids))
        src_padding_mask = (input_ids == self.pad_token_id)
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        # ACT Loop
        B, L, D = memory.shape
        z = self.hrm.state_init.expand(B, L, D)
        memory_pos = memory + self.hrm.pos_embed[:, :L, :]
        
        final_step = 0
        for step in range(Config.MAX_ACT_STEPS):
            x_in = self.hrm.input_adapter(memory_pos, step=step)
            z = self.hrm.forward_step(z, x_in, mask=(~src_padding_mask).long(), step=step)
            q_logits = self.hrm.predict_q(z)
            
            if q_logits[0, 0] > q_logits[0, 1]:  # Halt > Continue
                final_step = step + 1
                break
            final_step = step + 1
        
        # Prepare decoder memory
        z_out = self.hrm.output_adapter(z)
        enhanced_memory = memory + z_out
        
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
        return generated_text, final_step

def evaluate(model, dataloader, device, num_samples=5):
    """Evaluate model and optionally print samples"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    samples = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            step_results = model(input_ids, attention_mask, labels=labels, epoch=99)
            
            if step_results:
                final_step = step_results[-1]
                total_loss += final_step['lm_loss'].item()
                total_correct += final_step['is_correct'].sum().item()
                total_samples += len(input_ids)
                
                # Collect samples for printing
                if len(samples) < num_samples:
                    for i in range(min(num_samples - len(samples), len(input_ids))):
                        pred_ids = final_step['logits'][i].argmax(dim=-1)
                        pred_text = model.tokenizer.decode(pred_ids, skip_special_tokens=True)
                        samples.append({
                            'question': batch['raw_question'][i],
                            'expected': batch['raw_answer'][i],
                            'predicted': pred_text
                        })
    
    avg_loss = total_loss / len(dataloader) if dataloader else 0
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    model.train()
    return avg_loss, accuracy, samples

def train(config=Config):
    """Main training loop"""
    
    print("=" * 60)
    print("ACT-HRM-Sandwich Training on DROP + SQuAD")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Config: {config.to_dict()}")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    print(f"   Tokenizer: {config.TOKENIZER_NAME}")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    
    print("\n Building model...")
    model = NanoACTLarge(tokenizer, config).to(config.DEVICE)
    params = count_parameters(model)
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    print(f"   Model size: {params['total'] * 4 / 1e9:.2f} GB (fp32)")
    
    print("\n Loading datasets...")
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
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False,
        num_workers=2
    )
    
    optimizer = torch.optim.AdamW([
        {'params': model.hrm.q_head.parameters(), 'lr': 5e-4},
        {'params': [p for n, p in model.hrm.named_parameters() if 'q_head' not in n], 'lr': config.LEARNING_RATE},
        {'params': [p for n, p in model.named_parameters() if 'hrm' not in n], 'lr': config.LEARNING_RATE}
    ], weight_decay=config.WEIGHT_DECAY)
    
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
    
    print("\n Starting training...")
    print("-" * 60)
    
    for epoch in range(config.EPOCHS):
        model.train()
        is_warmup = epoch < 5
        q_loss_weight = 0.0 if is_warmup else 1.0
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'lm_loss': 0,
            'q_loss': 0,
            'thought_drift': 0,
            'memory': {}
        }
        
        pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{config.EPOCHS}",
            ncols=120
        )
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(config.DEVICE)
            attention_mask = batch['attention_mask'].to(config.DEVICE)
            labels = batch['labels'].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            step_results = model(input_ids, attention_mask, labels=labels, epoch=epoch)
            
            if not step_results:
                continue
            
            final_step = step_results[-1]
            lm_loss = final_step['lm_loss']
            
            q_losses = []
            if not is_warmup:
                next_value = step_results[-1]['is_correct']
                GAMMA = 0.9
                
                for i in reversed(range(len(step_results) - 1)):
                    curr = step_results[i]
                    discounted_future = next_value * GAMMA
                    
                    halt_loss = F.binary_cross_entropy_with_logits(
                        curr['q_logits'][:, 0], curr['is_correct']
                    )
                    cont_loss = F.binary_cross_entropy_with_logits(
                        curr['q_logits'][:, 1], discounted_future
                    )
                    q_losses.append(halt_loss + cont_loss)
                    
                    with torch.no_grad():
                        pred_val = torch.maximum(
                            torch.sigmoid(curr['q_logits'][:, 0]),
                            torch.sigmoid(curr['q_logits'][:, 1])
                        )
                        next_value = pred_val
            
            avg_q_loss = torch.stack(q_losses).mean() if q_losses else torch.tensor(0.0).to(config.DEVICE)
            
            z_drift = sum(
                (res['z_final'] - step_results[0]['z_final']).norm(p=2) 
                for res in step_results
            )
            
            loss = lm_loss + (q_loss_weight * avg_q_loss)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
            optimizer.step()
            scheduler.step()
            
            epoch_metrics['lm_loss'] += lm_loss.item()
            epoch_metrics['q_loss'] += avg_q_loss.item()
            epoch_metrics['thought_drift'] += z_drift.item() / (len(step_results) * config.BATCH_SIZE)
            
            mem = get_memory_usage()
            pbar.set_postfix({
                'LM': f"{lm_loss.item():.4f}",
                'Q': f"{avg_q_loss.item():.4f}",
                'Params': f"{params['trainable']/1e6:.1f}M",
                'GPU': f"{mem.get('gpu_allocated_gb', 0):.1f}GB"
            })
        
        num_batches = len(train_loader)
        epoch_metrics['lm_loss'] /= num_batches
        epoch_metrics['q_loss'] /= num_batches
        epoch_metrics['thought_drift'] /= num_batches
        epoch_metrics['memory'] = get_memory_usage()
        
        all_metrics['train'].append(epoch_metrics)
        
        print(f"\n Validation Epoch {epoch+1}:")
        val_loss, val_acc, val_samples = evaluate(
            model, val_loader, config.DEVICE, num_samples=config.NUM_VAL_SAMPLES
        )
        
        val_metrics = {
            'epoch': epoch + 1,
            'loss': val_loss,
            'accuracy': val_acc
        }
        all_metrics['validation'].append(val_metrics)
        
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"\n   Sample Outputs:")
        print("   " + "-" * 50)
        
        for i, sample in enumerate(val_samples):
            print(f"   [{i+1}] Q: {sample['question'][:80]}...")
            print(f"       Expected: {sample['expected'][:50]}")
            print(f"       Predicted: {sample['predicted'][:50]}")
            print()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.to_dict()
            }, config.MODEL_SAVE_PATH)
            print(f"Saved best model (val_loss: {val_loss:.4f})")
        
        save_metrics(all_metrics, f"{config.RESULTS_DIR}/metrics_{timestamp}.json")
        print("-" * 60)
    
    print("\n Final Test Evaluation:")
    print("=" * 60)
    
    checkpoint = torch.load(config.MODEL_SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_samples = evaluate(
        model, test_loader, config.DEVICE, num_samples=10
    )
    
    all_metrics['test'] = {
        'loss': test_loss,
        'accuracy': test_acc
    }
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"\nTest Sample Outputs:")
    print("-" * 50)
    
    for i, sample in enumerate(test_samples):
        print(f"[{i+1}] Q: {sample['question'][:80]}...")
        print(f"    Expected: {sample['expected']}")
        print(f"    Predicted: {sample['predicted']}")
        print()
    
    save_metrics(all_metrics, f"{config.RESULTS_DIR}/metrics_{timestamp}.json")
    save_metrics(all_metrics, f"{config.RESULTS_DIR}/metrics_latest.json")
    
    print(f"\n Training complete!")
    print(f"   Results saved to: {config.RESULTS_DIR}/metrics_{timestamp}.json")
    print(f"   Model saved to: {config.MODEL_SAVE_PATH}")
    
    return model, all_metrics


if __name__ == "__main__":

    model, metrics = train()
