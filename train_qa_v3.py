from torch.utils.data import Dataset, DataLoader
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


class ConfigV3:
    D_MODEL = 512
    N_HEADS = 8
    
    # HRM Architecture (from paper)
    N_HIGH_CYCLES = 2      # N: number of H-module updates per segment
    N_LOW_STEPS = 4        # T: number of L-module updates per H-cycle
    N_HRM_LAYERS = 2       # Transformer layers in each module
    
    # Deep supervision
    MAX_SEGMENTS = 8       # M_max: maximum supervision segments
    MIN_SEGMENTS_PROB = 0.1  # ε: probability of sampling M_min > 1
    
    # Training
    DROPOUT = 0.1
    BATCH_SIZE = 16        # Smaller for deep supervision memory
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1.0     # High weight decay (from HRM repo)
    EPOCHS = 10
    THINKING_WARMUP_EPOCHS = 5  # Force thinking for first 5 epochs
    WARMUP_STEPS = 1000
    GRADIENT_CLIP = 1.0
    
    # ALWAYS Train from scratch (Pretrained removed per instruction)
    MAX_SRC_LEN = 256      # Shorter for memory efficiency
    MAX_TGT_LEN = 32
    TRAIN_SIZE = 10000     # Start smaller
    VAL_SIZE = 1000
    TEST_SIZE = 1000
    NUM_VAL_SAMPLES = 5
    
    TOKENIZER_NAME = "google/flan-t5-base"
    
    RESULTS_DIR = "results"
    MODEL_SAVE_PATH = "models/act_qa_model_v3.pt"
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() 
                if not k.startswith('_') and not isinstance(v, classmethod) and not callable(v)}


def get_memory_usage():
    mem = {}
    if torch.cuda.is_available():
        mem['gpu_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
        mem['gpu_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
    mem['cpu_percent'] = psutil.virtual_memory().percent
    return mem


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable}


def save_metrics(metrics, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics, f, indent=2)


class SQuADDataset(Dataset):
    """SQuAD is better for initial testing than DROP (simpler answers)"""
    def __init__(self, tokenizer, split='train', max_samples=None):
        self.tokenizer = tokenizer
        self.data = load_dataset('rajpurkar/squad', split=split)
        
        if max_samples:
            indices = list(range(min(max_samples, len(self.data))))
            self.data = self.data.select(indices)
        
        self.max_src_len = ConfigV3.MAX_SRC_LEN
        self.max_tgt_len = ConfigV3.MAX_TGT_LEN
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context'][:1500]
        question = item['question']
        answers = item['answers']['text']
        answer = answers[0] if answers else ""
        
        input_text = f"question: {question} context: {context}"
        
        source = self.tokenizer(
            input_text,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            answer,
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
            'raw_answer': answer,
        }


class RMSNorm(nn.Module):
    """RMSNorm without learnable bias (from HRM paper)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SwiGLU(nn.Module):
    """SwiGLU FFN (from Llama/HRM)"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4 * 2 / 3)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GatedFusion(nn.Module):
    """
    Gated interaction for combining inputs.
    Controls how much context flows into the primary stream.
    """
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, context):
        # x: primary stream (e.g., zL)
        # context: context stream (e.g., zH or input)
        # gate = sigmoid(W * [x, context])
        g = self.gate(torch.cat([x, context], dim=-1))
        # out = x + g * proj(context)
        return x + g * self.proj(context)

class HRMTransformerBlock(nn.Module):
    """
    Post-Norm Transformer block for HRM modules.
    Post-norm is used (norm after residual) for stability in recurrent settings.
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout,
            bias=False  # No bias (from HRM)
        )
        self.mlp = SwiGLU(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, key_padding_mask=None):
        # Post-norm architecture
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x


class HRMModule(nn.Module):
    """
    A single HRM module (either H or L).
    Takes multiple inputs and combines them via element-wise addition.
    """
    def __init__(self, dim, num_heads, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            HRMTransformerBlock(dim, num_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Gated fusion for combining inputs
        self.fusion = GatedFusion(dim)
        
    def forward(self, *inputs, key_padding_mask=None):
        # Combine inputs via Gated Fusion (Primary + Contexts)
        # inputs[0] is assumed to be the primary state
        x = inputs[0]
        
        # Fuse other inputs (contexts) into x
        if len(inputs) > 1:
            for context in inputs[1:]:
                x = self.fusion(x, context)
        
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


def stablemax(logits, dim=-1, tau=1.0):
    """
    Stablemax activation (from HRM paper) for better generalization.
    More robust than softmax for small-sample learning.
    """
    # Shift for numerical stability
    logits = logits - logits.max(dim=dim, keepdim=True).values
    exp_logits = torch.exp(logits / tau)
    # Add small constant to prevent division issues
    return exp_logits / (exp_logits.sum(dim=dim, keepdim=True) + 1e-8)


class HierarchicalReasoningCore(nn.Module):
    """
    True HRM Core with H-module and L-module operating at different timescales.
    
    Architecture per segment:
    - N high-level cycles
    - T low-level steps per cycle
    - L-module updates rapidly, H-module updates slowly
    """
    
    def __init__(self, dim, num_heads, config):
        super().__init__()
        self.dim = dim
        self.N = config.N_HIGH_CYCLES  # H-module cycles
        self.T = config.N_LOW_STEPS    # L-module steps per H-cycle
        
        # High-level module (slow, abstract reasoning)
        self.H_module = HRMModule(dim, num_heads, config.N_HRM_LAYERS, config.DROPOUT)
        
        # Low-level module (fast, detailed computation)
        self.L_module = HRMModule(dim, num_heads, config.N_HRM_LAYERS, config.DROPOUT)
        
        # Q-head for ACT halting (outputs [q_halt, q_continue])
        self.q_head = nn.Linear(dim, 2, bias=True)
        # Initialize to favor continuing (prevent early halt collapse)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.data = torch.tensor([-2.0, 0.0])  # Slight bias to continue
        
        # Learnable initial states
        self.zH_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.zL_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    
    def init_state(self, batch_size, seq_len, device):
        """Initialize H and L states"""
        zH = self.zH_init.expand(batch_size, seq_len, -1).clone()
        zL = self.zL_init.expand(batch_size, seq_len, -1).clone()
        return zH.to(device), zL.to(device)
    
    def forward_segment(self, zH, zL, x, key_padding_mask=None):
        """
        One segment of HRM computation (N * T steps total).
        
        Returns final zH, zL after hierarchical processing.
        """
        for _ in range(self.N):
            # T low-level steps (L converges within H-cycle)
            for _ in range(self.T):
                zL = self.L_module(zL, zH, x, key_padding_mask=key_padding_mask)
            
            # 1 high-level step (H updates after L converges)
            zH = self.H_module(zH, zL, key_padding_mask=key_padding_mask)
        
        return zH, zL
    
    def get_q_values(self, zH):
        """Get Q-values for halt decision from H-module state (mean pooled)"""
        pooled = zH.mean(dim=1)  # [B, D]
        q_logits = self.q_head(pooled)  # [B, 2]
        q_values = torch.sigmoid(q_logits)  # Bounded [0, 1]
        return q_values[:, 0], q_values[:, 1]  # q_halt, q_continue


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(1)])


class NanoHRMv3(nn.Module):
    """
    Proper HRM implementation for text QA.
    
    Key features:
    - Optional pre-trained encoder for language understanding
    - True H/L hierarchical reasoning core
    - Deep supervision training with detached states
    - Q-learning for adaptive computation
    """
    
    def __init__(self, tokenizer, config=ConfigV3):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        vocab_size = tokenizer.vocab_size
        d_model = config.D_MODEL
        
        # Encoder: Always from scratch (removed pretrained option)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, config.MAX_SRC_LEN, config.DROPOUT)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=config.N_HEADS,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=config.DROPOUT
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.encoder_proj = nn.Identity()
        
        # HRM Reasoning Core
        self.hrm_core = HierarchicalReasoningCore(d_model, config.N_HEADS, config)
        
        # Decoder
        self.dec_embedding = nn.Embedding(vocab_size, d_model)
        self.dec_pos = PositionalEncoding(d_model, config.MAX_TGT_LEN, config.DROPOUT)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=config.N_HEADS,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=config.DROPOUT
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # Output head (shared with decoder embeddings)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.dec_embedding.weight
    
    def encode(self, input_ids, attention_mask=None):
        """Encode input text to representations"""
        src_emb = self.pos_encoder(self.embedding(input_ids))
        padding_mask = (input_ids == self.pad_token_id)
        memory = self.encoder(src_emb, src_key_padding_mask=padding_mask)
        return memory
    
    def decode(self, memory, labels, src_padding_mask):
        """Decode from enhanced memory to output logits"""
        B = memory.size(0)
        device = memory.device
        
        # Shift labels for teacher forcing
        decoder_input = labels.clone()
        decoder_input[decoder_input == -100] = self.pad_token_id
        
        sos_token = torch.zeros((B, 1), device=device, dtype=torch.long)
        decoder_input = torch.cat([sos_token, decoder_input[:, :-1]], dim=1)
        
        tgt_emb = self.dec_pos(self.dec_embedding(decoder_input))
        tgt_len = decoder_input.size(1)
        tgt_causal_mask = torch.triu(
            torch.ones(tgt_len, tgt_len, device=device) * float('-inf'),
            diagonal=1
        )
        tgt_padding_mask = (decoder_input == self.pad_token_id)
        
        dec_out = self.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        return self.lm_head(dec_out)
    
    def forward_one_segment(self, zH, zL, x, key_padding_mask=None):
        """
        Single HRM segment forward pass.
        Returns: (new_zH, new_zL, q_halt, q_continue)
        """
        zH, zL = self.hrm_core.forward_segment(zH, zL, x, key_padding_mask)
        q_halt, q_continue = self.hrm_core.get_q_values(zH)
        return zH, zL, q_halt, q_continue


def compute_loss(logits, labels, use_stablemax=False):
    """Compute language modeling loss"""
    if use_stablemax:
        # Stablemax cross-entropy
        probs = stablemax(logits, dim=-1)
        B, T, V = logits.shape
        labels_flat = labels.view(-1)
        probs_flat = probs.view(-1, V)
        
        # Mask out -100 labels
        mask = labels_flat != -100
        valid_probs = probs_flat[mask]
        valid_labels = labels_flat[mask]
        
        if valid_labels.numel() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Gather probabilities of correct tokens
        correct_probs = valid_probs.gather(1, valid_labels.unsqueeze(1)).squeeze()
        loss = -torch.log(correct_probs + 1e-8).mean()
    else:
        loss_fn = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
    
    return loss


def train_step_deep_supervision(model, batch, optimizer, config, epoch):
    """
    Deep supervision training step (from HRM paper).
    
    Key insight: Run multiple segments, compute loss after each,
    and DETACH state between segments. This provides:
    - More frequent gradient signal
    - Regularization
    - 1-step gradient approximation
    """
    device = config.DEVICE
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    B, L = input_ids.shape
    
    # Encode input
    # NOTE: We detach memory for deep supervision - each segment gets independent gradients
    # This is consistent with HRM's 1-step gradient approximation
    memory = model.encode(input_ids, attention_mask)
    # Keep a copy for encoder gradient (first segment only)
    memory_for_encoder_grad = memory
    
    # Detach memory so we don't try to backprop through encoder multiple times
    # The encoder learns from the first segment's gradients only (1-step approx)
    memory_detached = memory.detach().requires_grad_(True)
    
    src_padding_mask = (input_ids == model.pad_token_id)
    key_padding_mask = src_padding_mask
    
    # Initialize HRM state
    zH, zL = model.hrm_core.init_state(B, L, device)
    
    # Determine number of segments (ACT with exploration)
    M_max = config.MAX_SEGMENTS
    
    # Thinking Warmup Check
    is_warmup = epoch < config.THINKING_WARMUP_EPOCHS
    
    if is_warmup:
        # Force random number of segments between 2 and MAX during warmup
        # Rationale: Model needs to learn that thinking longer != bad before it can decide when to halt.
        M_min = torch.randint(2, M_max + 1, (1,)).item()
        # In warmup, min is effectively the forced length if we ignore halting
        # So we will just run until M_min-1 (M_min segments) or M_max
        M_forced_len = M_min 
    else:
        # Normal training
        M_forced_len = 1
    
    total_lm_loss = 0
    total_q_loss = 0
    total_segments = 0
    halted = torch.zeros(B, dtype=torch.bool, device=device)
    
    # Track metrics
    all_q_halt = []
    all_q_continue = []
    
    for m in range(M_max):
        # Forward one segment
        # Use detached memory for HRM (allows multiple backward passes)
        zH, zL, q_halt, q_continue = model.forward_one_segment(
            zH, zL, memory_detached, key_padding_mask
        )
        
        all_q_halt.append(q_halt.mean().item())
        all_q_continue.append(q_continue.mean().item())
        
        # Compute output for this segment
        # Enhanced memory = original + reasoning result
        enhanced_memory = memory_detached + zH
        logits = model.decode(enhanced_memory, labels, src_padding_mask)
        
        # Language modeling loss
        lm_loss = compute_loss(logits, labels)
        
        # Check correctness for Q-learning target
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct = (preds == labels) & mask
            seq_correct = ((correct.sum(dim=-1) == mask.sum(dim=-1)) & (mask.sum(dim=-1) > 0)).float()
        
        if is_warmup:
            # WARMUP: Disable Q-loss, Force Execution
            q_loss = torch.tensor(0.0, device=device, requires_grad=True) # Zero loss with grad to avoid unused param error
            
            # Total segment loss (only LM)
            segment_loss = lm_loss
        else:
            # Q-halt loss: train to predict correctness
            q_halt_loss = F.binary_cross_entropy(q_halt, seq_correct.detach())
            
            # Q-continue loss: bootstrap from next segment (if not last)
            if m < M_max - 1:
                with torch.no_grad():
                    next_zH, next_zL, next_q_halt, next_q_continue = model.forward_one_segment(
                        zH.clone(), zL.clone(), memory_detached, key_padding_mask
                    )
                    target_q = torch.max(next_q_halt, next_q_continue)
                q_continue_loss = F.binary_cross_entropy(q_continue, target_q.detach())
            else:
                q_continue_loss = F.binary_cross_entropy(q_continue, seq_correct.detach())
            
            q_loss = q_halt_loss + q_continue_loss
            
            # Total segment loss
            segment_loss = lm_loss + 0.5 * q_loss
        
        # Backprop for this segment
        optimizer.zero_grad()
        segment_loss.backward()
        
        # For first segment, also backprop encoder gradients if not frozen
        if m == 0 and memory_for_encoder_grad is not None and memory_detached.grad is not None:
            # Propagate gradient to encoder
            memory_for_encoder_grad.backward(memory_detached.grad)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRADIENT_CLIP)
        optimizer.step()
        
        # CRITICAL: Detach state for next segment (1-step gradient approximation)
        zH = zH.detach()
        zL = zL.detach()
        
        total_lm_loss += lm_loss.item()
        total_q_loss += q_loss.item()
        total_segments += 1
        
        # Halting decision
        with torch.no_grad():
            if is_warmup:
                # During warmup, ignore Q-head and just run until forced length
                should_halt = torch.tensor([False] * B, device=device)
                
                # Check if we reached the random forced length (m is 0-indexed)
                if m >= M_forced_len - 1:
                    should_halt = torch.tensor([True] * B, device=device)
                    
                halted = halted | should_halt
                if halted.all():
                    break
            else:
                # Normal ACT with exploration
                # Sample M_min for exploration
                if torch.rand(1).item() < config.MIN_SEGMENTS_PROB:
                    M_min = torch.randint(2, M_max + 1, (1,)).item()
                else:
                    M_min = 1
                
                # Halt when q_halt > q_continue and past minimum
                should_halt = (q_halt > q_continue) & (m >= M_min - 1)
                halted = halted | should_halt
                
                if halted.all():
                    break
    
    metrics = {
        'lm_loss': total_lm_loss / total_segments,
        'q_loss': total_q_loss / total_segments,
        'segments': total_segments,
        'q_halt_mean': sum(all_q_halt) / len(all_q_halt),
        'q_continue_mean': sum(all_q_continue) / len(all_q_continue),
    }
    
    return metrics


def evaluate(model, dataloader, config, num_samples=5):
    """Evaluate model with ACT (no exploration)"""
    model.eval()
    device = config.DEVICE
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    total_segments = 0
    samples = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            B, L = input_ids.shape
            
            memory = model.encode(input_ids, attention_mask)
            src_padding_mask = (input_ids == model.pad_token_id)
            key_padding_mask = src_padding_mask
            
            zH, zL = model.hrm_core.init_state(B, L, device)
            
            # Run segments until halt or max
            for m in range(config.MAX_SEGMENTS):
                zH, zL, q_halt, q_continue = model.forward_one_segment(
                    zH, zL, memory, key_padding_mask
                )
                
                # Check halt condition (no exploration at eval)
                if (q_halt > q_continue).all() or m == config.MAX_SEGMENTS - 1:
                    break
            
            segments = m + 1
            total_segments += segments * B
            
            # Final prediction
            enhanced_memory = memory + zH
            logits = model.decode(enhanced_memory, labels, src_padding_mask)
            
            loss = compute_loss(logits, labels)
            total_loss += loss.item()
            
            # Accuracy
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct = (preds == labels) & mask
            seq_correct = ((correct.sum(dim=-1) == mask.sum(dim=-1)) & (mask.sum(dim=-1) > 0))
            total_correct += seq_correct.sum().item()
            total_samples += B
            
            # Collect samples
            if len(samples) < num_samples:
                for i in range(min(num_samples - len(samples), B)):
                    pred_ids = preds[i]
                    eos_idx = (pred_ids == model.tokenizer.eos_token_id).nonzero()
                    if len(eos_idx) > 0:
                        pred_ids = pred_ids[:eos_idx[0]]
                    pred_text = model.tokenizer.decode(pred_ids, skip_special_tokens=True)
                    samples.append({
                        'question': batch['raw_question'][i],
                        'expected': batch['raw_answer'][i],
                        'predicted': pred_text,
                        'segments': segments
                    })
    
    model.train()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': total_correct / total_samples if total_samples > 0 else 0,
        'avg_segments': total_segments / total_samples if total_samples > 0 else 0,
        'samples': samples
    }


def train(config=ConfigV3):
    print("=" * 60)
    print("HRM Training V3 - Correct Hierarchical Implementation")
    print("=" * 60)
    print(f"Device: {config.DEVICE}")
    print(f"Config: {config.to_dict()}")
    
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    print("\nBuilding model...")
    model = NanoHRMv3(tokenizer, config).to(config.DEVICE)
    params = count_parameters(model)
    print(f"   Total parameters: {params['total']:,}")
    print(f"   Trainable parameters: {params['trainable']:,}")
    
    print("\nLoading datasets (SQuAD only for cleaner signal)...")
    train_dataset = SQuADDataset(tokenizer, 'train', config.TRAIN_SIZE)
    val_dataset = SQuADDataset(tokenizer, 'validation', config.VAL_SIZE)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    all_metrics = {
        'config': config.to_dict(),
        'params': params,
        'train': [],
        'validation': [],
    }
    
    best_val_loss = float('inf')
    
    print("\nStarting training with deep supervision...")
    print("-" * 60)
    
    for epoch in range(config.EPOCHS):
        model.train()
        
        epoch_metrics = {
            'epoch': epoch + 1,
            'lm_loss': 0,
            'q_loss': 0,
            'segments': 0,
        }
        
        # Freeze/unfreeze encoder
        # Warmup notification
        if epoch < config.THINKING_WARMUP_EPOCHS:
             print(f"\n   [WARMUP MODE]: ACT disabled. Q-head waiting. Forcing {config.MAX_SEGMENTS} segments logic.")
             
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", ncols=100)
        
        for batch in pbar:
            metrics = train_step_deep_supervision(model, batch, optimizer, config, epoch)
            
            epoch_metrics['lm_loss'] += metrics['lm_loss']
            epoch_metrics['q_loss'] += metrics['q_loss']
            epoch_metrics['segments'] += metrics['segments']
            
            pbar.set_postfix({
                'lm': f"{metrics['lm_loss']:.3f}",
                'seg': f"{metrics['segments']:.1f}",
                'qh': f"{metrics['q_halt_mean']:.2f}",
                'qc': f"{metrics['q_continue_mean']:.2f}"
            })
        
        n_batches = len(train_loader)
        epoch_metrics['lm_loss'] /= n_batches
        epoch_metrics['q_loss'] /= n_batches
        epoch_metrics['segments'] /= n_batches
        epoch_metrics['memory'] = get_memory_usage()
        
        all_metrics['train'].append(epoch_metrics)
        
        # Validation
        val_results = evaluate(model, val_loader, config, config.NUM_VAL_SAMPLES)
        
        all_metrics['validation'].append({
            'epoch': epoch + 1,
            'loss': val_results['loss'],
            'accuracy': val_results['accuracy'],
            'avg_segments': val_results['avg_segments']
        })
        
        print(f"\nEpoch {epoch+1}: Train LM={epoch_metrics['lm_loss']:.4f}, "
              f"Segments={epoch_metrics['segments']:.1f}, "
              f"Val Loss={val_results['loss']:.4f}, "
              f"Val Acc={val_results['accuracy']:.4f}, "
              f"Val Seg={val_results['avg_segments']:.2f}")
        
        # Print samples
        print("\nSample outputs:")
        for s in val_results['samples'][:3]:
            print(f"  Q: {s['question'][:70]}...")
            print(f"  Expected: {s['expected'][:40]}")
            print(f"  Predicted: {s['predicted'][:40]} (seg: {s['segments']})")
            print()
        
        # Save best
        if val_results['loss'] < best_val_loss:
            best_val_loss = val_results['loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_results['loss'],
                'val_accuracy': val_results['accuracy'],
                'config': config.to_dict()
            }, config.MODEL_SAVE_PATH)
            print(f"   Saved best model (val_loss={val_results['loss']:.4f})")
    
    # Save final metrics
    save_metrics(all_metrics, f"{config.RESULTS_DIR}/metrics_v3_{timestamp}.json")
    save_metrics(all_metrics, f"{config.RESULTS_DIR}/metrics_v3_latest.json")
    
    print(f"\nTraining complete!")
    print(f"   Results: {config.RESULTS_DIR}/metrics_v3_{timestamp}.json")
    print(f"   Model: {config.MODEL_SAVE_PATH}")
    
    return model, all_metrics


if __name__ == "__main__":
    model, metrics = train()
