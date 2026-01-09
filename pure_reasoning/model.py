"""
Pure Reasoning Architecture - Model

A reasoning-focused model without pretrained T5.
Three components:
1. Encoder (from scratch)
2. Reasoning Core (TRM-style recursive)
3. Task Heads (span, classification)

Note: CUDA workaround is set in run_pure_reasoning.py, not here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .config import PureReasoningConfig


# ============== Building Blocks ==============

class RMSNorm(nn.Module):
    """RMSNorm (simpler than LayerNorm)"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SwiGLU(nn.Module):
    """SwiGLU activation (Shazeer 2020)"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4 * 2 / 3)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm"""
    def __init__(self, dim, num_heads, ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim = ff_dim or dim * 4
        
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
            bias=False
        )
        self.mlp = SwiGLU(dim, ff_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, key_padding_mask=None, attn_mask=None):
        # Pre-norm self-attention
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, 
                                 key_padding_mask=key_padding_mask,
                                 attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        
        # Pre-norm FFN
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


# ============== Encoder ==============

class PureEncoder(nn.Module):
    """
    Transformer encoder trained from scratch.
    Input: [CLS] context [SEP] question [SEP]
    Output: contextualized embeddings
    """
    def __init__(self, config, vocab_size=30522):  # BERT vocab size
        super().__init__()
        self.d_model = config.D_MODEL
        
        # Token + position embeddings
        self.token_embed = nn.Embedding(vocab_size, config.D_MODEL)
        self.pos_encoder = SinusoidalPE(config.D_MODEL, config.MAX_CONTEXT_LEN + config.MAX_QUESTION_LEN)
        self.embed_dropout = nn.Dropout(config.DROPOUT)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config.D_MODEL, config.N_HEADS, config.D_FF, config.DROPOUT)
            for _ in range(config.N_ENCODER_LAYERS)
        ])
        
        self.final_norm = RMSNorm(config.D_MODEL)
        
        # Initialize embeddings
        nn.init.normal_(self.token_embed.weight, std=0.02)
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: [B, L] token indices
            attention_mask: [B, L] 1=valid, 0=padding
        Returns:
            memory: [B, L, D] contextualized embeddings
            padding_mask: [B, L] True=ignore (for attention)
        """
        x = self.token_embed(input_ids)
        x = self.pos_encoder(x)
        x = self.embed_dropout(x)
        
        # Convert attention_mask to key_padding_mask (True=ignore)
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        
        x = self.final_norm(x)
        return x, key_padding_mask


# ============== Reasoning Core (TRM-style) ==============

class ReasoningCore(nn.Module):
    """
    TRM-style recursive reasoning network.
    
    Updates latent state (y, z) through recursive refinement:
    - z: reasoning scratchpad
    - y: current answer representation
    
    ACT mechanism for adaptive computation.
    """
    def __init__(self, config):
        super().__init__()
        dim = config.D_MODEL
        self.n = config.N_RECURSIONS
        self.T = config.T_DEEP_RECURSIONS
        
        # Reasoning transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, config.N_HEADS, config.D_FF, config.DROPOUT)
            for _ in range(config.N_REASONING_LAYERS)
        ])
        
        # Positional encoding for reasoning steps
        self.pos_encoder = SinusoidalPE(dim)
        
        # Projection layers for combining inputs
        self.z_input_proj = nn.Linear(dim * 3, dim, bias=False)  # (x, y, z) -> z
        self.y_input_proj = nn.Linear(dim * 2, dim, bias=False)  # (y, z) -> y
        
        # Q-head for ACT (predicts halting)
        self.q_head = nn.Linear(dim, 1, bias=True)
        with torch.no_grad():
            nn.init.normal_(self.q_head.weight, std=0.02)
            self.q_head.bias.data = torch.tensor([config.Q_HEAD_BIAS_INIT])
        
        # Learnable initial states
        self.y_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    
    def init_state(self, batch_size, seq_len, device):
        """Initialize y and z for a new sample"""
        y = self.y_init.expand(batch_size, seq_len, -1).clone()
        z = self.z_init.expand(batch_size, seq_len, -1).clone()
        return y.to(device), z.to(device)
    
    def apply_network(self, x, key_padding_mask=None):
        """Apply reasoning transformer layers"""
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x
    
    def latent_recursion(self, x, y, z, key_padding_mask=None):
        """
        One cycle of latent recursion:
        1. n times: update z given (x, y, z)
        2. Once: update y given (y, z)
        """
        for _ in range(self.n):
            combined = torch.cat([x, y, z], dim=-1)
            z_input = self.z_input_proj(combined)
            z = self.apply_network(z_input, key_padding_mask)
        
        combined_y = torch.cat([y, z], dim=-1)
        y_input = self.y_input_proj(combined_y)
        y = self.apply_network(y_input, key_padding_mask)
        
        return y, z
    
    def deep_recursion(self, x, y, z, key_padding_mask=None):
        """
        Deep recursion: T-1 no-grad + 1 with-grad
        Returns: (y_detached, z_detached), y_out, q_hat
        """
        # T-1 recursions without gradient
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.latent_recursion(x, y, z, key_padding_mask)
        
        # Final recursion with gradient
        y, z = self.latent_recursion(x, y, z, key_padding_mask)
        
        # Q-value (halt probability)
        q_hat = self.get_q_values(y)
        
        return (y.detach(), z.detach()), y, q_hat
    
    def get_q_values(self, y):
        """Predict halt probability from answer state"""
        pooled = y.mean(dim=1)  # [B, D]
        return torch.sigmoid(self.q_head(pooled))  # [B, 1]


# ============== Task Heads ==============

class SpanHead(nn.Module):
    """
    Predicts start and end positions for extractive QA.
    Used for SQuAD and HotpotQA.
    """
    def __init__(self, dim):
        super().__init__()
        self.start_proj = nn.Linear(dim, 1)
        self.end_proj = nn.Linear(dim, 1)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [B, L, D] from reasoning core
            attention_mask: [B, L] 1=valid, 0=padding
        Returns:
            start_logits: [B, L]
            end_logits: [B, L]
        """
        start_logits = self.start_proj(hidden_states).squeeze(-1)  # [B, L]
        end_logits = self.end_proj(hidden_states).squeeze(-1)      # [B, L]
        
        # Mask out padding positions
        if attention_mask is not None:
            mask = (attention_mask == 0)
            start_logits = start_logits.masked_fill(mask, float('-inf'))
            end_logits = end_logits.masked_fill(mask, float('-inf'))
        
        return start_logits, end_logits


class ClassificationHead(nn.Module):
    """
    Classification head for yes/no answers (HotpotQA)
    and answer type prediction.
    """
    def __init__(self, dim, num_classes=3):
        super().__init__()
        # num_classes: 0=span, 1=yes, 2=no
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, num_classes)
        )
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [B, L, D]
        Returns:
            logits: [B, num_classes]
        """
        # Use [CLS] token (position 0) for classification
        cls_hidden = hidden_states[:, 0, :]  # [B, D]
        return self.classifier(cls_hidden)


class NumericHead(nn.Module):
    """
    For DROP-style counting/arithmetic.
    Predicts a number directly.
    """
    def __init__(self, dim, max_count=10):
        super().__init__()
        self.max_count = max_count
        self.counter = nn.Linear(dim, max_count + 1)  # 0 to max_count
    
    def forward(self, hidden_states):
        cls_hidden = hidden_states[:, 0, :]
        return self.counter(cls_hidden)  # [B, max_count+1]


# ============== Full Model ==============

class PureReasoningModel(nn.Module):
    """
    Complete pure reasoning model.
    
    Architecture:
    1. Encoder: input_ids -> contextualized memory
    2. Reasoning Core: iterative refinement with ACT
    3. Task Heads: span + classification + numeric
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = PureEncoder(config)
        self.reasoning = ReasoningCore(config)
        
        # Task heads
        self.span_head = SpanHead(config.D_MODEL)
        self.class_head = ClassificationHead(config.D_MODEL, num_classes=3)
        self.numeric_head = NumericHead(config.D_MODEL, max_count=10)
    
    def forward(self, input_ids, attention_mask=None, 
                start_positions=None, end_positions=None,
                answer_type=None, numeric_answer=None,
                n_supervision=None, min_steps=None):
        """
        Forward pass with deep supervision.
        
        Args:
            input_ids: [B, L]
            attention_mask: [B, L]
            start_positions: [B] for span loss
            end_positions: [B] for span loss
            answer_type: [B] 0=span, 1=yes, 2=no
            numeric_answer: [B] for counting
            n_supervision: override config.N_SUPERVISION
            min_steps: override config.MIN_SUPERVISION_STEPS
        
        Returns:
            dict with losses and predictions
        """
        n_supervision = n_supervision or self.config.N_SUPERVISION
        min_steps = min_steps or self.config.MIN_SUPERVISION_STEPS
        device = input_ids.device
        B, L = input_ids.shape
        
        # 1. Encode input
        memory, padding_mask = self.encoder(input_ids, attention_mask)
        
        # 2. Initialize reasoning state
        y, z = self.reasoning.init_state(B, L, device)
        
        # 3. Deep supervision loop
        total_span_loss = 0.0
        total_class_loss = 0.0
        total_act_loss = 0.0
        supervision_steps = 0
        
        all_start_logits = []
        all_end_logits = []
        all_q_hats = []
        
        for step in range(n_supervision):
            # Deep recursion step
            (y, z), y_out, q_hat = self.reasoning.deep_recursion(
                memory, y, z, key_padding_mask=padding_mask
            )
            
            supervision_steps = step + 1
            all_q_hats.append(q_hat.mean().item())
            
            # Get predictions from this step
            start_logits, end_logits = self.span_head(y_out, attention_mask)
            all_start_logits.append(start_logits)
            all_end_logits.append(end_logits)
            
            # Compute losses if labels provided
            if start_positions is not None and end_positions is not None:
                span_loss = F.cross_entropy(start_logits, start_positions, ignore_index=-1)
                span_loss += F.cross_entropy(end_logits, end_positions, ignore_index=-1)
                total_span_loss += span_loss
                
                # ACT loss: is current prediction correct?
                with torch.no_grad():
                    pred_start = start_logits.argmax(dim=-1)
                    pred_end = end_logits.argmax(dim=-1)
                    correct = ((pred_start == start_positions) & (pred_end == end_positions)).float()
                
                act_loss = F.binary_cross_entropy(q_hat.squeeze(-1), correct)
                total_act_loss += act_loss
            
            if answer_type is not None:
                class_logits = self.class_head(y_out)
                class_loss = F.cross_entropy(class_logits, answer_type)
                total_class_loss += class_loss
            
            # Early stopping (after minimum steps)
            if step >= min_steps - 1 and q_hat.mean().item() > 0.5:
                break
        
        # Use final predictions
        final_start_logits = all_start_logits[-1]
        final_end_logits = all_end_logits[-1]
        final_class_logits = self.class_head(y_out)
        final_numeric_logits = self.numeric_head(y_out)
        
        return {
            'start_logits': final_start_logits,
            'end_logits': final_end_logits,
            'class_logits': final_class_logits,
            'numeric_logits': final_numeric_logits,
            'span_loss': total_span_loss / supervision_steps if supervision_steps > 0 else 0,
            'class_loss': total_class_loss / supervision_steps if supervision_steps > 0 else 0,
            'act_loss': total_act_loss / supervision_steps if supervision_steps > 0 else 0,
            'supervision_steps': supervision_steps,
            'q_hats': all_q_hats,
        }
    
    def get_metrics(self):
        """Return model size info"""
        total = sum(p.numel() for p in self.parameters())
        encoder = sum(p.numel() for p in self.encoder.parameters())
        reasoning = sum(p.numel() for p in self.reasoning.parameters())
        heads = total - encoder - reasoning
        
        return {
            'total_params_M': total / 1e6,
            'encoder_params_M': encoder / 1e6,
            'reasoning_params_M': reasoning / 1e6,
            'heads_params_M': heads / 1e6,
        }
