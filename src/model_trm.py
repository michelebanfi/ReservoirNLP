"""
TRM (Tiny Recursion Model) Architecture

Based on arXiv:2510.04871 - "Less is More: Recursive Reasoning with Tiny Networks"

Key differences from HRM:
1. Single tiny network (2 layers) instead of separate f_H and f_L modules
2. Full gradient flow through n recursions (no 1-step gradient approximation)
3. z = latent reasoning, y = current answer (simpler interpretation than hierarchical)
4. Simpler ACT: q_hat directly predicts (y_hat == y_true)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import T5ForConditionalGeneration
from .config_trm import TRMConfig


class RMSNorm(nn.Module):
    """RMSNorm from LLaMA/Gemma"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding"""
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

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


class TRMTransformerBlock(nn.Module):
    """
    Single transformer block for TRM.
    Unlike HRM, there's only one network applied to both z and y updates.
    """
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout,
            bias=False
        )
        self.mlp = SwiGLU(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, key_padding_mask=None):
        # Pre-norm architecture
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x


class TinyRecursionCore(nn.Module):
    """
    TRM Core: Single tiny network applied recursively.
    
    From the paper (Algorithm 3):
    - latent_recursion: n steps updating z given (x, y, z), then update y given (y, z)
    - deep_recursion: T-1 no-grad + 1 with-grad iterations of latent_recursion
    
    Key: Uses SAME network for both z and y updates (weight sharing)
    """
    def __init__(self, dim, num_heads, config):
        super().__init__()
        self.dim = dim
        self.n = config.N_RECURSIONS          # latent recursion steps
        self.T = config.T_DEEP_RECURSIONS     # deep recursion iterations
        
        # Single tiny network (paper: 2 layers performs best)
        self.layers = nn.ModuleList([
            TRMTransformerBlock(dim, num_heads, config.DROPOUT)
            for _ in range(config.N_LAYERS)
        ])
        
        # Positional encoding
        self.pos_encoder = SinusoidalPositionalEncoding(dim)
        
        # Projection layers for combining inputs
        # z update: net(concat(x, y, z)) -> need 3*dim -> dim
        self.z_input_proj = nn.Linear(dim * 3, dim, bias=False)
        # y update: net(concat(y, z)) -> need 2*dim -> dim  
        self.y_input_proj = nn.Linear(dim * 2, dim, bias=False)
        
        # Q-head for ACT: predicts whether current answer is correct
        self.q_head = nn.Linear(dim, 1, bias=True)
        with torch.no_grad():
            nn.init.normal_(self.q_head.weight, mean=0.0, std=0.02)
            q_bias = getattr(config, 'Q_HEAD_BIAS_INIT', 0.0)
            self.q_head.bias.data = torch.tensor([q_bias])
        
        # Learnable initial states
        self.y_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.z_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    
    def init_state(self, batch_size, seq_len, device):
        """Initialize y and z for a new sample"""
        y = self.y_init.expand(batch_size, seq_len, -1).clone()
        z = self.z_init.expand(batch_size, seq_len, -1).clone()
        return y.to(device), z.to(device)
    
    def apply_network(self, x, key_padding_mask=None):
        """Apply the tiny network (shared for all updates)"""
        x = self.pos_encoder(x)
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x
    
    def latent_recursion(self, x, y, z, key_padding_mask=None):
        """
        One cycle of latent recursion (Algorithm 3, lines 2-5):
        1. n times: update z given (x, y, z)
        2. Once: update y given (y, z)
        
        Returns: updated (y, z)
        """
        # n steps of latent reasoning
        for _ in range(self.n):
            # Combine x, y, z -> project to dim
            combined = torch.cat([x, y, z], dim=-1)  # [B, L, 3*D]
            z_input = self.z_input_proj(combined)     # [B, L, D]
            z = self.apply_network(z_input, key_padding_mask)
        
        # Update answer y based on current z
        combined_y = torch.cat([y, z], dim=-1)       # [B, L, 2*D]
        y_input = self.y_input_proj(combined_y)       # [B, L, D]
        y = self.apply_network(y_input, key_padding_mask)
        
        return y, z
    
    def deep_recursion(self, x, y, z, key_padding_mask=None):
        """
        Deep recursion (Algorithm 3, lines 8-14):
        1. T-1 times: latent_recursion without gradient
        2. 1 time: latent_recursion with gradient
        
        Returns: (y.detach(), z.detach()), output_logits, q_hat
        """
        # T-1 recursions without gradient (improve y, z before final pass)
        with torch.no_grad():
            for _ in range(self.T - 1):
                y, z = self.latent_recursion(x, y, z, key_padding_mask)
        
        # Final recursion with gradient
        y, z = self.latent_recursion(x, y, z, key_padding_mask)
        
        # Get Q-value (halt probability)
        q_hat = self.get_q_values(y)
        
        # Return detached states for next supervision step
        return (y.detach(), z.detach()), y, q_hat
    
    def get_q_values(self, y):
        """Predict halt probability from current answer embedding"""
        pooled = y.mean(dim=1)  # [B, D]
        return torch.sigmoid(self.q_head(pooled))  # [B, 1]


class ReasoningPooler(nn.Module):
    """
    Pool reasoning state y [B, L, D] into K soft-prompt tokens [B, K, D].
    These tokens are prepended to encoder memory for decoder cross-attention.
    
    (Reused from HRM architecture for T5 integration)
    """
    def __init__(self, dim, n_tokens, num_heads=8, dropout=0.1):
        super().__init__()
        self.n_tokens = n_tokens
        
        # Learned query tokens for pooling
        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        
        # Cross-attention: queries attend over reasoning state
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
            bias=False
        )
        
        # FFN after cross-attention
        self.mlp = SwiGLU(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, y, key_padding_mask=None):
        """
        Args:
            y: [B, L, D] - current answer embedding from TRM
            key_padding_mask: [B, L] boolean, True = ignore
        Returns:
            reasoning_tokens: [B, K, D] - pooled soft-prompt tokens
        """
        B = y.size(0)
        queries = self.query_tokens.expand(B, -1, -1)  # [B, K, D]
        
        # Cross-attention: queries attend to y
        attn_out, _ = self.cross_attn(
            query=queries,
            key=y,
            value=y,
            key_padding_mask=key_padding_mask
        )
        
        # Residual + norm
        x = self.norm1(queries + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        
        return x


class TinyRecursionModel(nn.Module):
    """
    Full TRM model with T5 encoder/decoder integration.
    
    Architecture:
    1. T5 Encoder produces memory [B, L, D]
    2. TRM Core recursively refines (y, z) using memory as context
    3. y is pooled into K reasoning tokens, prepended to memory
    4. T5 Decoder cross-attends to enhanced memory
    """
    def __init__(self, tokenizer, config):
        super().__init__()
        self.config = config
        
        print(f"Loading Pretrained T5 from {config.TOKENIZER_NAME}...")
        self.t5_model = T5ForConditionalGeneration.from_pretrained(config.TOKENIZER_NAME)
        
        # Access T5 components
        self.shared = self.t5_model.shared
        self.encoder = self.t5_model.encoder
        self.decoder = self.t5_model.decoder
        self.lm_head = self.t5_model.lm_head
        
        d_model = config.D_MODEL
        
        # TRM Core (replaces HRM's H/L modules)
        self.trm_core = TinyRecursionCore(d_model, config.N_HEADS, config)
        
        # Reasoning Pooler: pools y into K soft-prompt tokens
        self.reasoning_pooler = ReasoningPooler(
            dim=d_model,
            n_tokens=4,  # K=4 reasoning tokens
            num_heads=config.N_HEADS,
            dropout=config.DROPOUT
        )
        
        # For EMA
        self.ema_model = None
        
    def freeze_t5(self):
        """Freeze T5 parameters for curriculum learning"""
        print("Freezing T5 parameters...")
        for param in self.t5_model.parameters():
            param.requires_grad = False
            
    def unfreeze_t5(self):
        """Unfreeze T5 parameters"""
        print("Unfreezing T5 parameters...")
        for param in self.t5_model.parameters():
            param.requires_grad = True
    
    def encode(self, input_ids):
        """Run T5 encoder"""
        attention_mask = (input_ids != 0).long()
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        memory = encoder_outputs.last_hidden_state  # [B, L, D]
        
        # Boolean mask for MHA (True = ignore)
        src_mask_bool = (attention_mask == 0)
        
        return memory, src_mask_bool
    
    def prepare_enhanced_memory(self, memory, y, src_mask_bool):
        """
        Pool y into K reasoning tokens and prepend to memory.
        Returns enhanced memory [B, K+L, D] and extended mask [B, K+L].
        """
        # Pool y -> [B, K, D]
        reasoning_tokens = self.reasoning_pooler(y, key_padding_mask=src_mask_bool)
        
        # Prepend reasoning tokens to memory
        enhanced_memory = torch.cat([reasoning_tokens, memory], dim=1)
        
        # Extend mask: reasoning tokens are always valid
        B = memory.size(0)
        K = reasoning_tokens.size(1)
        reasoning_mask = torch.zeros(B, K, dtype=torch.bool, device=memory.device)
        enhanced_mask = torch.cat([reasoning_mask, src_mask_bool], dim=1)
        
        return enhanced_memory, enhanced_mask
    
    def decode(self, memory, labels, src_padding_mask_bool):
        """Decode with teacher forcing (for training)"""
        enc_attn_mask = (~src_padding_mask_bool).long()
        decoder_input_ids = self.t5_model._shift_right(labels)
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=memory,
            encoder_attention_mask=enc_attn_mask,
            return_dict=True
        )
        
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        return logits
    
    def generate_step(self, memory, decoder_input_ids, src_padding_mask_bool):
        """Single step of autoregressive generation (for inference)"""
        enc_attn_mask = (~src_padding_mask_bool).long()
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=memory,
            encoder_attention_mask=enc_attn_mask,
            return_dict=True
        )
        
        logits = self.lm_head(decoder_outputs.last_hidden_state)
        return logits
    
    def get_metrics(self):
        """Return current TRM-related metrics for logging"""
        total_params = sum(p.numel() for p in self.parameters())
        t5_params = sum(p.numel() for p in self.t5_model.parameters())
        trm_params = total_params - t5_params
        
        return {
            'total_params_M': total_params / 1e6,
            't5_params_M': t5_params / 1e6,
            'trm_params_M': trm_params / 1e6,
            'n_recursions': self.trm_core.n,
            't_deep_recursions': self.trm_core.T,
        }


class EMAModel:
    """
    Exponential Moving Average (EMA) of model weights.
    Key for TRM generalization (Table 1: no EMA = 79.9%, with EMA = 87.4%).
    """
    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Update EMA weights"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_avg = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_avg.clone()
    
    def apply_shadow(self, model):
        """Apply EMA weights (for evaluation)"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model):
        """Restore original weights (after evaluation)"""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
