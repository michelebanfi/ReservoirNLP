"""
Pure Reasoning Architecture - Model

A reasoning-focused model without pretrained T5.
Three components:
1. Encoder (from scratch)
2. Reasoning Core (TRM-style recursive)
3. Decoder (for text generation)

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


class DecoderBlock(nn.Module):
    """Transformer decoder block with cross-attention"""
    def __init__(self, dim, num_heads, ff_dim=None, dropout=0.1):
        super().__init__()
        ff_dim = ff_dim or dim * 4
        
        # Self-attention (causal)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
            bias=False
        )
        
        # Cross-attention to encoder
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
            bias=False
        )
        
        self.mlp = SwiGLU(dim, ff_dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, memory, tgt_mask=None, memory_key_padding_mask=None):
        # Self-attention with causal mask
        normed = self.norm1(x)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=tgt_mask)
        x = x + self.dropout(attn_out)
        
        # Cross-attention to encoder memory
        normed = self.norm2(x)
        cross_out, _ = self.cross_attn(normed, memory, memory, 
                                        key_padding_mask=memory_key_padding_mask)
        x = x + self.dropout(cross_out)
        
        # FFN
        x = x + self.dropout(self.mlp(self.norm3(x)))
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


# ============== Decoder ==============

class PureDecoder(nn.Module):
    """
    Transformer decoder for text generation.
    Uses cross-attention to attend to encoder memory.
    """
    def __init__(self, config, vocab_size=30522):
        super().__init__()
        self.d_model = config.D_MODEL
        
        # Token + position embeddings (shared with output projection)
        self.token_embed = nn.Embedding(vocab_size, config.D_MODEL)
        self.pos_encoder = SinusoidalPE(config.D_MODEL, config.MAX_ANSWER_LEN + 10)
        self.embed_dropout = nn.Dropout(config.DROPOUT)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            DecoderBlock(config.D_MODEL, config.N_HEADS, config.D_FF, config.DROPOUT)
            for _ in range(config.N_REASONING_LAYERS)  # Same depth as reasoning
        ])
        
        self.final_norm = RMSNorm(config.D_MODEL)
        
        # Output projection (tied with embeddings)
        self.output_proj = nn.Linear(config.D_MODEL, vocab_size, bias=False)
        self.output_proj.weight = self.token_embed.weight  # Weight tying
        
        # Initialize
        nn.init.normal_(self.token_embed.weight, std=0.02)
    
    def forward(self, decoder_input_ids, memory, memory_key_padding_mask=None):
        """
        Args:
            decoder_input_ids: [B, T] target tokens (shifted right)
            memory: [B, L, D] encoder output
            memory_key_padding_mask: [B, L] True=ignore
        Returns:
            logits: [B, T, Vocab]
        """
        B, T = decoder_input_ids.shape
        device = decoder_input_ids.device
        
        # Embed tokens
        x = self.token_embed(decoder_input_ids)
        x = self.pos_encoder(x)
        x = self.embed_dropout(x)
        
        # Causal mask (upper triangular)
        causal_mask = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), 
            diagonal=1
        )
        
        # Apply decoder layers
        for layer in self.layers:
            x = layer(x, memory, tgt_mask=causal_mask, 
                     memory_key_padding_mask=memory_key_padding_mask)
        
        x = self.final_norm(x)
        logits = self.output_proj(x)
        
        return logits
    
    def generate(self, memory, memory_key_padding_mask, tokenizer, max_len=50):
        """
        Autoregressive generation.
        
        Args:
            memory: [B, L, D] encoder output
            memory_key_padding_mask: [B, L]
            tokenizer: for BOS/EOS tokens
            max_len: max tokens to generate
        Returns:
            generated_ids: [B, T] generated token ids
        """
        B = memory.size(0)
        device = memory.device
        
        # Start with [CLS] (id=101 in BERT tokenizer) as BOS
        bos_id = tokenizer.cls_token_id or 101
        eos_id = tokenizer.sep_token_id or 102
        
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        
        for _ in range(max_len):
            logits = self.forward(generated, memory, memory_key_padding_mask)
            next_token_logits = logits[:, -1, :]  # [B, Vocab]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have EOS
            if (next_token == eos_id).all():
                break
        
        return generated


# ============== Full Model ==============

class PureReasoningModel(nn.Module):
    """
    Complete pure reasoning model with generative output.
    
    Architecture:
    1. Encoder: input_ids -> contextualized memory
    2. Reasoning Core: iterative refinement with ACT
    3. Decoder: generates answer text from refined memory
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Components
        self.encoder = PureEncoder(config)
        self.reasoning = ReasoningCore(config)
        self.decoder = PureDecoder(config)
    
    def forward(self, input_ids, attention_mask=None, 
                decoder_input_ids=None, labels=None,
                n_supervision=None, min_steps=None):
        """
        Forward pass with deep supervision.
        
        Args:
            input_ids: [B, L] encoder input
            attention_mask: [B, L]
            decoder_input_ids: [B, T] target tokens shifted right (starts with BOS)
            labels: [B, T] target tokens for loss (ends with EOS)
            n_supervision: override config.N_SUPERVISION
            min_steps: override config.MIN_SUPERVISION_STEPS
        
        Returns:
            dict with loss and logits
        """
        n_supervision = n_supervision or self.config.N_SUPERVISION
        min_steps = min_steps or self.config.MIN_SUPERVISION_STEPS
        device = input_ids.device
        B, L = input_ids.shape
        
        # 1. Encode input
        memory, padding_mask = self.encoder(input_ids, attention_mask)
        
        # 2. Initialize reasoning state
        y, z = self.reasoning.init_state(B, L, device)
        
        # 3. Deep supervision loop (refine memory)
        total_act_loss = 0.0
        supervision_steps = 0
        all_q_hats = []
        
        for step in range(n_supervision):
            (y, z), y_out, q_hat = self.reasoning.deep_recursion(
                memory, y, z, key_padding_mask=padding_mask
            )
            
            supervision_steps = step + 1
            all_q_hats.append(q_hat.mean().item())
            
            # TODO: add ACT loss based on generation quality
            
            # Early stopping
            if step >= min_steps - 1 and q_hat.mean().item() > 0.5:
                break
        
        # The reasoning output y_out now contains refined representations
        # We use it as additional context for the decoder by concatenating
        # OR just use encoder memory directly (simpler for now)
        refined_memory = memory + y_out  # Residual connection
        
        # 4. Decode (if training)
        if decoder_input_ids is not None:
            logits = self.decoder(decoder_input_ids, refined_memory, padding_mask)
            
            # Compute loss if labels provided
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100  # Ignore padding
                )
            else:
                loss = 0.0
        else:
            logits = None
            loss = 0.0
        
        return {
            'logits': logits,
            'loss': loss,
            'act_loss': total_act_loss / supervision_steps if supervision_steps > 0 else 0,
            'supervision_steps': supervision_steps,
            'q_hats': all_q_hats,
            'refined_memory': refined_memory,
            'memory_key_padding_mask': padding_mask,
        }
    
    def generate(self, input_ids, attention_mask, tokenizer, max_len=50):
        """
        Generate answer text.
        
        Args:
            input_ids: [B, L] encoder input
            attention_mask: [B, L]
            tokenizer: for special tokens
            max_len: max answer tokens
        Returns:
            generated_ids: [B, T]
        """
        # Forward pass to get refined memory
        outputs = self.forward(input_ids, attention_mask)
        
        # Generate from decoder
        generated = self.decoder.generate(
            outputs['refined_memory'],
            outputs['memory_key_padding_mask'],
            tokenizer,
            max_len
        )
        
        return generated
    
    def get_metrics(self):
        """Return model size info"""
        total = sum(p.numel() for p in self.parameters())
        encoder = sum(p.numel() for p in self.encoder.parameters())
        reasoning = sum(p.numel() for p in self.reasoning.parameters())
        decoder = sum(p.numel() for p in self.decoder.parameters())
        
        return {
            'total_params_M': total / 1e6,
            'encoder_params_M': encoder / 1e6,
            'reasoning_params_M': reasoning / 1e6,
            'decoder_params_M': decoder / 1e6,
        }
