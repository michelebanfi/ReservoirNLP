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


# ============== PonderNet Components ==============

class HaltingNetwork(nn.Module):
    """
    PonderNet-style halting network (MLP).
    Outputs λ_n (halting probability) at each pondering step.
    """
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        # Initialize to encourage continuing early on
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)
    
    def forward(self, state):
        """
        Args:
            state: [B, L, D] reasoning state
        Returns:
            lambda_n: [B, 1] halting probability
        """
        pooled = state.mean(dim=1)  # [B, D]
        return torch.sigmoid(self.net(pooled))  # [B, 1]


class ReconstructionLoss(nn.Module):
    """
    PonderNet reconstruction loss: L_rec = Σ p_n * L(y, y_hat_n)
    Weights each step's loss by its halting probability.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, halt_probs, step_losses):
        """
        Args:
            halt_probs: [N] probability of halting at each step (sums to ~1)
            step_losses: [N] loss at each step
        Returns:
            weighted_loss: scalar
        """
        # Weight losses by halting probabilities
        return (halt_probs * step_losses).sum()


class RegularizationLoss(nn.Module):
    """
    PonderNet regularization loss: KL divergence from geometric prior.
    Encourages exploration and prevents collapse.
    """
    def __init__(self, lambda_p: float, max_steps: int = 20):
        super().__init__()
        # Pre-compute geometric distribution p_G(k) = (1-λp)^k * λp
        p_g = torch.zeros(max_steps)
        not_halted = 1.0
        for k in range(max_steps):
            p_g[k] = not_halted * lambda_p
            not_halted = not_halted * (1 - lambda_p)
        self.register_buffer('p_g', p_g)
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, p):
        """
        Args:
            p: [N, B] halting probabilities per step and batch
        Returns:
            kl_loss: scalar
        """
        # p: [N, B] -> [B, N]
        p = p.transpose(0, 1)
        # Get geometric prior up to N steps, expand across batch
        p_g = self.p_g[:p.shape[1]].unsqueeze(0).expand_as(p)
        # KL divergence (input is log probabilities)
        # Clamp to avoid log(0)
        p_clamped = p.clamp(min=1e-8)
        return self.kl_div(p_clamped.log(), p_g)


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
        
        # PonderNet halting network (replaces simple Q-head)
        self.halting_net = HaltingNetwork(dim, config.HALTING_HIDDEN_DIM)
        
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
        """Predict halting probability λ_n from answer state using PonderNet network"""
        return self.halting_net(y)  # [B, 1]


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
        
        # PonderNet loss modules
        self.reconstruction_loss = ReconstructionLoss()
        self.regularization_loss = RegularizationLoss(
            lambda_p=config.LAMBDA_P,
            max_steps=config.N_SUPERVISION
        )
    
    def forward(self, input_ids, attention_mask=None, 
                decoder_input_ids=None, labels=None,
                n_supervision=None, min_steps=None):
        """
        Forward pass with PonderNet-style deep supervision.
        
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
        
        # 3. PonderNet supervision loop
        # Collect: λ_n (halting prob), p_n (unconditioned halt), losses per step
        lambda_n_list = []      # [N] each is [B, 1]
        p_n_list = []           # [N] each is [B, 1]  
        step_losses = []        # [N] each is scalar
        all_y_outs = []         # For final prediction
        
        un_halted_prob = torch.ones(B, 1, device=device)  # Π(1-λ_j)
        
        for step in range(n_supervision):
            (y, z), y_out, lambda_n = self.reasoning.deep_recursion(
                memory, y, z, key_padding_mask=padding_mask
            )
            
            # Store lambda_n
            lambda_n_list.append(lambda_n.mean().item())
            
            # Compute p_n = λ_n × Π(1-λ_j) for j < n
            p_n = un_halted_prob * lambda_n  # [B, 1]
            p_n_list.append(p_n)
            
            # Update un_halted probability for next step
            un_halted_prob = un_halted_prob * (1 - lambda_n)
            
            # Compute loss at this step
            refined_memory = memory + y_out  # Residual
            all_y_outs.append(y_out)
            
            if decoder_input_ids is not None and labels is not None:
                logits = self.decoder(decoder_input_ids, refined_memory, padding_mask)
                step_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    label_smoothing=self.config.LABEL_SMOOTHING
                )
                step_losses.append(step_loss)
        
        # 4. Compute PonderNet losses
        if step_losses:
            # Stack p_n: [N, B, 1] -> mean over batch -> [N]
            p_n_stacked = torch.stack([p.mean() for p in p_n_list])  # [N]
            step_losses_stacked = torch.stack(step_losses)  # [N]
            
            # Normalize p_n to sum to 1 (handle numerical issues)
            p_n_normalized = p_n_stacked / (p_n_stacked.sum() + 1e-8)
            
            # Reconstruction loss: Σ p_n * L_n
            rec_loss = self.reconstruction_loss(p_n_normalized, step_losses_stacked)
            
            # Regularization loss: KL(p || p_geometric)
            # Stack p_n: [N, B] for KL computation
            p_for_kl = torch.stack([p.squeeze(-1) for p in p_n_list], dim=0)  # [N, B]
            reg_loss = self.regularization_loss(p_for_kl)
            
            # Total loss
            total_loss = rec_loss + self.config.REG_LOSS_WEIGHT * reg_loss
            
            # Use final step's logits for output
            final_refined_memory = memory + all_y_outs[-1]
            final_logits = self.decoder(decoder_input_ids, final_refined_memory, padding_mask)
        else:
            total_loss = torch.tensor(0.0, device=device)
            rec_loss = torch.tensor(0.0, device=device)
            reg_loss = torch.tensor(0.0, device=device)
            final_logits = None
            final_refined_memory = memory + all_y_outs[-1] if all_y_outs else memory
        
        # Compute expected number of steps: Σ n * p_n
        if p_n_list:
            step_indices = torch.arange(1, len(p_n_list) + 1, device=device, dtype=torch.float)
            p_n_stacked = torch.stack([p.mean() for p in p_n_list])
            expected_steps = (step_indices * p_n_stacked).sum().item()
        else:
            expected_steps = 0
        
        return {
            'logits': final_logits,
            'loss': total_loss,
            'rec_loss': rec_loss.item() if torch.is_tensor(rec_loss) else rec_loss,
            'reg_loss': reg_loss.item() if torch.is_tensor(reg_loss) else reg_loss,
            'supervision_steps': len(step_losses),
            'expected_steps': expected_steps,
            'q_hats': lambda_n_list,  # Keep for backwards compatibility
            'halt_probs': [p.mean().item() for p in p_n_list],
            'refined_memory': final_refined_memory,
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
