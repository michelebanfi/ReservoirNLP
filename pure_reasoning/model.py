"""
Pure Reasoning Architecture - Model

A reasoning-focused model without pretrained T5.
Three components:
1. Encoder (from scratch)
2. Reasoning Core (TRM-style recursive with PonderNet halting)
3. Decoder (for text generation)

CHANGELOG 2026-01-23: Major PonderNet fixes
- Fixed RegularizationLoss to use nn.KLDivLoss (was producing negative values)
- Fixed ReconstructionLoss to use per-sample weighting
- Added step embeddings for pondering position
- Continuous state refinement across pondering steps
- Better halting network initialization (biased toward continuing)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from contextlib import nullcontext

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
    
    Initialized to encourage continuing (low halt probability early in training).
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
        # Initialize to encourage CONTINUING (not halting)
        # sigmoid(-2) ≈ 0.12, so initially model has ~12% chance to halt per step
        nn.init.constant_(self.net[-1].bias, -2.0)
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
    PonderNet reconstruction loss: L_rec = Σ_n E_batch[p_n * L(y, y_hat_n)]
    
    FIXED: Now properly does per-sample weighting instead of batch mean.
    Each sample's loss is weighted by its own halting probability at each step.
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, p_n_list, step_losses_unreduced):
        """
        Args:
            p_n_list: list of [B,] tensors, halting probability per sample at each step
            step_losses_unreduced: list of [B,] tensors, CE loss per sample at each step
        Returns:
            weighted_loss: scalar
        """
        # Sum over steps: for each sample, weighted sum of losses
        # loss_i = Σ_n p_n[i] * L_n[i]
        total = 0.0
        for p_n, loss_n in zip(p_n_list, step_losses_unreduced):
            # p_n: [B], loss_n: [B]
            total = total + (p_n * loss_n).mean()  # Mean over batch
        return total


class RegularizationLoss(nn.Module):
    """
    PonderNet regularization loss: KL divergence from geometric prior.
    
    FIXED: Now uses nn.KLDivLoss which is always >= 0.
    Previous implementation used manual formula that could produce negative values.
    
    KL(p || p_G) where p_G(n) = (1-λp)^n * λp is geometric distribution.
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
        
        # Use PyTorch's KLDivLoss - CRITICAL FIX
        # KLDivLoss expects log-probabilities as input
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, p):
        """
        Args:
            p: [N, B] halting probabilities per step and batch
        Returns:
            kl_loss: scalar (always >= 0)
        """
        # p: [N, B] -> [B, N]
        p = p.transpose(0, 1)
        
        # Get geometric prior up to N steps, expand across batch
        N = p.shape[1]
        p_g = self.p_g[:N].unsqueeze(0).expand_as(p)
        
        # Clamp to avoid log(0)
        p_clamped = p.clamp(min=1e-8)
        
        # KLDivLoss(log(p), p_g) computes KL(p_g || p)
        # We want KL(p || p_g), so we swap: KLDivLoss(log(p_g), p) 
        # But actually, the standard way is: KL(input || target)
        # PyTorch KLDivLoss: input is log(Q), target is P, computes sum(P * (log(P) - input))
        # = sum(P * log(P) - P * log(Q)) = sum(P * log(P/Q)) = KL(P || Q)
        # So to get KL(p || p_g): input=log(p_g), target=p
        # But that's KL(target || exp(input)) = KL(p || p_g) ✓
        p_g_log = p_g.clamp(min=1e-8).log()
        
        kl = self.kl_div(p_g_log, p_clamped)
        return kl


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


# ============== Reasoning Core (TRM-style with PonderNet) ==============

class ReasoningCore(nn.Module):
    """
    TRM-style recursive reasoning network with PonderNet halting.
    
    Updates latent state (y, z) through recursive refinement:
    - z: reasoning scratchpad
    - y: current answer representation
    
    FIXED: 
    - Continuous state refinement across pondering steps
    - Step embeddings so model knows pondering position
    - Proper halting network initialization
    """
    def __init__(self, config):
        super().__init__()
        dim = config.D_MODEL
        self.n = config.N_RECURSIONS
        self.T = config.T_DEEP_RECURSIONS
        self.max_steps = config.N_SUPERVISION
        
        # Reasoning transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, config.N_HEADS, config.D_FF, config.DROPOUT)
            for _ in range(config.N_REASONING_LAYERS)
        ])
        
        # Positional encoding for reasoning steps
        self.pos_encoder = SinusoidalPE(dim)
        
        # Step embedding: tells model which pondering step we're on
        self.step_embed = nn.Embedding(config.N_SUPERVISION + 1, dim)
        nn.init.normal_(self.step_embed.weight, std=0.02)
        
        # Projection layers for combining inputs
        self.z_input_proj = nn.Linear(dim * 3, dim, bias=False)  # (x, y, z) -> z
        self.y_input_proj = nn.Linear(dim * 2, dim, bias=False)  # (y, z) -> y
        
        # PonderNet halting network
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
    
    def forward_step(self, memory, y, z, step_idx, key_padding_mask=None):
        """
        One complete pondering step with T deep recursions.
        
        FIXED: Uses gradient for final recursion only, detaches for T-1.
        Returns updated (y, z) and halt probability λ_n.
        """
        B = y.size(0)
        device = y.device
        
        # Add step embedding to y (tells model which pondering step)
        step_emb = self.step_embed(
            torch.full((B,), step_idx, device=device, dtype=torch.long)
        ).unsqueeze(1)  # [B, 1, D]
        y = y + step_emb
        
        # T deep recursions: T-1 without grad, 1 with grad
        for t in range(self.T):
            if t < self.T - 1:
                with torch.no_grad():
                    y, z = self.latent_recursion(memory, y, z, key_padding_mask)
            else:
                y, z = self.latent_recursion(memory, y, z, key_padding_mask)
        
        # Get halting probability
        lambda_n = self.halting_net(y)  # [B, 1]
        
        return y, z, lambda_n
    
    def get_q_values(self, y):
        """Predict halting probability λ_n from answer state"""
        return self.halting_net(y)


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
    Complete pure reasoning model with PonderNet adaptive computation.
    
    Architecture:
    1. Encoder: input_ids -> contextualized memory
    2. Reasoning Core: iterative refinement with ACT (PonderNet)
    3. Decoder: generates answer text from refined memory
    
    FIXED (2026-01-23):
    - Proper PonderNet reconstruction loss with per-sample weighting
    - Fixed KL divergence (always >= 0 now)
    - Continuous state refinement across steps
    - Step embeddings for pondering position
    - Last step always halts (λ_N = 1)
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
        Forward pass with PonderNet-style supervision.
        
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
        min_steps = min_steps or getattr(self.config, 'MIN_SUPERVISION_STEPS', 2)
        device = input_ids.device
        B, L = input_ids.shape
        
        # 1. Encode input
        memory, padding_mask = self.encoder(input_ids, attention_mask)
        
        # 2. Initialize reasoning state
        y, z = self.reasoning.init_state(B, L, device)
        
        # 3. PonderNet supervision loop with CONTINUOUS state refinement
        p_n_list = []           # [N] each is [B,] - halting probability at step n
        step_losses = []        # [N] each is [B,] - per-sample loss at step n  
        lambda_n_values = []    # For logging
        all_y_outs = []
        
        un_halted_prob = torch.ones(B, device=device)  # Π(1-λ_j)
        
        for step in range(n_supervision):
            # Forward one pondering step (CONTINUOUS: y, z not reset between steps)
            y, z, lambda_n = self.reasoning.forward_step(
                memory, y, z, step, key_padding_mask=padding_mask
            )
            lambda_n = lambda_n.squeeze(-1)  # [B]
            
            # Force λ_N = 1 at final step (PonderNet requirement)
            if step == n_supervision - 1:
                lambda_n = torch.ones_like(lambda_n)
            
            # For minimum steps, force λ_n = 0 (don't allow halting)
            if step < min_steps - 1:
                lambda_n = torch.zeros_like(lambda_n)
            
            lambda_n_values.append(lambda_n.mean().item())
            
            # Compute p_n = λ_n × Π(1-λ_j) for j < n
            p_n = un_halted_prob * lambda_n  # [B]
            p_n_list.append(p_n)
            
            # Update un_halted probability for next step
            un_halted_prob = un_halted_prob * (1 - lambda_n)
            
            # Compute per-sample loss at this step (no reduction!)
            refined_memory = memory + y  # Residual
            all_y_outs.append(y)
            
            if decoder_input_ids is not None and labels is not None:
                logits = self.decoder(decoder_input_ids, refined_memory, padding_mask)
                # Per-sample loss: [B*T] -> [B]
                step_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction='none',  # CRITICAL: no reduction!
                    label_smoothing=self.config.LABEL_SMOOTHING
                ).view(B, -1).mean(dim=1)  # [B]
                step_losses.append(step_loss)
        
        # 4. Compute PonderNet losses
        if step_losses:
            # Reconstruction loss: per-sample weighted sum
            rec_loss = self.reconstruction_loss(p_n_list, step_losses)
            
            # Regularization loss: KL(p || p_geometric)
            # Stack p_n: [N, B] for KL computation
            p_stacked = torch.stack(p_n_list, dim=0)  # [N, B]
            
            # Normalize to ensure sum=1 (handle numerical issues)
            p_stacked = p_stacked / (p_stacked.sum(dim=0, keepdim=True) + 1e-8)
            
            reg_loss = self.regularization_loss(p_stacked)
            
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
            p_n_means = torch.stack([p.mean() for p in p_n_list])
            expected_steps = (step_indices * p_n_means).sum().item()
        else:
            expected_steps = 0
        
        return {
            'logits': final_logits,
            'loss': total_loss,
            'rec_loss': rec_loss.item() if torch.is_tensor(rec_loss) else rec_loss,
            'reg_loss': reg_loss.item() if torch.is_tensor(reg_loss) else reg_loss,
            'supervision_steps': len(step_losses),
            'expected_steps': expected_steps,
            'q_hats': lambda_n_values,  # Keep for backwards compatibility
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
