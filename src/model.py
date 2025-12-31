import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import T5ForConditionalGeneration
from .config import Config

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
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
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        # T5-Base dim 768. 
        hidden_dim = hidden_dim or int(dim * 4 * 2 / 3)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=False),
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, context):
        g = self.gate(torch.cat([x, context], dim=-1))
        return x + g * self.proj(context)

class ReasoningPooler(nn.Module):
    """
    Pool zH [B, L, D] into K reasoning tokens [B, K, D] using cross-attention.
    These tokens serve as soft prompts that the decoder can attend to.
    
    If force_hrm=True, the gate is disabled and reasoning tokens are always at full strength.
    """
    def __init__(self, dim, n_tokens, num_heads=8, dropout=0.1, gate_init=0.1, force_hrm=False):
        super().__init__()
        self.n_tokens = n_tokens
        self.force_hrm = force_hrm
        
        # Learned query tokens for pooling
        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        
        # Cross-attention: queries attend over zH
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
        
        # Gating: only used if force_hrm=False
        # tanh(0.5) ≈ 0.46, so ~46% HRM contribution initially
        self.gate = nn.Parameter(torch.tensor([gate_init]))
        
    def forward(self, zH, key_padding_mask=None):
        """
        Args:
            zH: [B, L, D] - reasoning state from HRM
            key_padding_mask: [B, L] boolean, True = ignore
        Returns:
            reasoning_tokens: [B, K, D] - pooled soft-prompt tokens
        """
        B = zH.size(0)
        queries = self.query_tokens.expand(B, -1, -1)  # [B, K, D]
        
        # Cross-attention: queries attend to zH
        attn_out, _ = self.cross_attn(
            query=queries,
            key=zH,
            value=zH,
            key_padding_mask=key_padding_mask
        )
        
        # Residual + norm
        x = self.norm1(queries + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        
        # If force_hrm, return full strength; otherwise gate the contribution
        if self.force_hrm:
            return x  # No gate, full HRM
        else:
            return torch.tanh(self.gate) * x

class HRMTransformerBlock(nn.Module):
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
        # x: [B, Seq, Dim]
        # MultiheadAttention needs key_padding_mask as [B, Seq] (bool) or [B, 1, 1, Seq] (float)
        # T5 tokenizer returns 1 for attention, 0 for pad.
        # nn.MultiheadAttention expects:
        #   key_padding_mask: If specified, a binary mask of shape (N, S).
        #   "If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged."
        #   T5 mask is 1 (valid), 0 (pad). So we need (1-mask).bool() or (~mask.bool())
        
        # We handle mask conversion outside or here.
        # Assuming caller passes appropriate mask.
        
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.mlp(x)))
        return x

class HRMModule(nn.Module):
    def __init__(self, dim, num_heads, n_layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            HRMTransformerBlock(dim, num_heads, dropout)
            for _ in range(n_layers)
        ])
        self.fusion = GatedFusion(dim)
        
    def forward(self, x_primary, contexts, key_padding_mask=None):
        x = x_primary
        # Fuse contexts
        if isinstance(contexts, torch.Tensor):
            contexts = [contexts]
            
        for ctx in contexts:
            x = self.fusion(x, ctx)
        
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x

class HierarchicalReasoningCore(nn.Module):
    def __init__(self, dim, num_heads, config):
        super().__init__()
        self.dim = dim
        self.N = config.N_HIGH_CYCLES
        self.T = config.N_LOW_STEPS
        
        self.H_module = HRMModule(dim, num_heads, config.N_HRM_LAYERS, config.DROPOUT)
        self.L_module = HRMModule(dim, num_heads, config.N_HRM_LAYERS, config.DROPOUT)
        
        self.pos_encoder = SinusoidalPositionalEncoding(dim, max_len=2048)
        
        self.q_head = nn.Linear(dim, 1, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.data = torch.tensor([0.0]) # Init to 0.5 probability
        
        self.zH_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.zL_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    
    def init_state(self, batch_size, seq_len, device):
        zH = self.zH_init.expand(batch_size, seq_len, -1).clone()
        zL = self.zL_init.expand(batch_size, seq_len, -1).clone()
        return zH.to(device), zL.to(device)
    
    def forward_segment(self, zH, zL, x, key_padding_mask=None):
        N, T = self.N, self.T
        total_steps = N * T
        
        # 1-step gradient approximation
        with torch.no_grad():
            for i in range(total_steps - 1):
                # Add PE to zL before processing to reinforce order
                zL_pe = self.pos_encoder(zL) 
                # Note: inputs to L_module are (state, contexts). 
                # We should pass zL_pe as state? 
                # HRMModule forwards state through transformer blocks.
                zL = self.L_module(zL_pe, [zH, x], key_padding_mask=key_padding_mask)
                
                if (i + 1) % T == 0:
                     # Add PE to zH
                    zH_pe = self.pos_encoder(zH)
                    zH = self.H_module(zH_pe, [zL], key_padding_mask=key_padding_mask)
        
        zL = self.L_module(self.pos_encoder(zL), [zH.detach(), x], key_padding_mask=key_padding_mask) 
        zH = self.H_module(self.pos_encoder(zH), [zL], key_padding_mask=key_padding_mask)
        
        return zH, zL
    
    def get_q_values(self, zH):
        pooled = zH.mean(dim=1)
        return torch.sigmoid(self.q_head(pooled))

class NanoHRMv3(nn.Module):
    def __init__(self, tokenizer, config):
        super().__init__()
        self.config = config
        
        print(f"Loading Pretrained T5 from {config.TOKENIZER_NAME}...")
        # Load full T5 model (Encoder + Decoder + LM Head)
        # We don't freeze it.
        self.t5_model = T5ForConditionalGeneration.from_pretrained(config.TOKENIZER_NAME)
        
        # We access components directly
        self.shared = self.t5_model.shared # Embeddings
        self.encoder = self.t5_model.encoder
        self.decoder = self.t5_model.decoder
        self.lm_head = self.t5_model.lm_head
        
        d_model = config.D_MODEL # Should be 768 for Base
        
        # HRM Core (Random Init)
        self.hrm_core = HierarchicalReasoningCore(d_model, config.N_HEADS, config)
        
        # Reasoning Pooler: pools zH into K soft-prompt tokens
        self.reasoning_pooler = ReasoningPooler(
            dim=d_model, 
            n_tokens=config.N_REASONING_TOKENS,
            num_heads=config.N_HEADS,
            dropout=config.DROPOUT,
            gate_init=config.REASONING_GATE_INIT,
            force_hrm=getattr(config, 'FORCE_HRM', False)
        )
        
    def freeze_t5(self):
        print("Freezing T5 parameters...")
        for param in self.t5_model.parameters():
            param.requires_grad = False
            
    def unfreeze_t5(self):
        print("Unfreezing T5 parameters...")
        for param in self.t5_model.parameters():
            param.requires_grad = True
        
    def encode(self, input_ids):
        # T5 Encoder expects attention_mask (1=valid, 0=pad)
        # We construct it from input_ids (0 is pad for T5)
        attention_mask = (input_ids != 0).long()
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        memory = encoder_outputs.last_hidden_state # [B, L, D]
        
        # For MHA in HRM, we need boolean mask where True=IGNORED
        # T5 mask is 1=Keep. So we invert.
        # mask is [B, L] long.
        src_mask_bool = (attention_mask == 0) 
        
        return memory, src_mask_bool
    
    def prepare_enhanced_memory(self, memory, zH, src_mask_bool):
        """
        Pool zH into K reasoning tokens and prepend to memory.
        Returns enhanced memory [B, K+L, D] and extended mask [B, K+L].
        """
        # Pool zH -> [B, K, D]
        reasoning_tokens = self.reasoning_pooler(zH, key_padding_mask=src_mask_bool)
        
        # Prepend reasoning tokens to memory: [B, K+L, D]
        enhanced_memory = torch.cat([reasoning_tokens, memory], dim=1)
        
        # Extend mask: reasoning tokens are always valid (False = attend)
        B = memory.size(0)
        K = reasoning_tokens.size(1)
        reasoning_mask = torch.zeros(B, K, dtype=torch.bool, device=memory.device)
        enhanced_mask = torch.cat([reasoning_mask, src_mask_bool], dim=1)
        
        return enhanced_memory, enhanced_mask
    
    def decode(self, memory, labels, src_padding_mask_bool):
        """
        Decode with potentially enhanced memory (can include prepended reasoning tokens).
        This is for TRAINING with teacher forcing (labels are shifted right).
        """
        # Reconstruct standard attention mask from bool mask
        # src_padding_mask_bool is True where Pad, False where Valid
        enc_attn_mask = (~src_padding_mask_bool).long()
        
        # Prepare Decoder Input (shift right for teacher forcing)
        decoder_input_ids = self.t5_model._shift_right(labels)
        
        # Forward Decoder
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=memory,
            encoder_attention_mask=enc_attn_mask,
            return_dict=True
        )
        
        sequence_output = decoder_outputs.last_hidden_state
        
        # LM Head
        logits = self.lm_head(sequence_output)
        return logits
    
    def generate_step(self, memory, decoder_input_ids, src_padding_mask_bool):
        """
        Single step of autoregressive generation (for inference).
        Unlike decode(), this does NOT shift right - use the decoder_input_ids directly.
        """
        # Reconstruct standard attention mask from bool mask
        enc_attn_mask = (~src_padding_mask_bool).long()
        
        # Forward Decoder (NO shift right - decoder_input_ids is already correct)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=memory,
            encoder_attention_mask=enc_attn_mask,
            return_dict=True
        )
        
        sequence_output = decoder_outputs.last_hidden_state
        logits = self.lm_head(sequence_output)
        return logits
    
    def get_metrics(self):
        """
        Return current HRM-related metrics for logging/analysis.
        Call this at validation time to understand model behavior.
        """
        # Reasoning pooler gate (how much reasoning is being used)
        gate_raw = self.reasoning_pooler.gate.item()
        gate_effective = float(torch.tanh(self.reasoning_pooler.gate).item())
        
        # Parameter counts
        total_params = sum(p.numel() for p in self.parameters())
        t5_params = sum(p.numel() for p in self.t5_model.parameters())
        hrm_params = total_params - t5_params  # HRM + Pooler params
        
        return {
            'reasoning_gate_raw': gate_raw,
            'reasoning_gate_effective': gate_effective,  # After tanh: -1 to 1
            'total_params_M': total_params / 1e6,
            't5_params_M': t5_params / 1e6,
            'hrm_params_M': hrm_params / 1e6,
        }
