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

class GatedResidualAdapter(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.act = nn.Tanh()
        # Initialize gate to 0 to start as Identity (T5 baseline)
        self.gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, memory, reasoning):
        # memory: [B, L, D]
        # reasoning: [B, L, D] (zH)
        # Returns: memory + gate * tanh(proj(reasoning))
        return memory + self.gate * self.act(self.proj(reasoning))

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
        
        self.q_head = nn.Linear(dim, 2, bias=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.data = torch.tensor([-2.0, 0.0])
        
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
                zL = self.L_module(zL, [zH, x], key_padding_mask=key_padding_mask)
                if (i + 1) % T == 0:
                    zH = self.H_module(zH, [zL], key_padding_mask=key_padding_mask)
        
        zL = self.L_module(zL, [zH.detach(), x], key_padding_mask=key_padding_mask) 
        zH = self.H_module(zH, [zL], key_padding_mask=key_padding_mask)
        
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
        
        # Gated Residual Adapter (Fix for Concatenation/Addition issues)
        self.adapter = GatedResidualAdapter(d_model)
        
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
    
    def decode(self, memory, labels, src_padding_mask_bool):
        # Memory here is Enhanced Memory (Encoder Out + zH)
        
        # T5 Decoder needs:
        # - input_ids (shifted labels handled by T5ForConditionalGeneration usually, but here we access decoder directly)
        # - encoder_hidden_states (= memory)
        # - encoder_attention_mask (= src_padding_mask? No, T5 expects 1/0 int mask)
        
        # Reconstruct standard attention mask from bool mask
        # src_padding_mask_bool is True where Pad.
        # encoder_attention_mask should be 1 where Valid (False).
        enc_attn_mask = (~src_padding_mask_bool).long()
        
        # Prepare Decoder Input
        # labels are [-100, ..., EOS]
        # We need to Shift Right.
        # T5ForConditionalGeneration does this in forward() via `_shift_right`.
        # We should use that helper or replicate.
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
    
    # Helper for exposing components for T5 native methods if strictly needed
    # but we are wrapping manually.
