import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import math

# --- CONFIGURATION ---
MODEL_NAME = "t5-small"
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 10
SEQ_LEN = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HRM Hyperparameters (The "Brain" Structure)
H_CYCLES = 2  # How many "slow thoughts" to think
L_CYCLES = 2  # How many "fast thoughts" per slow thought
HIDDEN_DIM = 512 # Matches T5-Small
NUM_HEADS = 8

print(f"Running HRM-Sandwich on {DEVICE}...")

# --- 1. SIMPLIFIED DATASET (Arithmetic) ---
class ArithmeticReasoningDataset(Dataset):
    """
    Input: "predict next: 10 , 20 , 30" -> Target: "40"
    Now with COMMAS to force T5 to see separate numbers.
    """
    def __init__(self, tokenizer, size=2000):
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start = random.randint(1, 50)
        step = random.randint(1, 10)
        length = 4
        
        sequence = [start + i*step for i in range(length)]
        input_nums = sequence[:-1]
        target_num = sequence[-1]
        
        # Crucial Fix: Add spaces and commas so tokenizer sees [10, ',', 20]
        input_text = "predict next: " + " , ".join(map(str, input_nums))
        target_text = str(target_num)

        source = self.tokenizer(input_text, max_length=SEQ_LEN, padding="max_length", truncation=True, return_tensors="pt")
        target = self.tokenizer(target_text, max_length=10, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": source.input_ids.squeeze(),
            "attention_mask": source.attention_mask.squeeze(),
            "labels": target.input_ids.squeeze()
        }

# --- 2. THE ADAPTED HRM ARCHITECTURE ---

class RMSNorm(nn.Module):
    """Simple RMSNorm from the repo, adapted for standard PyTorch"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight

class SwiGLU(nn.Module):
    """The activation function used in Llama and the HRM repo"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class HRMBlock(nn.Module):
    """A single reasoning block (Attention + MLP)"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        # Expansion factor 4 is standard
        self.mlp = SwiGLU(dim, int(dim * 4 * 2 / 3)) 

    def forward(self, x, context=None):
        # Self-Attention
        # Note: If context is provided (Cross-Attention), we could use it here.
        # But the HRM repo uses injection via addition: z + input
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class HierarchicalReasoningCore(nn.Module):
    def __init__(self, dim, num_heads, h_cycles, l_cycles):
        super().__init__()
        self.dim = dim
        self.h_cycles = h_cycles
        self.l_cycles = l_cycles

        # ADAPTERS
        self.input_adapter = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        self.output_adapter = nn.Linear(dim, dim)

        # REASONING BLOCKS
        self.H_Block = HRMBlock(dim, num_heads)
        self.L_Block = HRMBlock(dim, num_heads)

        # CONTROL RODS (New Normalization Layers)
        self.norm_fusion_L = nn.LayerNorm(dim) # Stabilizes (z_L + context)
        self.norm_fusion_H = nn.LayerNorm(dim) # Stabilizes (z_H + z_L)

        # Init States
        self.H_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.L_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, SEQ_LEN, dim) * 0.02)

        # Zero-Init Output
        nn.init.zeros_(self.output_adapter.weight)
        nn.init.zeros_(self.output_adapter.bias)

    def forward(self, input_embeddings):
        B, S, D = input_embeddings.shape
        
        # 1. Adapt & Residual Base
        original_inputs = input_embeddings 
        x = self.input_adapter(input_embeddings)
        
        # 2. Init States
        z_H = self.H_init.expand(B, S, D)
        z_L = self.L_init.expand(B, S, D)
        x = x + self.pos_embed[:, :S, :]

        # 3. The Stabilized Loop
        for h_step in range(self.h_cycles):
            for l_step in range(self.l_cycles):
                # Input Injection with Stabilization
                # instead of just adding, we normalize the mix
                context = z_H + x 
                z_L_input = self.norm_fusion_L(z_L + context) 
                
                # Update Fast Stream
                z_L = self.L_Block(z_L_input)

            # Slow Stream Update with Stabilization
            z_H_input = self.norm_fusion_H(z_H + z_L)
            z_H = self.H_Block(z_H_input)

        # 4. Output with Skip
        z_out = self.output_adapter(z_H)
        
        # Scale the learned delta down initially to prevent shock
        # (The model has to learn to increase this scaling factor)
        return original_inputs + z_out

# --- 3. THE SANDWICH WRAPPER ---
class NeuroSymbolicSandwich(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(base_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_name)
        
        # FREEZE T5
        for param in self.t5.parameters():
            param.requires_grad = False
            
        # INSERT HRM
        hidden_dim = self.t5.config.d_model
        self.hrm = HierarchicalReasoningCore(hidden_dim, NUM_HEADS, H_CYCLES, L_CYCLES)
        
        print(f"T5 Frozen. Trainable HRM Params: {sum(p.numel() for p in self.hrm.parameters() if p.requires_grad)}")

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. ENCODE (Frozen)
        with torch.no_grad():
            encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = encoder_outputs.last_hidden_state
        
        # 2. REASON (Trainable HRM)
        reasoned_states = self.hrm(hidden_states)
        
        # 3. DECODE (Frozen)
        outputs = self.t5(
            encoder_outputs=(reasoned_states,), 
            labels=labels,
            decoder_input_ids=self.t5._shift_right(labels) if labels is not None else None
        )
        return outputs, reasoned_states, hidden_states

    def generate(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            enc_out = self.t5.encoder(inputs.input_ids).last_hidden_state
            # Run the Thinking Loop
            reasoned_state = self.hrm(enc_out)
            
            # Helper for generation
            from transformers.modeling_outputs import BaseModelOutput
            dummy_enc_out = BaseModelOutput(last_hidden_state=reasoned_state)
            
            generated_ids = self.t5.generate(encoder_outputs=dummy_enc_out, max_length=10)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# --- 4. TRAINING LOOP WITH ANCHOR LOSS ---
def train():
    model = NeuroSymbolicSandwich(MODEL_NAME).to(DEVICE)
    dataset = ArithmeticReasoningDataset(model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = torch.optim.AdamW(model.hrm.parameters(), lr=LEARNING_RATE)
    mse_loss_fn = nn.MSELoss()

    model.train()
    
    print("\nStarting Training with Latent Anchor Loss...")
    for epoch in range(EPOCHS):
        total_loss = 0
        total_mse = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            # Forward
            outputs, z_reasoned, z_input = model(input_ids, mask, labels=labels)
            
            # 1. Semantic Loss (Can the decoder read it?)
            lm_loss = outputs.loss 
            
            # 2. Anchor Loss (Does the thought stay grounded?)
            # We force the reasoning output (z_reasoned) to not drift too far in magnitude/variance from input
            # This helps the Frozen Decoder understand the vectors.
            anchor_loss = mse_loss_fn(z_reasoned, z_input) 
            
            # Combined Loss
            loss = lm_loss + (0.01 * anchor_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += lm_loss.item()
            total_mse += anchor_loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | LM Loss: {avg_loss:.4f} | Anchor Loss: {total_mse/len(dataloader):.4f}")
        
        # Verification
        test_q = "predict next: 10 , 20 , 30"
        print(f"  Input: {test_q} -> HRM Output: {model.generate(test_q)}")

if __name__ == "__main__":
    train()