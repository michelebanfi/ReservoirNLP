import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import math

# --- CONFIGURATION ---
MODEL_NAME = "t5-small"
BATCH_SIZE = 16 # Reduced batch size as we run decoder multiple times
LEARNING_RATE = 1e-4
EPOCHS = 10
SEQ_LEN = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ACT Configuration
MAX_ACT_STEPS = 8      # Maximum thinking time
HALT_EXPLORATION = 0.2 # Probability to force exploring more steps during training

print(f"Running ACT-HRM-Sandwich on {DEVICE}...")

# --- 1. DATASET (Same as before) ---
class ArithmeticReasoningDataset(Dataset):
    def __init__(self, tokenizer, size=1000):
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        start = random.randint(1, 50)
        step = random.randint(1, 10)
        length = 4
        sequence = [start + i*step for i in range(length)]
        
        input_text = "predict next: " + " , ".join(map(str, sequence[:-1]))
        target_text = str(sequence[-1])

        source = self.tokenizer(input_text, max_length=SEQ_LEN, padding="max_length", truncation=True, return_tensors="pt")
        target = self.tokenizer(target_text, max_length=10, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": source.input_ids.squeeze(),
            "attention_mask": source.attention_mask.squeeze(),
            "labels": target.input_ids.squeeze()
        }

# --- 2. THE ACT-ENABLED CORE ---
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
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, int(dim * 4 * 2 / 3)) 

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class ACTReasoningCore(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        
        # Adapters
        self.input_adapter = nn.Sequential(nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU())
        self.output_adapter = nn.Linear(dim, dim)
        nn.init.zeros_(self.output_adapter.weight)
        nn.init.zeros_(self.output_adapter.bias)

        # Q-Head (The "Manager")
        # Takes the global state (mean of sequence) and predicts [Halt, Continue]
        self.q_head = nn.Linear(dim, 2)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5) # Bias towards continuing initially

        # Reasoning Blocks (H/L simplified to 1 layer each for speed in ACT loop)
        self.Block = HRMBlock(dim, num_heads)
        self.norm_fusion = nn.LayerNorm(dim)

        # Init States
        self.state_init = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, SEQ_LEN, dim) * 0.02)

    def forward_step(self, z_state, x_input):
        # One Step of Thinking
        # Input Injection
        context = z_state + x_input
        z_next = self.norm_fusion(z_state + context)
        z_next = self.Block(z_next)
        return z_next

    def predict_q(self, z_state):
        # Predict Q-values from the first token's state (representing the whole thought)
        # z_state: [Batch, Seq, Dim] -> we take [Batch, 0, Dim]
        return self.q_head(z_state[:, 0, :])

# --- 3. THE ACT SANDWICH ---
class NeuroSymbolicACT(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(base_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_name)
        
        for param in self.t5.parameters():
            param.requires_grad = False
            
        hidden_dim = self.t5.config.d_model
        self.hrm = ACTReasoningCore(hidden_dim, num_heads=8)
        
    def forward(self, input_ids, attention_mask, labels=None):
        B = input_ids.shape[0]
        
        # 1. Encode
        with torch.no_grad():
            enc_out = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            
        # 2. Initialize ACT
        original_inputs = enc_out
        x = self.hrm.input_adapter(enc_out)
        z = self.hrm.state_init.expand(B, x.shape[1], x.shape[2])
        x = x + self.hrm.pos_embed[:, :x.shape[1], :]
        
        # ACT Storage
        step_outputs = []
        
        # 3. Dynamic Loop
        # In training, we run ALL steps to gather data for Q-learning
        # In inference, we stop when Q says stop
        
        steps_to_run = MAX_ACT_STEPS
        
        for step in range(steps_to_run):
            # A. Think
            z = self.hrm.forward_step(z, x)
            
            # B. Check Halting (Q-Values)
            q_logits = self.hrm.predict_q(z) # [Batch, 2]
            
            # C. Generate Output Candidate
            z_out = self.hrm.output_adapter(z)
            final_vector = original_inputs + z_out
            
            # D. If Training, check correctness immediately using T5 Decoder
            lm_loss = None
            is_correct = None
            
            if labels is not None:
                # We perform a "Virtual Decode" to see if this thought is correct
                # This uses the Frozen Decoder to get the CrossEntropy Loss
                decoder_out = self.t5(
                    encoder_outputs=(final_vector,),
                    labels=labels,
                    decoder_input_ids=self.t5._shift_right(labels)
                )
                lm_loss = decoder_out.loss
                
                # Check correctness (Exact Match on Argmax)
                # This is the "Reward" for the Q-Head
                logits = decoder_out.logits # [B, Seq, Vocab]
                preds = logits.argmax(dim=-1)
                
                # Mask ignored labels (-100)
                mask = labels != -100
                correct_tokens = (preds == labels) & mask
                # Sequence is correct if ALL tokens match
                seq_correct = (correct_tokens.sum(dim=-1) == mask.sum(dim=-1))
                is_correct = seq_correct.float() # 1.0 or 0.0

            step_outputs.append({
                "q_logits": q_logits,
                "lm_loss": lm_loss,
                "is_correct": is_correct,
                "z_final": final_vector
            })
            
            # E. Inference Halting Logic
            if labels is None:
                halt_score = q_logits[:, 0]
                cont_score = q_logits[:, 1]
                if (halt_score > cont_score).all(): # Simple: Halt if batch agrees (or handle batching properly)
                    break 

        return step_outputs

    def generate(self, input_text):
        inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)
        
        # Same init logic
        with torch.no_grad():
            enc_out = self.t5.encoder(inputs.input_ids).last_hidden_state
            original_inputs = enc_out
            x = self.hrm.input_adapter(enc_out)
            z = self.hrm.state_init.expand(1, x.shape[1], x.shape[2])
            x = x + self.hrm.pos_embed[:, :x.shape[1], :]
            
            for step in range(MAX_ACT_STEPS):
                z = self.hrm.forward_step(z, x)
                q_logits = self.hrm.predict_q(z)
                
                halt = q_logits[0, 0]
                cont = q_logits[0, 1]
                
                # Stop?
                if halt > cont:
                    break
            
            # Decode final state
            z_out = self.hrm.output_adapter(z)
            final_vec = original_inputs + z_out
            
            from transformers.modeling_outputs import BaseModelOutput
            dummy = BaseModelOutput(last_hidden_state=final_vec)
            gen = self.t5.generate(encoder_outputs=dummy, max_length=10)
            
        return self.tokenizer.decode(gen[0], skip_special_tokens=True) + f" (Steps: {step+1})"

# --- 4. TRAINING WITH Q-LOSS ---
def train():
    model = NeuroSymbolicACT(MODEL_NAME).to(DEVICE)
    dataset = ArithmeticReasoningDataset(model.tokenizer, size=2000) # Increased size
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Separate Learning Rates: Q-Head needs to learn FASTER than the Core to catch up
    optimizer = torch.optim.AdamW([
        {'params': model.hrm.q_head.parameters(), 'lr': 5e-4}, # Boosted Q-LR
        {'params': [p for n, p in model.hrm.named_parameters() if 'q_head' not in n], 'lr': 1e-4}
    ])

    model.train()
    print("\nTraining ACT-HRM with Q-Learning Bootstrapping...")
    
    for epoch in range(EPOCHS):
        total_lm_loss = 0
        total_q_loss = 0
        total_halt_correct = 0 # Track how often it halts correctly
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            # Forward Pass
            step_results = model(input_ids, mask, labels=labels)
            
            # --- Q-LEARNING LOSS CALCULATION ---
            batch_loss = 0
            
            # 1. LM Loss (Only from the FINAL step to save memory/compute)
            final_step = step_results[-1]
            lm_loss = final_step["lm_loss"]
            
            # 2. Temporal Difference (TD) Learning for Q-Values
            q_losses = []
            
            # We iterate BACKWARDS: The reward propagates from the end to the start
            # Target for the LAST step is just the correctness (1.0 or 0.0)
            next_value = step_results[-1]["is_correct"] 
            
            for i in reversed(range(len(step_results) - 1)):
                curr_res = step_results[i]
                
                # Current predictions
                halt_logit = curr_res["q_logits"][:, 0]
                cont_logit = curr_res["q_logits"][:, 1]
                
                # Target Construction:
                # If I Halt now, the value is "is_correct" (Immediate Reward)
                # If I Continue, the value is "next_value" (Future Reward)
                # We want the model to predict the MAX of these two options
                
                # Calculate Target Q-Value (Bellman Equation)
                # target = max(is_correct, next_value)
                # Note: We use Sigmoid/BCE logic here for stability
                target_q = torch.maximum(curr_res["is_correct"], next_value).detach()
                
                # If Target is 1.0 (Success is possible):
                # We want max(halt_logit, cont_logit) to be high.
                # Specifically, if is_correct=1, Halt Logit should be high.
                # If is_correct=0 but next_value=1, Cont Logit should be high.
                
                # Simplification for Stability:
                # 1. Train Halt Logit against Current Correctness
                halt_loss = F.binary_cross_entropy_with_logits(halt_logit, curr_res["is_correct"])
                
                # 2. Train Continue Logit against Future Value (Bootstrapping)
                cont_loss = F.binary_cross_entropy_with_logits(cont_logit, next_value)
                
                q_losses.append(halt_loss + cont_loss)
                
                # Update next_value for the previous step
                # The value of being at step `i` is the best of (Halting, Continuing)
                # We use the Model's OWN prediction for bootstrapping (Self-Consistency)
                with torch.no_grad():
                     best_future = torch.maximum(torch.sigmoid(halt_logit), torch.sigmoid(cont_logit))
                     next_value = best_future
            
            avg_q_loss = torch.stack(q_losses).mean() if q_losses else torch.tensor(0.0).to(DEVICE)
            
            # Total Loss
            loss = lm_loss + (1.0 * avg_q_loss) # Boost Q-Loss weight
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_lm_loss += lm_loss.item()
            total_q_loss += avg_q_loss.item()
            
        print(f"Epoch {epoch+1} | LM Loss: {total_lm_loss/len(dataloader):.3f} | Q Loss: {total_q_loss/len(dataloader):.3f}")
        
        # Test
        q = "predict next: 10 , 20 , 30"
        print(f"  {q} -> {model.generate(q)}")

if __name__ == "__main__":
    train()