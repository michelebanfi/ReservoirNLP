import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import math
import numpy as np

# --- CONFIGURATION ---
MODEL_NAME = "t5-small"
BATCH_SIZE = 16 # Reduced batch size as we run decoder multiple times
LEARNING_RATE = 1e-4
EPOCHS = 10
SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ACT Configuration
MAX_ACT_STEPS = 8      # Maximum thinking time
HALT_EXPLORATION = 0.2 # Probability to force exploring more steps during training

print(f"Running ACT-HRM-Sandwich on {DEVICE}...")

class TextLogicDataset(Dataset):
    """
    Generates bAbI-style object tracking stories.
    Focus: Temporal Reasoning & State Updates.
    """
    def __init__(self, tokenizer, size=5000):
        self.tokenizer = tokenizer
        self.size = size
        
        self.people = ["Mary", "John", "Daniel", "Sandra", "Bill", "Lisa"]
        self.locations = ["kitchen", "hallway", "garden", "office", "bedroom", "bathroom"]
        self.actions = ["moved to", "went to", "journeyed to", "travelled to"]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 1. Initialize State
        story_len = random.randint(2, 6) # How many moves?
        # Track where everyone is. Initially unknown.
        current_locs = {} 
        story_lines = []
        
        # 2. Generate Story
        active_people = random.sample(self.people, 3) # Pick 3 actors
        
        for _ in range(story_len):
            person = random.choice(active_people)
            loc = random.choice(self.locations)
            action = random.choice(self.actions)
            
            # Record the logic
            current_locs[person] = loc
            
            # Generate Text
            line = f"{person} {action} the {loc}."
            story_lines.append(line)
            
        # 3. Generate Question
        # Pick a person who actually moved
        target_person = random.choice(list(current_locs.keys()))
        target_loc = current_locs[target_person]
        
        question = f"Question: Where is {target_person}?"
        
        # Format Input: "Story: ... Question: ..."
        input_text = "track state: " + " ".join(story_lines) + " " + question
        target_text = target_loc

        # 4. Tokenize
        source = self.tokenizer(input_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        target = self.tokenizer(target_text, max_length=10, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": source.input_ids.squeeze(),
            "attention_mask": source.attention_mask.squeeze(),
            "labels": target.input_ids.squeeze()
        }

class SudokuDataset(Dataset):
    """
    Generates 4x4 Sudoku puzzles on the fly.
    Input:  "solve sudoku: 1 0 4 3 | 0 0 2 1 | ..." (0 is empty)
    Target: "1 2 4 3 | 3 4 2 1 | ..."
    """
    def __init__(self, tokenizer, size=5000):
        self.tokenizer = tokenizer
        self.size = size
        
        # A single valid 4x4 seed board
        self.base_board = np.array([
            [1, 2, 3, 4],
            [3, 4, 1, 2],
            [2, 1, 4, 3],
            [4, 3, 2, 1]
        ])

    def __len__(self):
        return self.size

    def generate_puzzle(self):
        # 1. Start with valid board
        board = self.base_board.copy()
        
        # 2. Shuffle Digits (Relabel 1->4, 2->1, etc.)
        # Logic adapted from your uploaded build_sudoku_dataset.py
        mapping = np.random.permutation(np.arange(1, 5))
        # Create a lookup table where index 0 is 0 (empty), and index 1..4 map to the shuffled values
        mapper = np.zeros(5, dtype=int)
        mapper[1:] = mapping
        board = mapper[board]
        
        # 3. Shuffle Rows/Cols within Bands (2x2 blocks)
        # 4x4 has 2 bands of 2 rows each
        if random.random() < 0.5:
            # Swap row 0 and 1
            board[[0, 1]] = board[[1, 0]]
        if random.random() < 0.5:
            # Swap row 2 and 3
            board[[2, 3]] = board[[3, 2]]
            
        # Swap the two large bands (rows 0-1 vs rows 2-3)
        if random.random() < 0.5:
            board[[0, 1, 2, 3]] = board[[2, 3, 0, 1]]
            
        # Same for columns
        board = board.T
        if random.random() < 0.5: board[[0, 1]] = board[[1, 0]]
        if random.random() < 0.5: board[[2, 3]] = board[[3, 2]]
        if random.random() < 0.5: board[[0, 1, 2, 3]] = board[[2, 3, 0, 1]]
        board = board.T
        
        # 4. Create Mask (The Puzzle)
        # Remove K random cells to make it a puzzle
        solution = board.copy()
        mask_count = random.randint(4, 8) # Remove 4 to 8 numbers
        mask_indices = np.random.choice(16, mask_count, replace=False)
        flat_board = board.flatten()
        flat_board[mask_indices] = 0 # 0 represents empty
        puzzle = flat_board.reshape(4, 4)
        
        return puzzle, solution

    def __getitem__(self, idx):
        puzzle, solution = self.generate_puzzle()
        
        # Format for T5: "1 0 4 3 | 0 2 ..."
        # We use | to separate rows to help the model understand the grid structure
        def to_str(grid):
            rows = [" ".join(map(str, row)) for row in grid]
            return " | ".join(rows)
            
        input_text = "solve sudoku: " + to_str(puzzle)
        target_text = to_str(solution)

        source = self.tokenizer(input_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        target = self.tokenizer(target_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": source.input_ids.squeeze(),
            "attention_mask": source.attention_mask.squeeze(),
            "labels": target.input_ids.squeeze()
        }

# --- 1. DATASET (Same as before) ---
class ComplexArithmeticDataset(Dataset):
    def __init__(self, tokenizer, size=5000):
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # 33% Linear (Easy), 33% Geometric (Medium), 33% Fibonacci (Hard)
        task_type = random.choice(['linear', 'geometric', 'fibonacci'])
        
        if task_type == 'linear':
            # Sequence: x, x+d, x+2d ...
            start = random.randint(1, 50)
            step = random.randint(1, 10)
            seq = [start + i*step for i in range(4)]
            
        elif task_type == 'geometric':
            # Sequence: x, x*r, x*r^2 ...
            start = random.randint(1, 5)
            ratio = random.randint(2, 3) # Keep numbers small to fit T5 vocab
            seq = [start * (ratio ** i) for i in range(4)]
            
        elif task_type == 'fibonacci':
            # Sequence: a, b, a+b, a+2b+a ...
            a = random.randint(1, 10)
            b = random.randint(1, 10)
            seq = [a, b]
            for _ in range(2):
                seq.append(seq[-1] + seq[-2])
        
        # Format Input
        input_text = "predict next: " + " , ".join(map(str, seq[:-1]))
        target_text = str(seq[-1])

        # Tokenize
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

    def forward(self, x, key_padding_mask=None):
        # Pass the mask to attention
        # key_padding_mask: [Batch, SeqLen] where True = PAD (ignore)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
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

    def forward_step(self, z_state, x_input, mask=None):
        # One Step of Thinking
        context = z_state + x_input
        z_next = self.norm_fusion(z_state + context)
        
        # Invert mask for PyTorch MultiheadAttention if necessary
        # T5 mask: 1=Valid, 0=Pad. 
        # PyTorch key_padding_mask: True=Pad, False=Valid.
        padding_mask = (mask == 0) if mask is not None else None
        
        z_next = self.Block(z_next, key_padding_mask=padding_mask)
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
        
        with torch.no_grad():
            enc_out = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            
        original_inputs = enc_out
        x = self.hrm.input_adapter(enc_out)
        z = self.hrm.state_init.expand(B, x.shape[1], x.shape[2])
        x = x + self.hrm.pos_embed[:, :x.shape[1], :]
        
        step_outputs = []
        steps_to_run = MAX_ACT_STEPS
        
        for step in range(steps_to_run):
            # PASS THE MASK HERE
            z = self.hrm.forward_step(z, x, mask=attention_mask)
            
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
        
        with torch.no_grad():
            # 1. Encode
            enc_out = self.t5.encoder(inputs.input_ids).last_hidden_state
            
            # 2. ACT Loop
            original_inputs = enc_out
            x = self.hrm.input_adapter(enc_out)
            z = self.hrm.state_init.expand(1, x.shape[1], x.shape[2])
            x = x + self.hrm.pos_embed[:, :x.shape[1], :]
            
            # Use the MASK (Fixed logic)
            mask = inputs.attention_mask
            
            final_step = 0
            for step in range(MAX_ACT_STEPS):
                z = self.hrm.forward_step(z, x, mask=mask)
                q_logits = self.hrm.predict_q(z)
                
                halt = q_logits[0, 0]
                cont = q_logits[0, 1]
                
                if halt > cont:
                    final_step = step + 1
                    break
                final_step = step + 1
            
            # 3. Decode
            z_out = self.hrm.output_adapter(z)
            final_vec = original_inputs + z_out
            
            from transformers.modeling_outputs import BaseModelOutput
            dummy = BaseModelOutput(last_hidden_state=final_vec)
            
            # FIX: Increase max_length to 64 to see the whole board
            gen = self.t5.generate(encoder_outputs=dummy, max_length=64)
            
        return self.tokenizer.decode(gen[0], skip_special_tokens=True) + f" (Steps: {final_step})"

# --- 4. TRAINING WITH Q-LOSS ---
def train():
    # 1. SETUP
    model = NeuroSymbolicACT(MODEL_NAME).to(DEVICE)
    dataset = TextLogicDataset(model.tokenizer, size=5000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Separate LRs for Stability
    optimizer = torch.optim.AdamW([
        {'params': model.hrm.q_head.parameters(), 'lr': 5e-4},
        {'params': [p for n, p in model.hrm.named_parameters() if 'q_head' not in n], 'lr': 1e-4}
    ])

    model.train()
    print("\nTraining TextReasoning-ACT (with Mask Fix & Warmup)...")
    
    # 2. DEFINE A VALID TEST PUZZLE
    # A simple 4x4 puzzle with one missing number (top-left should be 1)
    # Row 1: . 2 3 4 | Row 2: 3 4 1 2 | Row 3: 2 1 4 3 | Row 4: 4 3 2 1
    test_puzzle = "solve sudoku: 0 2 3 4 | 3 4 1 2 | 2 1 4 3 | 4 3 2 1"
    
    for epoch in range(EPOCHS):
        # 3. WARMUP LOGIC
        # First 5 epochs: Force 6 steps of thinking, Ignore Q-Loss.
        # This helps the HRM learn "how to think" before "when to stop".
        is_warmup = epoch < 5
        q_loss_weight = 0.0 if is_warmup else 1.0
        
        total_lm_loss = 0
        total_q_loss = 0
        
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE) # T5 Mask (1=Valid, 0=Pad)
            labels = batch["labels"].to(DEVICE)
            
            # Forward Pass (PASS THE MASK!)
            # Ensure your model.forward() accepts and uses the mask as discussed!
            step_results = model(input_ids, mask, labels=labels)
            
            # --- LOSS CALCULATION ---
            
            # A. Reasoning Loss (Language Model)
            final_step = step_results[-1]
            lm_loss = final_step["lm_loss"]
            
            # B. Q-Learning Loss (Halting)
            q_losses = []
            if not is_warmup:
                # Standard Q-Learning with Gamma
                next_value = step_results[-1]["is_correct"]
                GAMMA = 0.9
                
                for i in reversed(range(len(step_results) - 1)):
                    curr = step_results[i]
                    # Target = Max(Immediate Reward, Discounted Future)
                    discounted_future = next_value * GAMMA
                    target_q = torch.maximum(curr["is_correct"], discounted_future).detach()
                    
                    # Losses
                    halt_loss = F.binary_cross_entropy_with_logits(curr["q_logits"][:, 0], curr["is_correct"])
                    cont_loss = F.binary_cross_entropy_with_logits(curr["q_logits"][:, 1], discounted_future)
                    
                    q_losses.append(halt_loss + cont_loss)
                    
                    # Update Value for next iteration (backwards)
                    with torch.no_grad():
                        pred_val = torch.maximum(torch.sigmoid(curr["q_logits"][:, 0]), torch.sigmoid(curr["q_logits"][:, 1]))
                        next_value = pred_val
            
            avg_q_loss = torch.stack(q_losses).mean() if q_losses else torch.tensor(0.0).to(DEVICE)
            
            # C. Drift Regularization (Tiny)
            # Prevent the "Empty Output" bug
            z_drift = 0
            for res in step_results:
                z_drift += (res['z_final'] - step_results[0]['z_final']).norm(p=2) # deviation from first thought
            drift_loss = z_drift / (len(step_results) * BATCH_SIZE)

            # Total Loss
            loss = lm_loss + (q_loss_weight * avg_q_loss) + (0.001 * drift_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_lm_loss += lm_loss.item()
            total_q_loss += avg_q_loss.item()
            
        print(f"Epoch {epoch+1} | LM Loss: {total_lm_loss/len(dataloader):.4f} | Q Loss: {total_q_loss/len(dataloader):.4f}")
        
        # 4. RELEVANT TEST
        # We now test on a Placement string
        test_text = "track state: Mary went to the hallway. Mary moved to the office. Question: Where is Mary?"
        hard_text = "track state: John went to the garden. Mary moved to the kitchen. John moved to the office. Question: Where is John?"

        print(f"  Test: {model.generate(test_text)}")
        print(f"  Hard: {model.generate(hard_text)}")

        

if __name__ == "__main__":
    train()
