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
LEARNING_RATE = 3e-4
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
        # NOTE: Use max_length=64 for labels to match SudokuDataset for batching compatibility
        source = self.tokenizer(input_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        target = self.tokenizer(target_text, max_length=64, padding="max_length", truncation=True, return_tensors="pt")

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

        # NOTE: Use max_length=64 for labels (enough for 4x4 board, matches TextLogicDataset)
        source = self.tokenizer(input_text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        target = self.tokenizer(target_text, max_length=64, padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": source.input_ids.squeeze(),
            "attention_mask": source.attention_mask.squeeze(),
            "labels": target.input_ids.squeeze()
        }

class HeterogeneousDataset(Dataset):
    def __init__(self, tokenizer, size=10000):
        self.tokenizer = tokenizer
        self.size = size
        self.tasks = ['arithmetic', 'sort', 'reverse', 'logic']

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Randomly select a task type to force the model to adapt
        task = random.choice(self.tasks)
        
        if task == 'arithmetic':
            return self.gen_arithmetic()
        elif task == 'sort':
            return self.gen_sort()
        elif task == 'reverse':
            return self.gen_reverse()
        elif task == 'logic':
            return self.gen_logic()

    # --- TASK 1: Sequential Arithmetic ---
    # Forces the model to hold a value in memory
    def gen_arithmetic(self):
        ops = ['+', '-', '*']
        a = random.randint(1, 10)
        b = random.randint(1, 10)
        c = random.randint(1, 5)
        
        op1 = random.choice(ops)
        op2 = random.choice(ops)
        
        # Example: (3 + 5) * 2
        input_text = f"calc: ({a} {op1} {b}) {op2} {c}"
        
        # Python eval handles the logic (Ground Truth)
        try:
            res = eval(f"({a} {op1} {b}) {op2} {c}")
        except:
            res = 0
            
        target_text = str(res)
        return self.format(input_text, target_text)

    # --- TASK 2: List Sorting ---
    # Forces global comparison logic
    def gen_sort(self):
        length = random.randint(3, 6)
        nums = [random.randint(0, 99) for _ in range(length)]
        input_text = "sort: " + " ".join(map(str, nums))
        
        sorted_nums = sorted(nums)
        target_text = " ".join(map(str, sorted_nums))
        return self.format(input_text, target_text)

    # --- TASK 3: Reversal ---
    # Forces positional understanding
    def gen_reverse(self):
        length = random.randint(4, 8)
        # Use random letters
        chars = [chr(random.randint(97, 122)) for _ in range(length)]
        input_text = "reverse: " + " ".join(chars)
        
        target_text = " ".join(chars[::-1])
        return self.format(input_text, target_text)

    # --- TASK 4: Logic (Your Existing Logic) ---
    def gen_logic(self):
        people = ["Mary", "John", "Daniel", "Sandra"]
        locs = ["kitchen", "garden", "office", "bedroom"]
        
        # Simple 2-step story
        p = random.choice(people)
        l1 = random.choice(locs)
        l2 = random.choice(locs)
        
        story = f"{p} went to {l1}. {p} moved to {l2}."
        question = f"Where is {p}?"
        
        input_text = f"track: {story} {question}"
        target_text = l2
        return self.format(input_text, target_text)

    def format(self, input_text, target_text):
        source = self.tokenizer(input_text, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
        target = self.tokenizer(target_text, max_length=16, padding="max_length", truncation=True, return_tensors="pt")
        
        return {
            "input_ids": source.input_ids.squeeze(),
            "attention_mask": source.attention_mask.squeeze(), # 1=Valid
            "labels": target.input_ids.squeeze()
        }


class MixedReasoningDataset(Dataset):
    """
    Combines TextLogicDataset and SudokuDataset for multi-task training.
    Randomly samples from both datasets to train on diverse reasoning tasks.
    """
    def __init__(self, tokenizer, size=5000, text_logic_ratio=0.5):
        self.tokenizer = tokenizer
        self.size = size
        self.text_logic_ratio = text_logic_ratio  # Probability of sampling TextLogic vs Sudoku
        
        # Initialize both sub-datasets
        self.text_logic = TextLogicDataset(tokenizer, size=size)
        self.sudoku = SudokuDataset(tokenizer, size=size)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Randomly choose which dataset to sample from
        if random.random() < self.text_logic_ratio:
            return self.text_logic[idx]
        else:
            return self.sudoku[idx]

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

class ContextualAdapter(nn.Module):
    """
    Reads a window of tokens to understand local grammar 
    (e.g., 'moved to garden' vs 'moved from garden').
    """
    def __init__(self, dim):
        super().__init__()
        # Kernel size 3 looks at [Prev, Current, Next]
        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1)
        self.norm = nn.LayerNorm(dim)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        # x: [Batch, Seq, Dim]
        # Conv1d expects [Batch, Dim, Seq]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return self.gelu(self.norm(x))

class ACTReasoningCore(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        
        # Adapters
        self.input_adapter = ContextualAdapter(dim)
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [Batch, Seq, Dim]
        return x + self.pe[:x.size(1), :].unsqueeze(0)

class NanoACT(nn.Module):
    def __init__(self, tokenizer, d_model=256, n_heads=4, dropout=0.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        vocab_size = tokenizer.vocab_size
        
        # 1. Embeddings (Learned from scratch)
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # 2. Mini Encoder (Just to parse syntax)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. YOUR HRM CORE (The Brain)
        self.hrm = ACTReasoningCore(dim=d_model, num_heads=n_heads)
        
        # 4. Mini Decoder (To translate thoughts to text)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        
        # 5. Output Head
        self.lm_head = nn.Linear(d_model, vocab_size)

    def create_masks(self, src, tgt):
        # Src Mask: [Batch, SrcSeq] (True = Pad)
        src_key_padding_mask = (src == self.pad_token_id)
        
        # Tgt Mask: [Batch, TgtSeq]
        tgt_key_padding_mask = (tgt == self.pad_token_id)
        
        # Tgt Causal Mask: [TgtSeq, TgtSeq] (Standard triangular mask)
        sz = tgt.size(1)
        tgt_mask = torch.triu(torch.ones(sz, sz, device=tgt.device) * float('-inf'), diagonal=1)
        
        return src_key_padding_mask, tgt_key_padding_mask, tgt_mask

    def forward(self, input_ids, attention_mask=None, labels=None):
        # NOTE: attention_mask from T5 tokenizer is 1 for Valid, 0 for Pad.
        # PyTorch Transformer expects "True" for Pad. So we invert it or recalculate.
        
        # A. Encode
        src_emb = self.dropout(self.pos_encoder(self.embedding(input_ids)))
        # Invert T5 mask: 1->False (Valid), 0->True (Pad)
        src_padding_mask = (input_ids == self.pad_token_id)
        
        memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
        
        # B. ACT REASONING LOOP (The Sandwich)
        # We process the 'memory' from the encoder through the ACT loop
        
        # Prepare HRM inputs
        B, L, D = memory.shape
        x_in = self.hrm.input_adapter(memory)
        z = self.hrm.state_init.expand(B, L, D)
        x_in = x_in + self.hrm.pos_embed[:, :L, :]
        
        step_outputs = []
        
        # The Decoder needs a target to calculate loss
        # If training, use labels. If inference, we can't fully run decoder loop here easily 
        # without beam search, so we usually only run forward pass if labels exist.
        
        if labels is not None:
            # Shift labels for teaching forcing: Input to decoder is [SOS, ...], Target is [..., EOS]
            # T5 labels usually have -100 for ignored, we need to fix that for embedding lookup
            decoder_input = labels.clone()
            decoder_input[decoder_input == -100] = self.pad_token_id 
            
            # Shift right (naive implementation for demo)
            # In real T5, the decoder_start_token_id is usually 0
            sos_token = torch.full((B, 1), 0, device=labels.device, dtype=torch.long) # Assuming 0 is pad/start
            decoder_input = torch.cat([sos_token, decoder_input[:, :-1]], dim=1)
            
            tgt_emb = self.dropout(self.pos_encoder(self.embedding(decoder_input)))
            tgt_padding_mask = (decoder_input == self.pad_token_id)
            tgt_causal_mask = torch.triu(torch.ones(decoder_input.size(1), decoder_input.size(1), device=labels.device) * float('-inf'), diagonal=1)

            # ACT Loop
            for step in range(MAX_ACT_STEPS):
                # 1. Think
                z = self.hrm.forward_step(z, x_in, mask=(~src_padding_mask).long()) # mask: 1=valid
                
                # 2. Q-Head (Halting)
                q_logits = self.hrm.predict_q(z)
                
                # 3. Create "Thought Vectors" for Decoder
                z_out = self.hrm.output_adapter(z)
                enhanced_memory = memory + z_out 
                
                # 4. Decode (Virtual Attempt)
                # We feed the "enhanced memory" to the decoder
                dec_out = self.decoder(
                    tgt=tgt_emb,
                    memory=enhanced_memory,
                    tgt_mask=tgt_causal_mask,
                    tgt_key_padding_mask=tgt_padding_mask,
                    memory_key_padding_mask=src_padding_mask
                )
                
                logits = self.lm_head(dec_out) # [B, TgtLen, Vocab]
                
                # 5. Calculate Loss for this step
                # Flatten for CrossEntropy: [B*T, Vocab] vs [B*T]
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                lm_loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # 6. Reward Calculation (for Q-Learning)
                preds = logits.argmax(dim=-1)
                mask = labels != -100
                correct = (preds == labels) & mask
                seq_correct = (correct.sum(dim=-1) == mask.sum(dim=-1)).float()
                
                step_outputs.append({
                    "q_logits": q_logits,
                    "lm_loss": lm_loss,
                    "is_correct": seq_correct,
                    "z_final": enhanced_memory
                })
        
        return step_outputs

    def generate(self, input_text, max_len=64):
        # Simple Greedy Decoding for Inference
        self.eval()
        inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)
        input_ids = inputs.input_ids
        
        # Encode & Think
        with torch.no_grad():
            src_emb = self.dropout(self.pos_encoder(self.embedding(input_ids)))
            src_padding_mask = (input_ids == self.pad_token_id)
            memory = self.encoder(src_emb, src_key_padding_mask=src_padding_mask)
            
            # ACT Loop (Inference)
            B, L, D = memory.shape
            x_in = self.hrm.input_adapter(memory)
            z = self.hrm.state_init.expand(B, L, D)
            x_in = x_in + self.hrm.pos_embed[:, :L, :]
            
            final_step = 0
            for step in range(MAX_ACT_STEPS):
                z = self.hrm.forward_step(z, x_in, mask=(~src_padding_mask).long())
                q_logits = self.hrm.predict_q(z)
                if q_logits[0, 0] > q_logits[0, 1]: # Halt > Cont
                    final_step = step + 1
                    break
                final_step = step + 1
            
            # Prepare Decoder Memory
            z_out = self.hrm.output_adapter(z)
            enhanced_memory = memory + z_out
            
            # Decode Loop
            curr_tokens = torch.tensor([[0]], device=DEVICE) # Start token
            for _ in range(max_len):
                tgt_emb = self.dropout(self.pos_encoder(self.embedding(curr_tokens)))
                tgt_causal_mask = torch.triu(torch.ones(curr_tokens.size(1), curr_tokens.size(1), device=DEVICE) * float('-inf'), diagonal=1)
                
                dec_out = self.decoder(
                    tgt=tgt_emb,
                    memory=enhanced_memory,
                    tgt_mask=tgt_causal_mask,
                    memory_key_padding_mask=src_padding_mask
                )
                logits = self.lm_head(dec_out[:, -1, :])
                next_token = logits.argmax(dim=-1).unsqueeze(0)
                
                if next_token.item() == 1: # EOS token for T5 is usually 1
                    break
                    
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
                
        return self.tokenizer.decode(curr_tokens[0], skip_special_tokens=True) + f" (Steps: {final_step})"

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
            # enc_out = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            enc_out = self.t5.shared(input_ids)
            
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
                
                num_valid = mask.sum(dim=-1).float()
                num_correct = correct_tokens.sum(dim=-1).float()
                partial_reward = num_correct / (num_valid + 1e-8) # Result is 0.0 to 1.
                
                # Sequence is correct if ALL tokens match
                seq_correct = (correct_tokens.sum(dim=-1) == mask.sum(dim=-1))
                is_correct = seq_correct.float() # 1.0 or 0.0

            step_outputs.append({
                "q_logits": q_logits,
                "lm_loss": lm_loss,
                # "is_correct": is_correct,
                "is_correct": partial_reward,
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
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = NanoACT(tokenizer).to(DEVICE)
    
    # Use the new MixedReasoningDataset (50% TextLogic, 50% Sudoku)
    dataset = HeterogeneousDataset(tokenizer, size=5000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Separate LRs for Stability
    optimizer = torch.optim.AdamW([
        {'params': model.hrm.q_head.parameters(), 'lr': 5e-4},
        {'params': [p for n, p in model.hrm.named_parameters() if 'q_head' not in n], 'lr': 1e-4},
        {'params': [p for n, p in model.named_parameters() if 'hrm' not in n], 'lr': 1e-4}  # Encoder/Decoder/Embeddings
    ])

    model.train()
    print("\nTraining NanoACT on Mixed Reasoning Dataset (TextLogic + Sudoku)...")
    
    # 2. TEST CASES FOR BOTH TASKS
    # TextLogic tests
    test_text = "track state: Mary went to the hallway. Mary moved to the office. Question: Where is Mary?"
    hard_text = "track state: John went to the garden. Mary moved to the kitchen. John moved to the office. Question: Where is John?"
    
    # Sudoku test (simple puzzle with one missing number - top-left should be 1)
    test_sudoku = "solve sudoku: 0 2 3 4 | 3 4 1 2 | 2 1 4 3 | 4 3 2 1"
    
    print(f"\n--- TEST CASES ---")
    print(f"EASY TextLogic: {test_text}")
    print(f"HARD TextLogic: {hard_text}")
    print(f"SUDOKU: {test_sudoku}")
    print("-" * 50)
    
    # Print initial (untrained) outputs
    print(f"\\n--- INITIAL (UNTRAINED) OUTPUTS ---")
    print(f"EASY TextLogic: {model.generate(test_text)}")
    print(f"HARD TextLogic: {model.generate(hard_text)}")
    print(f"SUDOKU: {model.generate(test_sudoku)}")
    print("-" * 50)
    
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
            drift_weight = 0.1
            # Total Loss
            loss = lm_loss + (q_loss_weight * avg_q_loss) + (drift_weight * drift_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_lm_loss += lm_loss.item()
            total_q_loss += avg_q_loss.item()
            
        print(f"Epoch {epoch+1} | LM Loss: {total_lm_loss/len(dataloader):.4f} | Q Loss: {total_q_loss/len(dataloader):.4f}")
        
        # Test on all task types
        model.eval()
        print(f"  EASY TextLogic: {model.generate(test_text)}")
        print(f"  HARD TextLogic: {model.generate(hard_text)}")
        print(f"  SUDOKU: {model.generate(test_sudoku)}")
        model.train()
        print("-" * 50)
        
        def debug_inference(model, text, force_steps=8):
            model.eval()
            inputs = model.tokenizer(text, return_tensors="pt").to(DEVICE)
            input_ids = inputs.input_ids
            
            with torch.no_grad():
                # Encode
                src_emb = model.dropout(model.pos_encoder(model.embedding(input_ids)))
                src_padding_mask = (input_ids == model.pad_token_id)
                memory = model.encoder(src_emb, src_key_padding_mask=src_padding_mask)
                
                # ACT Loop (Force Steps)
                B, L, D = memory.shape
                x_in = model.hrm.input_adapter(memory)
                z = model.hrm.state_init.expand(B, L, D)
                x_in = x_in + model.hrm.pos_embed[:, :L, :]
                
                # FORCE THE THINKING
                for step in range(force_steps):
                    z = model.hrm.forward_step(z, x_in, mask=(~src_padding_mask).long())
                    
                # Decode
                z_out = model.hrm.output_adapter(z)
                enhanced_memory = memory + z_out
                
                # Greedy Decode
                curr_tokens = torch.tensor([[0]], device=DEVICE) 
                for _ in range(64):
                    tgt_emb = model.dropout(model.pos_encoder(model.embedding(curr_tokens)))
                    tgt_causal_mask = torch.triu(torch.ones(curr_tokens.size(1), curr_tokens.size(1), device=DEVICE) * float('-inf'), diagonal=1)
                    
                    dec_out = model.decoder(
                        tgt=tgt_emb, 
                        memory=enhanced_memory,
                        tgt_mask=tgt_causal_mask, 
                        memory_key_padding_mask=src_padding_mask
                    )
                    logits = model.lm_head(dec_out[:, -1, :])
                    next_token = logits.argmax(dim=-1).unsqueeze(0)
                    if next_token.item() == 1: break
                    curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
                    
            return model.tokenizer.decode(curr_tokens[0], skip_special_tokens=True)

        # Test the hypothesis
        bad_sudoku = "solve sudoku: 2 0 4 3 | 0 0 2 1 | 0 1 0 0 | 4 3 0 0" 
        print("1 Step (Lazy):", debug_inference(model, bad_sudoku, force_steps=1))
        print("8 Steps (Forced):", debug_inference(model, bad_sudoku, force_steps=8))

        

if __name__ == "__main__":
    train()
