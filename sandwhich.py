import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import numpy as np

# --- CONFIGURATION ---
MODEL_NAME = "t5-small" # Micro model (60M params)
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 5
SEQ_LEN = 32 # Keep sequences short for efficiency
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Running on {DEVICE}...")

# --- 1. THE DATASET (Synthetic Reasoning) ---
class ArithmeticReasoningDataset(Dataset):
    """
    Generates simple arithmetic sequences.
    Input: "2 4 6 8"
    Target: "10"
    """
    def __init__(self, tokenizer, size=5000):
        self.tokenizer = tokenizer
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random start and step for arithmetic progression
        start = random.randint(1, 50)
        step = random.randint(1, 10)
        length = 5
        
        sequence = [start + i*step for i in range(length)]
        input_nums = sequence[:-1] # e.g., [2, 4, 6, 8]
        target_num = sequence[-1]  # e.g., 10
        
        input_text = "predict next: " + " ".join(map(str, input_nums))
        target_text = str(target_num)

        # Tokenize
        source = self.tokenizer(
            input_text, 
            max_length=SEQ_LEN, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        target = self.tokenizer(
            target_text, 
            max_length=10, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        return {
            "input_ids": source.input_ids.squeeze(),
            "attention_mask": source.attention_mask.squeeze(),
            "labels": target.input_ids.squeeze()
        }

# --- 2. THE REASONING CORE (HRM) ---
class LatentReasoningCore(nn.Module):
    """
    The 'Brain' inserted between Encoder and Decoder.
    It takes the Encoder's hidden states, processes them, 
    and passes them to the Decoder.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        # A simple 2-layer Transformer Encoder as the reasoning unit
        # In a real HRM, this would be your hierarchical/symbolic module
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.reasoning_layers = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Optional: Projection layers if dimensions mismatch (not needed for T5-to-T5)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, hidden_states):
        # hidden_states shape: [Batch, Seq_Len, Dim]
        thought_vector = self.reasoning_layers(hidden_states)
        return self.layer_norm(thought_vector)

# --- 3. THE FULL MODEL WRAPPER ---
class NeuroSymbolicSandwich(nn.Module):
    def __init__(self, base_model_name):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(base_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(base_model_name)
        
        # FREEZE T5
        for param in self.t5.parameters():
            param.requires_grad = False
            
        # The Reasoning Core
        hidden_dim = self.t5.config.d_model
        self.hrm = LatentReasoningCore(hidden_dim)
        
        print(f"T5 Frozen. Trainable Parameters: {sum(p.numel() for p in self.hrm.parameters() if p.requires_grad)}")

    def forward(self, input_ids, attention_mask, labels=None):
        # 1. ENCODE (Using Frozen T5 Encoder)
        # We manually call the encoder to get hidden states
        encoder_outputs = self.t5.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state
        
        # 2. REASON (Using Trainable HRM)
        # We intercept the signal here!
        reasoned_states = self.hrm(hidden_states)
        
        # 3. DECODE (Using Frozen T5 Decoder)
        # We pass our 'reasoned' states as 'encoder_hidden_states' to the decoder
        outputs = self.t5(
            encoder_outputs=(reasoned_states,), # Inject our vector
            labels=labels,
            decoder_input_ids=self.t5._shift_right(labels) if labels is not None else None
        )
        
        return outputs

    def generate(self, input_text):
        # Inference helper
        inputs = self.tokenizer(input_text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            # 1. Encode
            enc_out = self.t5.encoder(inputs.input_ids).last_hidden_state
            # 2. Reason
            reasoned_state = self.hrm(enc_out)
            # 3. Decode (Greedy generation)
            # We must construct a dummy EncoderOutput object for the T5 generate method
            from transformers.modeling_outputs import BaseModelOutput
            dummy_enc_out = BaseModelOutput(last_hidden_state=reasoned_state)
            
            generated_ids = self.t5.generate(
                encoder_outputs=dummy_enc_out,
                max_length=10
            )
        
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# --- 4. TRAINING LOOP ---
def train():
    # Setup
    model = NeuroSymbolicSandwich(MODEL_NAME).to(DEVICE)
    dataset = ArithmeticReasoningDataset(model.tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.hrm.parameters(), lr=LEARNING_RATE) # optimize ONLY HRM

    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            # Forward pass (Encoder -> HRM -> Decoder)
            outputs = model(input_ids, mask, labels=labels)
            
            loss = outputs.loss # T5 calculates CrossEntropy loss automatically
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
        
        # Quick Test during training
        test_q = "predict next: 10 20 30"
        print(f"  Test: {test_q} -> {model.generate(test_q)}")

if __name__ == "__main__":
    train()