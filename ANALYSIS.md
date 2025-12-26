# Analysis: Why Your HRM Implementation Isn't Working for Text QA

## Summary of Results
- **Test Accuracy**: 5.5% (essentially random)
- **Validation Loss**: Getting worse after epoch 19 (overfitting)
- **ACT Steps**: Always 8.0 at validation (halting not learning)
- **Predictions**: Garbled text ("Titanders", "Lionders", "d Sterfer")

## Critical Architectural Differences from Original HRM

### 1. Missing Two-Level Hierarchy
**Original HRM:**
```
For each segment (deep supervision):
    For N high-level cycles:
        For T low-level steps:
            zL = L_module(zL, zH, x)  # Fast updates
        zH = H_module(zH, zL)         # Slow updates
    Compute loss, backprop, detach state
```

**Your Implementation:**
```
For step in max_steps:
    z = single_reasoning_block(z, memory)  # No H/L separation
    Check halt condition
```

The hierarchical convergence (L converges within each H-cycle, then resets) is what prevents premature convergence and enables deep computation.

### 2. Missing Deep Supervision
Original HRM runs multiple **segments**, computing loss after each and detaching state. This provides:
- More frequent gradient signal
- Regularization
- Curriculum-like training

Your code runs everything in one forward pass.

### 3. Wrong Problem Domain
HRM excels at:
- **Sudoku**: 81 cells, 9 values each, clear constraint satisfaction
- **ARC**: Small grids, discrete colors, pattern transformation
- **Mazes**: Path finding with clear optimal solutions

Text QA is fundamentally different:
- Open vocabulary (~32K tokens)
- Semantic understanding required
- No clear "search/backtrack" reasoning pattern

### 4. Model Size vs Task Complexity
- Original HRM: 27M params for discrete symbolic tasks
- Your model: 57M params but needs to learn:
  - Language understanding (normally needs billions of tokens)
  - Semantic reasoning
  - Answer generation

## Why Your Specific Results Are Bad

### Garbled Predictions ("Titanders", "Lionders")
This indicates the decoder is not learning proper token sequences. Likely causes:
1. Encoder can't produce meaningful representations
2. Too few parameters for language modeling
3. Need pre-trained embeddings/encoder

### Validation Steps Always 8
The Q-head isn't learning useful halt signals because:
1. Task correctness (reward) is ~5% - almost no positive signal
2. Without proper hierarchical structure, there's no meaningful "convergence" to detect
3. The exploration mechanism only matters if training works

### Overfitting (val loss increasing after epoch 19)
The model memorizes training patterns rather than learning generalizable reasoning.

## Recommendations

### Option A: Make HRM Work (Harder Path)
If you want to pursue HRM-style architecture for text:

1. **Add proper H/L hierarchy:**
```python
class HRMCore(nn.Module):
    def __init__(self, dim, num_heads):
        self.H_module = TransformerBlock(dim, num_heads)  # High-level
        self.L_module = TransformerBlock(dim, num_heads)  # Low-level
        self.T = 4  # L steps per H cycle
        self.N = 2  # H cycles per segment
        
    def forward_segment(self, zH, zL, x):
        for _ in range(self.N):
            for _ in range(self.T):
                zL = self.L_module(zL + zH + x)
            zH = self.H_module(zH + zL)
        return zH, zL
```

2. **Implement deep supervision:**
```python
def train_step(model, x, y):
    zH, zL = init_state(x)
    total_loss = 0
    for segment in range(M_max):
        zH, zL = model.forward_segment(zH, zL, x)
        pred = model.output_head(zH)
        loss = compute_loss(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        zH, zL = zH.detach(), zL.detach()  # Critical!
```

3. **Use pre-trained encoder:**
```python
from transformers import T5EncoderModel
encoder = T5EncoderModel.from_pretrained("t5-base")
# Freeze or fine-tune encoder, add HRM on top
```

### Option B: Different Architecture (Easier Path)
For text QA, consider:

1. **Fine-tune existing models:** T5, BART, or Flan-T5 already understand language
2. **Add lightweight reasoning:** Universal Transformer or PonderNet on top
3. **Retrieval-augmented generation:** For factual QA

### Option C: Different Task (HRM-Appropriate)
If you want to explore HRM properly:

1. **Symbolic math:** Learn arithmetic operations
2. **Logic puzzles:** Propositional logic, constraint satisfaction
3. **Grid transformations:** Simple pattern completion
4. **Sorting/searching:** Algorithmic tasks

## Code Changes Needed for Option A

See `train_qa_v3.py` for a corrected implementation with:
- Proper H/L module separation
- Deep supervision training loop
- Pre-trained encoder integration
- Correct Q-learning for halting

## References
- Original HRM paper: https://arxiv.org/abs/2506.21734
- HRM code: https://github.com/sapientinc/HRM
- Universal Transformer: https://arxiv.org/abs/1807.03819
- PonderNet: https://arxiv.org/abs/2107.05407
