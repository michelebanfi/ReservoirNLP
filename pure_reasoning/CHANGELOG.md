# Pure Reasoning Architecture - Changelog

This document tracks the design decisions, motivations, and evolution of the Pure Reasoning architecture.

---

## 2026-01-09: Project Inception

### Motivation

After extensive experimentation with the TRM (Tiny Recursion Model) integrated with T5, we observed persistent issues:

1. **Capacity Mismatch**: The 50M TRM core struggled to maintain its learned policies when competing with T5's 220M parameters during joint training.

2. **ACT Collapse**: The Q-head repeatedly collapsed to trivial policies (always halt immediately or never halt) after T5 unfreezing.

3. **Conflicting Objectives**: T5's decoder is optimized for fluent next-token prediction, not for reasoning. The model had to balance:
   - Learning to reason (TRM's job)
   - Learning to generate text (T5 decoder's job)
   - These objectives can interfere with each other

4. **Pretrained Bias**: T5 was pretrained on general text, not QA reasoning. Its attention patterns may not align with what TRM needs for multi-hop reasoning.

### Core Insight

> **Next-token prediction is not the same as reasoning.**

For QA tasks, we don't need to generate arbitrary text—we need to:
- Understand context and question
- Perform reasoning steps (comparison, multi-hop, arithmetic)
- Produce a specific answer

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────┐
│  ENCODER (trained from scratch, ~50M params)            │
│  - Transformer-based, no pretrained weights             │
│  - Input: [CLS] context [SEP] question [SEP]            │
│  - Output: contextualized token embeddings              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  REASONING CORE (TRM-style, ~50M params)                │
│  - Recursive refinement: (x, y, z) → (y', z')           │
│  - ACT mechanism for adaptive computation               │
│  - This is where reasoning happens                      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  TASK HEADS (lightweight, ~5M params)                   │
│  - Span Head: predict start/end positions               │
│  - Classification Head: yes/no, comparison              │
│  - Numeric Head: count/arithmetic (optional)            │
└─────────────────────────────────────────────────────────┘
```

### Key Differences from TRM+T5

| Aspect | TRM+T5 | Pure Reasoning |
|--------|--------|----------------|
| Encoder | Pretrained T5 (220M) | From scratch (~50M) |
| Output | Autoregressive decoding | Direct prediction |
| Training Signal | Cross-entropy on tokens | Span loss, BCE |
| Generation | Token-by-token | Single forward pass |
| Total Params | ~300M | ~100M |

### Dataset Compatibility

All existing datasets can be used with task-specific heads:

- **SQuAD**: Span extraction (start/end positions in context)
- **HotpotQA**: Span extraction + supporting fact prediction
- **DROP**: Span + number prediction head

### Expected Benefits

1. **Focused Learning**: All parameters devoted to reasoning, not text generation
2. **Faster Training**: Single forward pass vs autoregressive decoding
3. **Cleaner Gradients**: Direct supervision on answers, not token-by-token
4. **Smaller Model**: ~100M total vs ~300M with T5

### Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Encoder from scratch may underperform | Start small, scale up if needed |
| No pretrained knowledge | Larger training dataset, longer training |
| Limited to extractive answers | Span head covers most QA; can add generation later |

## 2026-01-12: Pivot to Generative Architecture

### Motivation

The span-extraction approach was not yielding good results:

1. **SQuAD at 0% EM**: Despite decreasing losses, the model couldn't produce correct spans
2. **Dataset Issues**: Span alignment bugs caused incorrect supervision
3. **Debugging Difficulty**: Span positions don't reveal *what* the model is thinking

### Decision

Pivot to **text generation** instead of span extraction. This gives us:

- **Inspectable outputs**: We can see exactly what the model generates
- **Flexibility**: Not limited to extractive answers
- **Familiar territory**: We have more experience debugging generative models

### New Architecture

```
┌─────────────────────────────────────────────────────────┐
│  ENCODER (from scratch, ~50M params)                    │
│  - Input: [CLS] context [SEP] question [SEP]            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  REASONING CORE (TRM-style, ~50M params)                │
│  - Recursive refinement with ACT                        │
│  - Refines encoder memory through y,z states            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  DECODER (from scratch, ~50M params)                    │
│  - Cross-attention to refined memory                    │
│  - Autoregressive text generation                       │
└─────────────────────────────────────────────────────────┘
```

### Changes Made

| File | Change |
|------|--------|
| `config.py` | Added `MAX_ANSWER_LEN`, `VOCAB_SIZE`, removed span settings |
| `model.py` | Removed task heads, added `PureDecoder` with cross-attention |
| `dataset.py` | Changed to `GenerativeQADataset`, returns decoder targets |
| `train.py` | Cross-entropy loss, text EM/F1, sample generation logging |

### Metrics Logging

The `pure_reasoning_metrics.json` now includes `sample_generations` with:
- Source dataset
- Question text
- Gold answer
- Predicted answer

This allows direct inspection of model behavior.

---

## Next Steps

1. Run training with generative architecture
2. Inspect sample generations in metrics
3. Iterate based on observed failure modes

---

## 2026-01-18: Training Fixes for Overfitting and ACT

### Problem Analysis

After 50 epochs (5 days training), the model showed:
- Train loss dropped 7.11 → 0.49 (93% reduction)
- Val EM stayed ~4%, F1 ~6%
- ACT steps frozen at 4.0 with act_loss = 0.0
- Generated text was gibberish unrelated to questions

### Root Causes

1. **ACT not learning**: The ACT loss was never computed (TODO placeholder in code)
2. **Overfitting**: Model memorizing training data with 0.1 dropout, no label smoothing
3. **Insufficient capacity**: Only 6 encoder layers for from-scratch learning

### Fixes Applied

| Fix | Before | After |
|-----|--------|-------|
| ACT loss | TODO (not computed) | Ponder cost: Σ(1-q_hat) per step |
| Label smoothing | None | 0.1 |
| Dropout | 0.1 | 0.3 |
| Encoder layers | 6 | 8 |

### Config Changes

```python
N_ENCODER_LAYERS = 8      # Was: 6
DROPOUT = 0.3             # Was: 0.1
LABEL_SMOOTHING = 0.1     # NEW
ACT_LOSS_LAMBDA = 0.01    # NEW
```

### Code Changes

**model.py**: Replaced TODO with actual ACT ponder cost:
```python
ponder_cost = (1 - q_hat).mean()
total_act_loss = total_act_loss + ponder_cost
```

Added label smoothing to cross-entropy:
```python
loss = F.cross_entropy(..., label_smoothing=self.config.LABEL_SMOOTHING)
```

**train.py**: Added ACT loss to total loss:
```python
if isinstance(act_loss, torch.Tensor):
    total_loss_val = loss + config.ACT_LOSS_LAMBDA * act_loss
```

### Expected Improvements

1. ACT should now show varying `avg_steps` as Q-head learns
2. Label smoothing should reduce overfitting
3. Higher dropout should prevent memorization
4. More encoder layers should improve representation quality

---

## 2026-01-18: PonderNet-style Halting Network

### Motivation

Based on the paper "PonderNet: Learning to Ponder" (Banino et al., 2021), we replaced the simple linear Q-head with a proper adaptive computation mechanism.

### Changes

**New Classes in model.py:**
- `HaltingNetwork`: MLP that predicts λ_n (halting probability) per step
- `ReconstructionLoss`: L_rec = Σ p_n * L(y, ŷ_n) - weighted loss across steps
- `RegularizationLoss`: KL divergence with geometric prior for exploration

**Key Algorithm:**
```
p_n = λ_n × Π(1-λ_j) for j < n  # Unconditioned halt probability
L = L_rec + β * L_reg           # Total loss
```

**Config Updates:**
```python
HALTING_HIDDEN_DIM = D_MODEL  # Halting network MLP hidden dim
LAMBDA_P = 0.2                # Geometric prior (~5 expected steps)
REG_LOSS_WEIGHT = 0.01        # KL regularization weight
HALTING_LR_MULTIPLIER = 0.1   # Separate LR for halting network
```

### Expected Benefits

1. **Proper probability distribution**: halt probs sum to ~1
2. **Exploration via KL regularization**: prevents collapse to always-halt
3. **Per-step loss weighting**: harder samples can use more steps
4. **Expected steps metric**: smooth measure of computation used

---

## 2026-01-22: Early-Halting Collapse Fix

### Problem Observed

After 28 epochs of training, the model showed:
- Train loss decreasing well: 7.44 → 3.29 ✅
- Expected steps collapsed: ~0.6 (should be ~3-4) ❌
- Val EM stuck at ~3% despite lower loss ❌
- Generated text: random historical phrases ("battle of poland", "dutch war")

### Root Cause

The `REG_LOSS_WEIGHT = 0.01` was too weak to prevent the model from taking a shortcut by halting immediately. The model learned to minimize reconstruction loss without doing proper reasoning.

### Fix Applied

| Parameter | Before | After |
|-----------|--------|-------|
| `LAMBDA_P` | 0.35 | 0.25 |
| `REG_LOSS_WEIGHT` | 0.01 | **0.1** (10x increase) |

**Rationale:**
- Lower `LAMBDA_P` (0.25) → geometric prior expects ~4 steps
- Higher `REG_LOSS_WEIGHT` (0.1) → stronger penalty for deviating from prior

### Expected Outcome

The model should now:
1. Use 3-4 reasoning steps on average
2. Actually compute refinements before generating
3. Show improving validation accuracy as training progresses

---

## 2026-01-23: Major PonderNet Architecture Fixes

### Problem Observed

After 5 epochs, the model showed critical issues:
- avg_steps collapsed to ~0.56 (should be 3-5) ❌
- SQuAD EM = 0% ❌
- **reg_loss was NEGATIVE** (-0.20 to -0.25) ❌
- Generated text: degenerate "the the the..." patterns

### Root Cause Analysis

7 critical bugs were identified by comparing with reference PonderNet implementation:

| Issue | Description |
|-------|-------------|
| **KL Divergence** | Manual formula could produce negative values |
| **Reconstruction Loss** | Used batch mean, lost per-sample weighting |
| **State Refinement** | States detached between steps (no continuity) |
| **Halting Init** | Neutral bias (should favor continuing) |
| **No Step Embedding** | Model didn't know pondering position |
| **No Minimum Steps** | Could halt immediately |
| **Hyperparameters** | LR too high, warmup too short |

### Fixes Applied

**Fix 1: RegularizationLoss (CRITICAL)**
```python
# Before: manual formula that could be negative!
kl = (p_clamped * (p_clamped.log() - p_g_clamped.log())).sum()

# After: PyTorch KLDivLoss (always >= 0)
self.kl_div = nn.KLDivLoss(reduction='batchmean')
kl = self.kl_div(p_g.log(), p_clamped)
```

**Fix 2: ReconstructionLoss**
```python
# Before: batch-averaged loss
rec_loss = (p_n.mean() * step_loss.mean()).sum()

# After: per-sample weighted loss
for p_n, loss_n in zip(p_n_list, step_losses_unreduced):
    total += (p_n * loss_n).mean()  # Per-sample, then mean
```

**Fix 3: Continuous State Refinement**
```python
# Before: y, z detached between supervision steps
for step in range(n_supervision):
    (y, z), y_out, lambda_n = self.reasoning.deep_recursion(...)  # Detaches!

# After: continuous state, only detach T-1 inner recursions
for step in range(n_supervision):
    y, z, lambda_n = self.reasoning.forward_step(memory, y, z, step, ...)
    # y, z carry forward to next step
```

**Fix 4: Halting Network Initialization**
```python
# Before: neutral bias
nn.init.zeros_(self.net[-1].bias)

# After: bias toward continuing (sigmoid(-2) ≈ 0.12 halt prob)
nn.init.constant_(self.net[-1].bias, -2.0)
```

**Fix 5: Step Embeddings**
```python
class ReasoningCore(nn.Module):
    def __init__(self, config):
        ...
        self.step_embed = nn.Embedding(config.N_SUPERVISION + 1, config.D_MODEL)
    
    def forward_step(self, memory, y, z, step_idx, ...):
        step_emb = self.step_embed(torch.full((B,), step_idx, ...))
        y = y + step_emb  # Now model knows which step it's on
```

**Fix 6: Minimum Steps & Final Halt**
```python
for step in range(n_supervision):
    if step == n_supervision - 1:
        lambda_n = torch.ones_like(lambda_n)  # Must halt at final step
    if step < min_steps - 1:
        lambda_n = torch.zeros_like(lambda_n)  # Can't halt before min_steps
```

**Fix 7: Hyperparameters**
| Parameter | Before | After |
|-----------|--------|-------|
| LAMBDA_P | 0.25 | 0.2 (expect 5 steps) |
| REG_LOSS_WEIGHT | 0.1 | 0.05 (was too strong) |
| LEARNING_RATE | 3e-4 | 1e-4 |
| WARMUP_STEPS | 1000 | 2000 |
| N_SUPERVISION | 4 | 6 |

### Expected Improvements

1. reg_loss should now always be >= 0
2. avg_steps should be ~3-5 (controlled by LAMBDA_P)
3. Continuous refinement should allow iterative reasoning
4. Step embeddings inform the model about pondering progress
5. Lower LR and longer warmup should stabilize training
