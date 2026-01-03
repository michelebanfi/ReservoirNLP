# Architecture Changelog

This file documents all architectural edits to the T5-HRM model, including the motivation and rationale for each change. This is meant to prevent repeating failed experiments and provide context for future modifications.

---

## [IMPLEMENTED] 2026-01-03: ACT Overcorrection Fix - Adaptive Ponder Cost

**Status**: Implemented

### Problem
After the 2026-01-01 fixes, training showed ACT overcorrected in the opposite direction:
- q_halt saturated to 1.0 by epoch 2
- Model halted after exactly 1 segment (was stuck at 2 before)
- ponder cost (τ=0.1) dominated as LM loss decreased from 3.6→1.5
- Best accuracy was epoch 1 (50% EM at 2 segments), degraded with 1 segment

### Root Cause
As training progressed, LM loss dropped from ~3.6 to ~1.5, but ACT loss remained constant relative to ponder cost. This caused ACT to represent an increasing fraction of total loss (5% → 13%), pushing the model to minimize computation over accuracy.

### Changes

1. **Reduced Ponder Cost** (`config.py`):
   - `ACT_PONDER_COST_TAU`: 0.1 → 0.01 (10x reduction)

2. **Adaptive Scaling** (`config.py`, `train.py`):
   - Added `ACT_ADAPTIVE_SCALING = True`
   - Added `INITIAL_LM_LOSS = 3.5` 
   - ACT loss now scales by `(current_lm_loss / initial_lm_loss)`
   - At epoch 1: `act_loss = 0.01 * ponder * 1.0`
   - At epoch 20: `act_loss = 0.01 * ponder * 0.43` (auto-reduced)

### Expected Behavior
- ACT signal remains proportional to LM loss throughout training
- Model should use 2-4 segments instead of collapsing to 1
- Q-halt should vary based on question difficulty, not saturate to 1.0

---

## [IMPLEMENTED] 2026-01-01: Critical Fixes - Gradient Flow & ACT Mechanism

**Status**: Implemented

### Problems Fixed

Analysis of 12 epochs of training revealed three critical issues:

1. **Gradient Starvation**: With N=2, T=4 (8 steps), only the final step received gradients due to `torch.no_grad()` block. HRM learned from only 12.5% of computation.

2. **ACT Stuck at 2 Segments**: Q-head initialized to 0 → sigmoid(0)=0.5. After 2 steps cumulative_halt reached 1.0, causing immediate halt regardless of task difficulty.

3. **Weak ACT Loss Signal**: τ=0.001 resulted in negligible ACT loss (~0.002) vs LM loss (~1.8), preventing Q-head learning.

4. **Small Validation Set**: Only 9 samples caused high variance in accuracy metrics.

### Changes

1. **Full Gradient Flow** (`model.py`):
   - Reduced `N_HIGH_CYCLES=1, N_LOW_STEPS=2` (total 2 steps from 8)
   - Removed `torch.no_grad()` block entirely
   - All HRM reasoning steps now receive gradients

2. **ACT Parameter Fixes** (`config.py`):
   - `ACT_PONDER_COST_TAU`: 0.001 → 0.1 (100x stronger)
   - `Q_HEAD_BIAS_INIT`: 0.0 → -2.0 (sigmoid(-2)≈0.12 = encourage continuing)
   - Now Q-head learns WHEN to halt instead of halting by default

3. **Validation Improvement** (`config.py`):
   - `NUM_VAL_SAMPLES`: 10 → 50 (more stable accuracy metrics)

4. **Gradient Monitoring** (`train.py`):
   - Added optional `DEBUG_GRADIENTS` flag to monitor Q-head and H-module gradient flow

### Expected Behavior
- HRM should receive meaningful gradients and learn effectively
- ACT should initially use 3-8 segments, then learn to optimize
- More segments for HotpotQA/DROP (multi-hop) than SQuAD (extractive)
- Validation accuracy should be more stable and ≥ baseline

---

## [PROPOSAL] Training HRM From Scratch (Future Experiment)

**Status**: Not yet implemented - proposed for future exploration

### Motivation
Current approach integrates HRM with pretrained T5, which may cause conflicts between:
- T5's learned attention patterns vs HRM's reasoning tokens
- Pretrained decoder expectations vs HRM-enhanced memory

### Proposed Staged Approach

```
Stage 1: Train small encoder-only LM (~10M params) on WikiText masked LM
Stage 2: Add HRM module, train on sentence completion (requires reasoning)
Stage 3: Add decoder, fine-tune on QA datasets
```

### Benefits
- HRM co-evolves with encoder from start (no pretrained bias)
- Unified architecture without adaptation layers
- Curriculum flexibility: language first, reasoning second

### Challenges
- Requires significant compute and training time
- Need to balance language learning vs reasoning learning
- May need custom datasets for Stage 2 (reasoning-focused completion)

### Implementation Notes (for when this is implemented)
- Consider starting with ~50M parameter model (smaller than T5-Base)
- Use gradient accumulation to simulate larger batches
- Track both perplexity (language) and reasoning metrics separately

---

## [BUGFIX/FEATURE] 2025-12-31: Force HRM Mode & Validation Fix

**Status**: Implemented

### Problems Fixed

1. **Validation only testing SQuAD**: The `get_baseline_samples` function only grabbed one batch which contained only SQuAD samples. HotpotQA/DROP were never tested.

2. **Gate collapse**: Reasoning gate kept dropping (0.43 → 0.26), causing HRM contribution to diminish and accuracy to degrade.

### Changes

1. **Fixed Validation Sampling** (`train.py`):
   - Now explicitly loads samples from EACH dataset in `config.DATASETS`
   - Ensures balanced representation (e.g., 3 samples × 3 datasets = 9 samples)

2. **Added FORCE_HRM Mode** (`config.py`, `model.py`):
   - New config: `FORCE_HRM = True`
   - When True, the reasoning gate is bypassed entirely
   - Reasoning tokens are always used at full strength (no skip connection)
   - This forces the model to rely on HRM for all predictions

### Rationale
The gate was allowing the model to "cheat" by bypassing HRM and relying on the T5 baseline. By forcing HRM usage, we ensure the model must learn to use the reasoning mechanism effectively.

---


## [IMPLEMENTED] 2025-12-31: Multi-Hop Dataset Integration

**Status**: Implemented

### Motivation
Previous training showed ACT halting very quickly (1 segment) on simple SQuAD questions. To properly exercise the multi-step reasoning capability, we need harder datasets that require genuine reasoning.

### Changes

1. **New Datasets Added**:
   - **HotpotQA** (distractor): Multi-hop reasoning, comparison and bridge questions
   - **DROP**: Discrete reasoning, arithmetic over text
   - **SQuAD**: Retained as baseline extractive QA

2. **Unified Dataset Loader** (`dataset.py`):
   - `UnifiedQADataset` class normalizes column names across datasets
   - Tracks `source` and `difficulty` per sample
   - Balanced sampling: 5000 samples per dataset

3. **ACT Tuning** (`config.py`):
   - `ACT_PONDER_COST_TAU`: 0.01 → 0.001 (allow more reasoning steps)
   - `MAX_SRC_LEN`: 256 → 512 (longer multi-hop contexts)
   - `MAX_TGT_LEN`: 32 → 64 (longer answers)

4. **Per-Dataset Metrics** (`train.py`):
   - Validation now shows source dataset for each sample
   - Prints per-dataset summary (avg segments, accuracy)
   - Logs `per_dataset` segments in metrics.json

### Expected Behavior
- SQuAD: ~1-2 segments (quick, extractive)
- HotpotQA/DROP: ~2-4 segments (more reasoning needed)
- ACT should learn to spend more compute on harder questions

---


## [IMPLEMENTED] 2025-12-31: Proper ACT Implementation (Graves 2016)

**Status**: Implemented

### Problem
Previous ACT implementation had issues:
- **Hard threshold halting**: Used `if p_halt > 0.5: break` instead of proper cumulative probability
- **No weighted output**: Used last state instead of weighted combination of all states
- **Weak loss signal**: ACT loss based on loss improvement was noisy and ineffective
- **Halting oscillating**: `avg_segments_used` would flip between 2 and 4 randomly

### Changes (Based on arXiv:1603.08983)

1. **Cumulative Halting Probabilities**:
   - At each step, accumulate `p_halt` until total reaches ~1.0
   - Stop when `1.0 - cumulative < epsilon` (ACT_EPSILON = 0.01)
   - Allows smooth, differentiable halting decisions

2. **Weighted State Combination**:
   - Each intermediate state `zH_m` is weighted by its halting contribution
   - Final state: `final_zH = Σ(normalized_weight_m * zH_m)`
   - Provides smooth gradient flow through all reasoning steps

3. **Ponder Cost Regularizer**:
   - `ρ = N + R` where N = steps taken, R = remainder probability
   - ACT loss: `tau * ponder_cost` (ACT_PONDER_COST_TAU = 0.01)
   - Gently encourages model to minimize computation

4. **Configuration Updates**:
   - `EPOCHS = 20` (increased from 10 for complex reasoning task)
   - Removed old ACT_LOSS_WEIGHT and MIN_SEGMENTS_PROB
   - Added ACT_EPSILON and ACT_PONDER_COST_TAU

### Expected Benefits
- Model learns *when* to halt based on task difficulty
- Smoother gradient flow through reasoning steps
- Reduced oscillation in segments used
- Better compute efficiency over time

---


## [IMPLEMENTED] 2025-12-30: ACT Loss & Training Improvements

**Status**: Implemented

### Problem
Analysis of training metrics showed:
- **Gate stuck negative** (~-0.08): Model was actively suppressing HRM
- **Q-values frozen** (q_halt=0.119, q_continue=0.5): ACT not learning
- **Segments always max** (4): Halting mechanism not working

### Changes
1. **ACT Loss**: Added explicit loss to train Q-head based on loss improvement
   - If continuing improves loss → encourage p_continue
   - If no improvement → encourage p_halt
   - Weight: `ACT_LOSS_WEIGHT = 0.1`

2. **Gate Initialization**: Changed from 0 → 0.1 (positive)
   - Model now starts with ~10% HRM contribution
   - Prevents immediate suppression

3. **More Validation Samples**: Increased from 3 → 10
   - `NUM_VAL_SAMPLES = 10` for stable accuracy metrics

---

## [BUGFIX] 2025-12-29: Autoregressive Generation Token Duplication

**Status**: Fixed

### Problem
Validation and inference code called `decode()` for autoregressive generation, but `decode()` internally calls `_shift_right(labels)` which is meant for **teacher-forced training**. This caused token duplication in outputs.

### Root Cause
- `decode()` expects labels and shifts them right
- For generation, we were passing `decoder_input_ids` as "labels"
- `_shift_right([0])` → `[0, 0]` - creates extra tokens and off-by-one errors

### Solution
Added `generate_step()` method that does NOT shift right:
- `decode()` → for training with labels (teacher forcing)
- `generate_step()` → for autoregressive inference

---

## [IMPLEMENTED] 2025-12-29: Soft-Prompt Context Architecture

**Status**: Implemented

### Problem
The current `GatedResidualAdapter` approach performs `memory + gate * zH`, directly adding the "Latent Reasoning" state (zH) onto the crisp token embeddings. This causes:
- **Token blurring**: Each encoder token becomes a mix of its original embedding + reasoning state
- **Decoder confusion**: The decoder's cross-attention sees "smeared" representations
- **Token duplication**: Outputs like "Denver Denver Bro Bronncocoss" instead of "Denver Broncos"

### Root Cause Analysis
The HRM reasoning state `zH` is evolved via global attention and contains "smeared" concepts across the whole sequence. When we add this directly to individual token embeddings, we destroy the distinctness that the decoder relies on to generate clean text.

### Solution
Treat the Reasoning State as **Additional Context** (soft prompts) rather than modifying the original text:

1. **Pool zH** into K learned "reasoning tokens" (e.g., K=4-8 tokens)
2. **Prepend** these tokens to the encoder memory: `[reasoning_tokens; original_memory]`
3. **Extend the attention mask** appropriately (reasoning tokens are always valid)
4. The decoder can now **attend to reasoning context** while keeping original tokens crisp

### Benefits
- Encoder memory tokens remain unmodified and crisp
- Decoder can selectively attend to reasoning when needed
- Clean separation between "what the text says" and "what the reasoning implies"

---

## [Historical] Previous Approaches

### [SUPERSEDED] GatedResidualAdapter (v3)
- **Approach**: `memory + gate * GELU(proj(reasoning))`
- **Gate initialization**: Start at 0 (pure T5 baseline)
- **Issue**: Even small gate values cause token blurring and duplication
- **Status**: FAILED - causes token duplication in decoder outputs

### [SUPERSEDED] Direct Concatenation (v2)
- **Approach**: Concatenate `[memory; zH]` along sequence dimension
- **Issue**: Doubles sequence length, incompatible with positional encoding
- **Status**: FAILED - positional encoding mismatch, high memory cost

### [SUPERSEDED] Direct Addition (v1)
- **Approach**: `memory + zH`
- **Issue**: Uncontrolled blending, immediate T5 baseline degradation
- **Status**: FAILED - catastrophic forgetting of T5 capabilities

---

## Notes

- **Positional Encoding**: Added sinusoidal PE to zH/zL in HRM loops
- **Curriculum Learning**: Freeze T5 for first 2 epochs, then unfreeze
- **1-Step Gradient Approximation**: Used for memory efficiency in deep supervision loop
