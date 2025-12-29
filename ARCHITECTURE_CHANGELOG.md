# Architecture Changelog

This file documents all architectural edits to the T5-HRM model, including the motivation and rationale for each change. This is meant to prevent repeating failed experiments and provide context for future modifications.

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
