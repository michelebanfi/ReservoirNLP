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

---

## Next Steps

1. Implement basic encoder + reasoning core
2. Add span prediction head
3. Test on SQuAD (pure extractive)
4. Extend to HotpotQA/DROP with additional heads
