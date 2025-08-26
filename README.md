Simple Reservoir (ESN) LM

- Character-level tokenizer built from your own small corpus
- Minimal ESN language model (fixed reservoir, trainable embedding + readout)
- Tiny runner to verify training end-to-end quickly

Quick start

1) Ensure you're in your existing venv and using python3.
2) Run the tiny experiment:

    python -m simple_reservoir.run_small

Scaling up

- Replace build_tiny_corpus() with loading a bigger text file or a folder of texts.
- Rebuild the CharTokenizer from that data; its vocab adapts automatically.
- Increase hidden_size, reduce sparsity, and train longer.
- Swap the char-level tokenizer with BPE later if needed.
