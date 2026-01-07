"""
TRM (Tiny Recursion Model) Configuration

Based on arXiv:2510.04871 - "Less is More: Recursive Reasoning with Tiny Networks"
Key differences from HRM:
- Single tiny network (2 layers) instead of separate H/L modules
- Full gradient flow through n recursions (no 1-step approximation)
- Simpler ACT without extra forward pass
- EMA for weight smoothing
"""
import torch


class TRMConfig:
    # Model Dimensions (match T5-Base)
    D_MODEL = 768
    N_HEADS = 12
    
    # TRM Architecture (Paper: Table 1 - best config T=3, n=6, 2 layers)
    N_LAYERS = 2              # Tiny: only 2 transformer layers (paper key finding)
    N_RECURSIONS = 6          # n: latent recursion steps per deep recursion
    T_DEEP_RECURSIONS = 3     # T: total deep recursions (T-1 no-grad + 1 with-grad)
    N_SUPERVISION = 16        # Deep supervision steps (same as HRM)
    
    # EMA (Exponential Moving Average on weights)
    USE_EMA = True
    EMA_DECAY = 0.99          # Paper: helps generalization
    
    # ACT (Adaptive Computation Time) - Simplified
    # TRM uses simple BCE loss: q_hat predicts (y_hat == y_true)
    # No ponder cost needed, no extra forward pass for Q-learning
    Q_HEAD_BIAS_INIT = -1.0   # sigmoid(-1) ≈ 0.27, encourage continuing initially
    Q_HEAD_LR_MULTIPLIER = 0.1  # Q-head learns 10x slower to prevent dominating
    MIN_SUPERVISION_STEPS = 3   # Minimum steps before early stopping allowed
    
    # Training
    DROPOUT = 0.1
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01
    EPOCHS = 20
    GRADIENT_CLIP = 1.0
    
    # Curriculum: freeze T5 for first N epochs
    FREEZE_T5_EPOCHS = 2
    
    # Data (same as HRM for fair comparison)
    MAX_SRC_LEN = 512
    MAX_TGT_LEN = 64
    SAMPLES_PER_DATASET = 5000
    DATASETS = ['squad', 'hotpotqa', 'drop']
    NUM_VAL_SAMPLES = 50
    
    TOKENIZER_NAME = "google/flan-t5-base"
    
    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR = "results"
    MODEL_SAVE_PATH = "models/trm_qa_model_v1.pt"
    
    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() 
                if not k.startswith('_') and not isinstance(v, classmethod)}
