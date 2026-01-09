"""
Pure Reasoning Architecture - Configuration

A reasoning-focused model without pretrained T5.
Designed for QA tasks with direct answer prediction.
"""
import torch


class PureReasoningConfig:
    # ============== Model Dimensions ==============
    D_MODEL = 512             # Hidden dimension (smaller than T5's 768)
    N_HEADS = 8               # Attention heads
    N_ENCODER_LAYERS = 6      # Encoder depth
    N_REASONING_LAYERS = 4    # Reasoning core depth
    D_FF = 2048               # Feedforward dimension
    
    # ============== Reasoning Core (TRM-style) ==============
    N_RECURSIONS = 4          # Latent recursion steps per deep step
    T_DEEP_RECURSIONS = 3     # Deep recursion iterations
    N_SUPERVISION = 6         # Max supervision steps
    MIN_SUPERVISION_STEPS = 2 # Minimum before early stopping
    
    # ACT (Adaptive Computation Time)
    Q_HEAD_BIAS_INIT = -1.0   # sigmoid(-1) ≈ 0.27, encourage continuing
    Q_HEAD_LR_MULTIPLIER = 0.1
    
    # ============== Task Heads ==============
    # Span head: predicts start/end positions
    # Classification head: yes/no or multi-class
    MAX_ANSWER_SPAN = 50      # Maximum span length
    
    # ============== Training ==============
    BATCH_SIZE = 8            # Reduced for OOM workaround (no caching allocator)
    LEARNING_RATE = 3e-4      # Can be higher from scratch
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 1000
    EPOCHS = 50               # More epochs needed from scratch
    GRADIENT_CLIP = 1.0
    DROPOUT = 0.1
    
    # Curriculum: warmup reasoning gradually
    WARMUP_REASONING_EPOCHS = 5  # Only span loss initially
    
    # ============== Data ==============
    MAX_CONTEXT_LEN = 512
    MAX_QUESTION_LEN = 64
    DATASETS = ['squad', 'hotpotqa', 'drop']  # All datasets for ACT visualization
    SAMPLES_PER_DATASET = 10000
    NUM_VAL_SAMPLES = 100
    
    # ============== System ==============
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TOKENIZER_NAME = "bert-base-uncased"  # Just for tokenization
    RESULTS_DIR = "pure_reasoning/results"
    MODEL_SAVE_PATH = "pure_reasoning/models/pure_reasoning_v1.pt"
    
    @classmethod
    def total_params_estimate(cls):
        """Rough parameter count estimate"""
        encoder = cls.N_ENCODER_LAYERS * (4 * cls.D_MODEL ** 2 + 2 * cls.D_MODEL * cls.D_FF)
        reasoning = cls.N_REASONING_LAYERS * (4 * cls.D_MODEL ** 2 + 2 * cls.D_MODEL * cls.D_FF)
        heads = cls.D_MODEL * 4  # Start/end/classification heads
        total = encoder + reasoning + heads
        return total / 1e6
    
    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() 
                if not k.startswith('_') and not callable(getattr(cls, k))}
