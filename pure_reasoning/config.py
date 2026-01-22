"""
Pure Reasoning Architecture - Configuration

A reasoning-focused model without pretrained T5.
Designed for QA tasks with direct answer prediction.
"""
import torch


class PureReasoningConfig:
    # ============== Debug Mode ==============
    # Set to True for quick validation with tiny model
    DEBUG_MODE = False
    
    # ============== Model Dimensions ==============
    if DEBUG_MODE:
        D_MODEL = 128             # Tiny for fast iteration
        N_HEADS = 4
        N_ENCODER_LAYERS = 2
        N_REASONING_LAYERS = 2
        D_FF = 512
    else:
        D_MODEL = 256             # Full size
        N_HEADS = 8
        N_ENCODER_LAYERS = 8      # Increased from 6 for deeper understanding
        N_REASONING_LAYERS = 2
        D_FF = 1024
    
    # ============== Reasoning Core (TRM-style) ==============
    N_RECURSIONS = 2 if DEBUG_MODE else 4
    T_DEEP_RECURSIONS = 2 if DEBUG_MODE else 3
    N_SUPERVISION = 2 if DEBUG_MODE else 4
    MIN_SUPERVISION_STEPS = 1 if DEBUG_MODE else 2
    
    # ============== PonderNet Halting ==============
    HALTING_HIDDEN_DIM = D_MODEL  # Hidden dim for halting network MLP
    LAMBDA_P = 0.25               # Geometric prior (~4 expected steps)
    REG_LOSS_WEIGHT = 0.1         # Increased 10x to prevent early-halting collapse
    HALTING_LR_MULTIPLIER = 0.1   # Separate LR for halting network
    
    # ============== Generation Settings ==============
    MAX_ANSWER_LEN = 30 if DEBUG_MODE else 50
    VOCAB_SIZE = 30522        # BERT tokenizer vocab size
    
    # ============== Training ==============
    BATCH_SIZE = 16 if DEBUG_MODE else 8
    LEARNING_RATE = 1e-3 if DEBUG_MODE else 3e-4  # Higher LR for debug
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100 if DEBUG_MODE else 1000
    EPOCHS = 10 if DEBUG_MODE else 40
    GRADIENT_CLIP = 1.0
    DROPOUT = 0.3                # Increased for better regularization
    LABEL_SMOOTHING = 0.1         # Prevent overconfident predictions
    
    # Curriculum: warmup reasoning gradually
    WARMUP_REASONING_EPOCHS = 2 if DEBUG_MODE else 5
    
    # ============== Data ==============
    MAX_CONTEXT_LEN = 256 if DEBUG_MODE else 512
    MAX_QUESTION_LEN = 64
    DATASETS = ['squad'] if DEBUG_MODE else ['squad', 'hotpotqa', 'drop']
    SAMPLES_PER_DATASET = 500 if DEBUG_MODE else 10000  # Tiny dataset for debug
    NUM_VAL_SAMPLES = 50 if DEBUG_MODE else 100
    
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
        decoder = cls.N_REASONING_LAYERS * (4 * cls.D_MODEL ** 2 + 2 * cls.D_MODEL * cls.D_FF)
        total = encoder + reasoning + decoder
        return total / 1e6
    
    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() 
                if not k.startswith('_') and not callable(getattr(cls, k))}
