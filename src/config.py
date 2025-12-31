import torch

class Config:
    # Model Dimensions (Paper: 27M params -> d_model=512)
    # UPDATED: T5-Base uses d_model=768, heads=12
    D_MODEL = 768
    N_HEADS = 12
    
    # HRM Architecture
    N_HIGH_CYCLES = 2      # N: H-module updates per segment
    N_LOW_STEPS = 4        # T: L-module updates per H-cycle
    N_HRM_LAYERS = 2       # Layers per module (H/L)
    N_REASONING_TOKENS = 4 # K: soft-prompt tokens pooled from zH
    REASONING_GATE_INIT = 0.5  # Initialize gate slightly positive to encourage HRM usage
    
    # Deep Supervision & ACT (Graves 2016)
    MAX_SEGMENTS = 8       # M_max
    ACT_EPSILON = 0.01     # Halt when cumulative prob within epsilon of 1
    ACT_PONDER_COST_TAU = 0.01  # Ponder cost regularizer weight
    NUM_VAL_SAMPLES = 10   # Number of validation samples for stable metrics
    
    # Training
    DROPOUT = 0.1
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01    # Adjusted standard value
    EPOCHS = 20
    THINKING_WARMUP_EPOCHS = 2
    GRADIENT_CLIP = 1.0
    
    # Data
    MAX_SRC_LEN = 256
    MAX_TGT_LEN = 32
    TRAIN_SIZE = 10000
    VAL_SIZE = 1000
    
    TOKENIZER_NAME = "google/flan-t5-base"
    
    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR = "results"
    MODEL_SAVE_PATH = "models/act_qa_model_v4.pt"

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() 
                if not k.startswith('_') and not isinstance(v, classmethod)}
