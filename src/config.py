import torch

class Config:
    # Model Dimensions (Paper: 27M params -> d_model=512)
    # UPDATED: T5-Base uses d_model=768, heads=12
    D_MODEL = 768
    N_HEADS = 12
    
    # HRM Architecture
    N_HIGH_CYCLES = 1      # N: H-module updates per segment (reduced for gradient flow)
    N_LOW_STEPS = 2        # T: L-module updates per H-cycle (reduced for gradient flow)
    N_HRM_LAYERS = 2       # Layers per module (H/L)
    N_REASONING_TOKENS = 4 # K: soft-prompt tokens pooled from zH
    REASONING_GATE_INIT = 0.5  # Initialize gate slightly positive to encourage HRM usage
    FORCE_HRM = True  # If True, disable gate/skip connection (100% HRM, no bypass)
    
    # Deep Supervision & ACT (Graves 2016)
    MAX_SEGMENTS = 4       # M_max (reduced for memory)
    ACT_EPSILON = 0.01     # Halt when cumulative prob within epsilon of 1
    ACT_PONDER_COST_TAU = 0.01   # Ponder cost weight (reduced 10x to prevent early halting)
    ACT_ADAPTIVE_SCALING = True  # Scale ponder cost relative to initial LM loss
    INITIAL_LM_LOSS = 3.5        # Approximate initial LM loss for adaptive scaling
    Q_HEAD_BIAS_INIT = -2.0      # Init Q-head bias: sigmoid(-2)≈0.12 = encourage continuing
    NUM_VAL_SAMPLES = 50   # Number of validation samples for stable metrics
    
    # Training
    DROPOUT = 0.1
    BATCH_SIZE = 4         # Reduced for driver-constrained GPU memory
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.01    # Adjusted standard value
    EPOCHS = 20
    THINKING_WARMUP_EPOCHS = 2
    GRADIENT_CLIP = 1.0
    
    # Data
    MAX_SRC_LEN = 512      # Increased for longer multi-hop contexts
    MAX_TGT_LEN = 64       # Increased for longer answers
    SAMPLES_PER_DATASET = 5000  # Equal sampling from each dataset
    DATASETS = ['squad', 'hotpotqa', 'drop']  # Multi-hop reasoning datasets
    
    TOKENIZER_NAME = "google/flan-t5-base"
    
    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    RESULTS_DIR = "results"
    MODEL_SAVE_PATH = "models/act_qa_model_v4.pt"

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() 
                if not k.startswith('_') and not isinstance(v, classmethod)}
