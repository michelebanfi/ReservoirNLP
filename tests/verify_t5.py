import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
from src.config import Config
from src.model import NanoHRMv3
from src.train import get_baseline_samples
from transformers import AutoTokenizer

def verify_t5_setup():
    print("Verifying T5 Integration...")
    cfg = Config()
    cfg.DEVICE = "cpu" # Quick check
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    
    print("Checking Baseline Generation (Validation Comparison logic)...")
    # Should work even if we don't have model yet
    # But it calls get_dataloaders which might need internet
    try:
        samples = get_baseline_samples(tokenizer, cfg, num_samples=1)
        print(f"Sample Baseline Answer: {samples[0]['baseline']}")
    except Exception as e:
        print(f"Baseline Gen Skipped/Failed: {e}")
    
    print("Loading NanoHRMv3 with T5...")
    model = NanoHRMv3(tokenizer, cfg)
    
    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Total Parameters: {params:.2f}M")
    
    # Check if T5 params are roughly 220M and HRM ~40M (d=768)
    # 768^2 * 4 * 2 (layers) * 2 (modules) ... roughly
    
    input_ids = torch.randint(0, 1000, (1, 10))
    print("Testing Forward Pass...")
    memory, mask = model.encode(input_ids)
    print(f"Memory: {memory.shape}")
    
    zH, zL = model.hrm_core.init_state(1, 10, "cpu")
    zH, zL = model.hrm_core.forward_segment(zH, zL, memory, key_padding_mask=mask)
    
    enhanced = memory + zH
    logits = model.decode(enhanced, input_ids, mask) # input_ids as dummy labels
    print(f"Logits: {logits.shape}")
    
    print("Verification Successful!")

if __name__ == "__main__":
    verify_t5_setup()
