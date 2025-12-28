import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import torch.nn.functional as F
from src.config import Config
from src.model import NanoHRMv3
from transformers import AutoTokenizer

def verify():
    print("Verifying implementation...")
    cfg = Config()
    
    # Override device for quick CPU check
    cfg.DEVICE = "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.TOKENIZER_NAME)
    model = NanoHRMv3(tokenizer, cfg)
    
    print(f"Model created. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Dummy input
    input_ids = torch.randint(0, 1000, (2, 32)) # B=2, L=32
    
    print("Testing Encoder...")
    memory, mask = model.encode(input_ids)
    print(f"Memory shape: {memory.shape}") # Should be [2, 32, 512]
    
    print("Testing HRM Core Init...")
    zH, zL = model.hrm_core.init_state(2, 32, "cpu")
    print(f"zH shape: {zH.shape}")
    
    print("Testing Forward Segment (1-step calc)...")
    # Detach inputs simulating Deep Supervision
    zH_in = zH.detach()
    zL_in = zL.detach()
    
    zH_out, zL_out = model.hrm_core.forward_segment(zH_in, zL_in, memory, key_padding_mask=mask)
    print(f"zH_out shape: {zH_out.shape}")
    
    print("Testing Backward Pass...")
    target = torch.randn_like(zH_out)
    loss = F.mse_loss(zH_out, target)
    loss.backward()
    
    # Check if gradients exist
    assert model.hrm_core.H_module.layers[0].attn.in_proj_weight.grad is not None
    print("Gradients computed successfully!")
    
    print("Verification Complete.")

if __name__ == "__main__":
    verify()
