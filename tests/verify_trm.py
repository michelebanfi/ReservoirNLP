"""
Verification script for TRM (Tiny Recursion Model) architecture.
Tests forward pass, gradient flow, and EMA functionality.

Based on tests/verify_setup.py pattern.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from src.config_trm import TRMConfig
from src.model_trm import TinyRecursionModel, EMAModel
from transformers import AutoTokenizer


def verify_trm():
    print("=" * 60)
    print("TRM Architecture Verification")
    print("=" * 60)
    
    config = TRMConfig()
    config.DEVICE = "cpu"  # Use CPU for quick testing
    
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    print("\n1. Creating TRM model...")
    model = TinyRecursionModel(tokenizer, config)
    total_params = sum(p.numel() for p in model.parameters())
    trm_params = sum(p.numel() for p in model.trm_core.parameters())
    print(f"   Total params: {total_params/1e6:.2f}M")
    print(f"   TRM Core params: {trm_params/1e6:.2f}M")
    
    # Dummy input
    B, L = 2, 32
    input_ids = torch.randint(0, 1000, (B, L))
    
    print("\n2. Testing Encoder...")
    memory, src_mask = model.encode(input_ids)
    print(f"   Memory shape: {memory.shape}")  # Should be [2, 32, 768]
    assert memory.shape == (B, L, config.D_MODEL), "Memory shape mismatch!"
    print("   ✓ Encoder OK")
    
    print("\n3. Testing TRM Core Init...")
    y, z = model.trm_core.init_state(B, L, "cpu")
    print(f"   y (answer) shape: {y.shape}")  # [2, 32, 768]
    print(f"   z (latent) shape: {z.shape}")  # [2, 32, 768]
    assert y.shape == (B, L, config.D_MODEL), "y shape mismatch!"
    assert z.shape == (B, L, config.D_MODEL), "z shape mismatch!"
    print("   ✓ Init state OK")
    
    print("\n4. Testing Latent Recursion (single cycle)...")
    y_new, z_new = model.trm_core.latent_recursion(memory, y, z, key_padding_mask=src_mask)
    print(f"   y_new shape: {y_new.shape}")
    print(f"   z_new shape: {z_new.shape}")
    assert y_new.shape == y.shape, "y shape changed unexpectedly!"
    assert z_new.shape == z.shape, "z shape changed unexpectedly!"
    print("   ✓ Latent recursion OK")
    
    print("\n5. Testing Deep Recursion...")
    (y_det, z_det), y_out, q_hat = model.trm_core.deep_recursion(
        memory, y, z, key_padding_mask=src_mask
    )
    print(f"   y_out shape: {y_out.shape}")
    print(f"   q_hat shape: {q_hat.shape}")  # [B, 1]
    print(f"   q_hat values: {q_hat.squeeze().tolist()}")
    assert q_hat.shape == (B, 1), "q_hat shape mismatch!"
    assert (q_hat >= 0).all() and (q_hat <= 1).all(), "q_hat not in [0, 1]!"
    print("   ✓ Deep recursion OK")
    
    print("\n6. Testing Reasoning Pooler & Enhanced Memory...")
    enhanced_memory, enhanced_mask = model.prepare_enhanced_memory(memory, y_out, src_mask)
    K = 4  # N_REASONING_TOKENS
    print(f"   Enhanced memory shape: {enhanced_memory.shape}")  # [B, K+L, D]
    assert enhanced_memory.shape == (B, K + L, config.D_MODEL), "Enhanced memory shape mismatch!"
    print("   ✓ Reasoning pooler OK")
    
    print("\n7. Testing Backward Pass (gradient flow)...")
    # Create dummy target and compute loss
    # Include q_hat in loss so Q-head gets gradients
    target = torch.randn_like(y_out)
    loss = F.mse_loss(y_out, target) + F.binary_cross_entropy(q_hat.squeeze(), torch.ones(B))
    loss.backward()
    
    # Check gradients exist on TRM core parameters
    sample_param = model.trm_core.layers[0].attn.in_proj_weight
    assert sample_param.grad is not None, "No gradient on TRM attention weights!"
    grad_norm = sample_param.grad.norm().item()
    print(f"   TRM attention grad norm: {grad_norm:.6f}")
    
    q_head_grad = model.trm_core.q_head.weight.grad
    assert q_head_grad is not None, "No gradient on Q-head!"
    print(f"   Q-head grad norm: {q_head_grad.norm().item():.6f}")
    print("   ✓ Gradients computed successfully")
    
    print("\n8. Testing EMA...")
    model.zero_grad()
    ema = EMAModel(model, decay=0.99)
    
    # Store original weight
    orig_weight = model.trm_core.layers[0].attn.in_proj_weight.data.clone()
    
    # Simulate a parameter update
    with torch.no_grad():
        model.trm_core.layers[0].attn.in_proj_weight.data += 0.1
    
    # Update EMA
    ema.update(model)
    
    # Check EMA shadow is different from original but between original and new
    shadow_weight = ema.shadow['trm_core.layers.0.attn.in_proj_weight']
    new_weight = model.trm_core.layers[0].attn.in_proj_weight.data
    
    # EMA should be: 0.01 * new + 0.99 * old
    expected = 0.01 * new_weight + 0.99 * orig_weight
    assert torch.allclose(shadow_weight, expected, atol=1e-5), "EMA update incorrect!"
    print("   ✓ EMA update OK")
    
    # Test apply_shadow and restore
    ema.apply_shadow(model)
    current = model.trm_core.layers[0].attn.in_proj_weight.data
    assert torch.allclose(current, shadow_weight), "apply_shadow failed!"
    print("   ✓ EMA apply_shadow OK")
    
    ema.restore(model)
    current = model.trm_core.layers[0].attn.in_proj_weight.data
    assert torch.allclose(current, new_weight), "EMA restore failed!"
    print("   ✓ EMA restore OK")
    
    print("\n" + "=" * 60)
    print("All TRM verification tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    verify_trm()
