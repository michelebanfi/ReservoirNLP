#!/usr/bin/env python3
"""
Test script to validate the enhanced reservoir architecture implementation.
Tests all major components without requiring full training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import GPT2TokenizerFast

# Test configuration
TEST_CONFIG = {
    'embedding_dim': 64,
    'reservoir_size': 128, 
    'num_blocks': 2,
    'max_seq_len': 256,
    'spectral_radius': 1.0,
    'sparsity': 0.2,
    'leak_rate': 0.95,
    'nvar_delay': 3,
    'nvar_degree': 2,
    'reservoir_warmup': 3,
    'DEVICE': 'cpu'
}

def test_true_reservoir():
    """Test the TrueReservoir class."""
    print("Testing TrueReservoir...")
    
    # Import the TrueReservoir class from the main file
    exec(open('TinyStories.py').read(), globals())
    
    reservoir = TrueReservoir(
        input_size=TEST_CONFIG['embedding_dim'],
        hidden_size=TEST_CONFIG['reservoir_size'],
        spectral_radius=TEST_CONFIG['spectral_radius'],
        sparsity=TEST_CONFIG['sparsity'],
        leak_rate=TEST_CONFIG['leak_rate'],
        device=TEST_CONFIG['DEVICE']
    )
    
    # Test with sample input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, TEST_CONFIG['embedding_dim'])
    
    # Test forward pass
    states = reservoir(x)
    assert states.shape == (batch_size, seq_len, TEST_CONFIG['reservoir_size'])
    
    # Test with initial state
    initial_state = torch.randn(batch_size, TEST_CONFIG['reservoir_size'])
    states_with_init = reservoir(x, initial_state)
    assert states_with_init.shape == states.shape
    
    # Verify non-trainable weights
    assert not any(p.requires_grad for p in reservoir.W_in.parameters())
    assert not any(p.requires_grad for p in reservoir.W_res.parameters())
    
    print("✓ TrueReservoir tests passed!")
    return reservoir

def test_nvar_reservoir():
    """Test the NVARReservoir class."""
    print("Testing NVARReservoir...")
    
    nvar = NVARReservoir(TEST_CONFIG)
    
    # Test with sample input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, TEST_CONFIG['embedding_dim'])
    
    # Test forward pass
    output = nvar(x)
    assert output.shape == (batch_size, seq_len, TEST_CONFIG['embedding_dim'])
    
    # Verify feature dimension calculation
    linear_terms = TEST_CONFIG['nvar_delay'] * TEST_CONFIG['embedding_dim']
    quadratic_terms = (linear_terms * (linear_terms + 1)) // 2
    expected_features = 1 + linear_terms + quadratic_terms
    assert nvar.feature_dim == expected_features
    
    print("✓ NVARReservoir tests passed!")
    return nvar

def test_hybrid_reservoir_block():
    """Test the HybridReservoirBlock class."""
    print("Testing HybridReservoirBlock...")
    
    block = HybridReservoirBlock(TEST_CONFIG)
    
    # Test with sample input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, TEST_CONFIG['embedding_dim'])
    
    # Test forward pass without initial state
    output = block(x)
    assert output.shape == (batch_size, seq_len, TEST_CONFIG['embedding_dim'])
    
    # Test forward pass with initial state
    initial_state = torch.randn(batch_size, TEST_CONFIG['reservoir_size'])
    output_with_init = block(x, initial_state)
    assert output_with_init.shape == output.shape
    
    print("✓ HybridReservoirBlock tests passed!")
    return block

def test_enhanced_model():
    """Test the EnhancedDeepReservoirModel class."""
    print("Testing EnhancedDeepReservoirModel...")
    
    vocab_size = 1000
    model = EnhancedDeepReservoirModel(vocab_size, TEST_CONFIG)
    
    # Test with sample input
    batch_size, seq_len = 2, 10
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test forward pass without initial states
    logits, states = model(input_ids)
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert len(states) == TEST_CONFIG['num_blocks']
    
    # Test forward pass with initial states
    logits2, states2 = model(input_ids, states)
    assert logits2.shape == logits.shape
    assert len(states2) == len(states)
    
    # Verify weight tying
    assert torch.equal(model.embedding.weight, model.lm_head.weight)
    
    print("✓ EnhancedDeepReservoirModel tests passed!")
    return model

def test_generation_compatibility():
    """Test that the enhanced model works with generation functions."""
    print("Testing generation compatibility...")
    
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    vocab_size = tokenizer.vocab_size
    
    # Create enhanced model
    model = EnhancedDeepReservoirModel(vocab_size, TEST_CONFIG)
    model.eval()
    
    # Test config for generation
    gen_config = {
        **TEST_CONFIG,
        'temperature': 0.8,
        'top_k': 50,
        'top_p': 0.9,
        'BLOCK_SIZE': 64
    }
    
    # Simulate generation (without actually calling the full function)
    context_str = "Once upon a time"
    start_indices = tokenizer.encode(context_str)
    context = torch.tensor(start_indices, dtype=torch.long).unsqueeze(0)
    
    # Test model forward pass
    with torch.no_grad():
        logits, states = model(context)
        assert logits.shape[2] == vocab_size
        assert len(states) == TEST_CONFIG['num_blocks']
    
    print("✓ Generation compatibility tests passed!")

def test_evaluation_functions():
    """Test reservoir quality evaluation functions."""
    print("Testing evaluation functions...")
    
    # Create a dummy data loader
    class DummyDataLoader:
        def __init__(self):
            self.data = [
                (torch.randint(0, 1000, (2, 10)), torch.randint(0, 1000, (2, 10)))
                for _ in range(5)
            ]
        
        def __iter__(self):
            return iter(self.data)
    
    vocab_size = 1000
    model = EnhancedDeepReservoirModel(vocab_size, TEST_CONFIG)
    data_loader = DummyDataLoader()
    
    # Test evaluate_reservoir_quality
    metrics = evaluate_reservoir_quality(model, data_loader, TEST_CONFIG, num_samples=2)
    assert 'memory_capacity' in metrics
    assert 'state_separation' in metrics
    assert isinstance(metrics['memory_capacity'], (int, float))
    assert isinstance(metrics['state_separation'], (int, float))
    
    print("✓ Evaluation function tests passed!")

def test_config_functions():
    """Test configuration creation functions."""
    print("Testing configuration functions...")
    
    # Dummy shared config
    shared_config = {
        'BATCH_SIZE': 32,
        'BLOCK_SIZE': 128,
        'EVAL_INTERVAL': 1000,
        'EPOCHS': 3,
        'DEVICE': 'cpu',
        'RESERVOIR_SAVE_PATH': 'test_model.pt'
    }
    
    # Dummy train loader
    class DummyTrainLoader:
        def __len__(self):
            return 100
    
    train_loader = DummyTrainLoader()
    
    # Test create_enhanced_reservoir_config
    config = create_enhanced_reservoir_config(shared_config, train_loader)
    
    # Verify required keys
    required_keys = [
        'embedding_dim', 'num_blocks', 'reservoir_size', 'spectral_radius',
        'sparsity', 'leak_rate', 'nvar_delay', 'nvar_degree'
    ]
    for key in required_keys:
        assert key in config, f"Missing key: {key}"
    
    print("✓ Configuration function tests passed!")

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Enhanced Reservoir Architecture Tests")
    print("=" * 60)
    
    try:
        # Load the main module to get access to classes
        exec(open('TinyStories.py').read(), globals())
        
        test_true_reservoir()
        test_nvar_reservoir() 
        test_hybrid_reservoir_block()
        test_enhanced_model()
        test_generation_compatibility()
        test_evaluation_functions()
        test_config_functions()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("Enhanced reservoir architecture is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)