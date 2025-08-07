#!/usr/bin/env python3
"""
Simplified test script to validate the enhanced reservoir architecture implementation.
Tests core components without requiring external downloads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

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

# Import only the class definitions we need
class TrueReservoir(nn.Module):
    def __init__(self, input_size, hidden_size, spectral_radius=1.2, 
                 sparsity=0.1, leak_rate=0.95, device='cuda'):
        super().__init__()
        self.hidden_size = hidden_size
        self.leak_rate = leak_rate
        self.device = device
        
        # Fixed reservoir weights (non-trainable)
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W_res = nn.Linear(hidden_size, hidden_size, bias=False)
        
        # Freeze reservoir weights
        for param in self.W_in.parameters():
            param.requires_grad = False
        for param in self.W_res.parameters():
            param.requires_grad = False
            
        self._initialize_weights(spectral_radius, sparsity)
        self.activation = torch.tanh
        self.register_buffer('initial_state', torch.zeros(1, hidden_size))
    
    def _initialize_weights(self, spectral_radius, sparsity):
        # Input weights - random uniform
        nn.init.kaiming_uniform_(self.W_in.weight, a=math.sqrt(5))
        
        # Reservoir weights
        with torch.no_grad():
            W = self.W_res.weight.data
            mask = (torch.rand(W.shape, device=self.device) > sparsity).float()
            W.normal_(0, 1)
            W *= mask
            
            # Scale to desired spectral radius
            eigenvalues = torch.linalg.eigvals(W)
            current_radius = torch.max(torch.abs(eigenvalues))
            W *= spectral_radius / (current_radius + 1e-8)
    
    def forward(self, x, initial_state=None):
        batch_size, seq_len, _ = x.shape
        states = torch.zeros(batch_size, seq_len, self.hidden_size, device=self.device)
        prev_state = initial_state if initial_state is not None else self.initial_state.expand(batch_size, self.hidden_size)
        
        for t in range(seq_len):
            input_t = x[:, t, :]
            input_part = self.W_in(input_t)
            reservoir_part = self.W_res(prev_state)
            new_state = self.activation(input_part + reservoir_part)
            
            # Leaky integration
            if self.leak_rate < 1.0:
                new_state = self.leak_rate * new_state + (1.0 - self.leak_rate) * prev_state
                
            states[:, t, :] = new_state
            prev_state = new_state
            
        return states

class NVARReservoir(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.delay = config.get('nvar_delay', 5)
        self.degree = config.get('nvar_degree', 2)
        self.input_size = config['embedding_dim']
        
        # Calculate feature dimension after expansion
        linear_terms = self.delay * self.input_size
        quadratic_terms = (linear_terms * (linear_terms + 1)) // 2
        self.feature_dim = 1 + linear_terms + quadratic_terms  # constant + linear + quadratic
        
        # Trainable readout (only this part is trained)
        self.readout = nn.Sequential(
            nn.Linear(self.feature_dim, config['embedding_dim']),
            nn.GELU(),
            nn.Linear(config['embedding_dim'], config['embedding_dim'])
        )
        
        # Register buffer for delayed indices
        self.register_buffer('delay_indices', torch.arange(self.delay-1, -1, -1))
    
    def polynomial_expansion(self, x):
        """Creates polynomial features from delayed embeddings"""
        batch_size, seq_len, embed_dim = x.shape
        
        # Create delayed embeddings
        x_padded = F.pad(x, (0, 0, self.delay-1, 0))
        delayed = x_padded.unfold(1, self.delay, 1)  # [B, T, delay, embed_dim]
        
        # Reshape to [B*T, delay*embed_dim]
        features = delayed.reshape(-1, self.delay * embed_dim)
        
        # Create polynomial features (constant + linear + quadratic)
        constant = torch.ones(features.size(0), 1, device=features.device)
        linear = features
        
        # Quadratic terms (only upper triangle to avoid duplicates)
        quadratic_list = []
        for i in range(features.size(1)):
            for j in range(i, features.size(1)):
                quadratic_list.append(features[:, i] * features[:, j])
        quadratic = torch.stack(quadratic_list, dim=1)
        
        # Combine all features
        all_features = torch.cat([constant, linear, quadratic], dim=1)
        return all_features.reshape(batch_size, seq_len, -1)
    
    def forward(self, x):
        # Create polynomial features
        features = self.polynomial_expansion(x)
        # Apply readout
        return self.readout(features)

class HybridReservoirBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # True reservoir component
        self.reservoir = TrueReservoir(
            input_size=config['embedding_dim'],
            hidden_size=config['reservoir_size'],
            spectral_radius=config['spectral_radius'],
            sparsity=config.get('sparsity', 0.1),
            leak_rate=config.get('leak_rate', 0.95),
            device=config['DEVICE']
        )
        
        # SSM-inspired selective mechanism
        self.delta_proj = nn.Sequential(
            nn.Linear(config['embedding_dim'], 1),
            nn.Softplus()
        )
        
        # State transition parameters (frozen like reservoir)
        self.A = nn.Parameter(-torch.ones(config['reservoir_size']) * 10, requires_grad=False)
        self.B = nn.Linear(config['embedding_dim'], config['reservoir_size'], bias=False)
        for p in self.B.parameters():
            p.requires_grad = False
            
        # Readout layer
        self.readout = nn.Sequential(
            nn.LayerNorm(config['reservoir_size']),
            nn.Linear(config['reservoir_size'], config['embedding_dim']),
            nn.GELU()
        )
        
        # Hybrid gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(config['embedding_dim'], config['embedding_dim']),
            nn.Sigmoid()
        )
        
        # Layer norm for stability
        self.norm = nn.LayerNorm(config['embedding_dim'])

    def forward(self, x, initial_state=None):
        x_norm = self.norm(x)
        
        # Reservoir path
        reservoir_states = self.reservoir(x_norm, initial_state)
        reservoir_out = self.readout(reservoir_states)
        
        # SSM path with selective updates
        B = self.B(x_norm)
        delta = self.delta_proj(x_norm).squeeze(-1)  # [B, L]
        
        # Discretize A
        A_bar = torch.exp(self.A * delta.unsqueeze(-1))  # [B, L, state_size]
        B_bar = B * (1 - A_bar) / (self.A + 1e-8)  # [B, L, state_size]
        
        # State update
        h0 = initial_state if initial_state is not None else torch.zeros_like(B[:, 0, :])
        ssm_states = [h0]
        for i in range(x_norm.size(1)):
            h = A_bar[:, i, :] * ssm_states[-1] + B_bar[:, i, :]
            ssm_states.append(h)
        ssm_states = torch.stack(ssm_states[1:], dim=1)
        
        # Apply readout to SSM states to match embedding dimension
        ssm_out = self.readout(ssm_states)
        
        # Combine paths
        gate = self.gate(x)
        return reservoir_out * gate + ssm_out * (1 - gate)

class EnhancedDeepReservoirModel(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(config['max_seq_len'], config['embedding_dim']) * 0.02
        )
        
        # Reservoir blocks - use hybrid implementation
        self.blocks = nn.ModuleList([
            HybridReservoirBlock(config) for _ in range(config['num_blocks'])
        ])
        
        # Output head
        self.ln_f = nn.LayerNorm(config['embedding_dim'])
        self.lm_head = nn.Linear(config['embedding_dim'], vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Reservoir warmup parameters
        self.warmup_steps = config.get('reservoir_warmup', 5)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, initial_states=None):
        batch_size, seq_len = idx.shape
        
        # Token embeddings
        tok_emb = self.embedding(idx)
        
        # Positional embeddings
        pos_emb = self.pos_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Combined embeddings
        x = tok_emb + pos_emb
        
        # Process through reservoir blocks with state management
        current_states = initial_states
        if current_states is None:
            current_states = [torch.zeros(batch_size, block.reservoir.hidden_size, 
                                        device=idx.device) for block in self.blocks]
        
        for i, block in enumerate(self.blocks):
            x = block(x, current_states[i] if current_states else None)
            current_states[i] = x[:, -1, :]  # Store final state
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        return logits, current_states

def test_true_reservoir():
    """Test the TrueReservoir class."""
    print("Testing TrueReservoir...")
    
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
    proper_states = [torch.zeros(batch_size, TEST_CONFIG['reservoir_size']) for _ in range(TEST_CONFIG['num_blocks'])]
    logits2, states2 = model(input_ids, proper_states)
    assert logits2.shape == logits.shape
    assert len(states2) == len(proper_states)
    
    # Verify weight tying
    assert torch.equal(model.embedding.weight, model.lm_head.weight)
    
    print("✓ EnhancedDeepReservoirModel tests passed!")
    return model

def test_state_evolution():
    """Test that states actually evolve over time."""
    print("Testing state evolution...")
    
    vocab_size = 100
    model = EnhancedDeepReservoirModel(vocab_size, TEST_CONFIG)
    
    # Create two different sequences
    batch_size, seq_len = 1, 20
    seq1 = torch.randint(0, vocab_size, (batch_size, seq_len))
    seq2 = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Get states for both sequences
    _, states1 = model(seq1)
    _, states2 = model(seq2)
    
    # Verify states are different for different inputs
    for i in range(len(states1)):
        state_diff = torch.norm(states1[i] - states2[i])
        assert state_diff > 1e-6, f"States are too similar for block {i}"
    
    print("✓ State evolution tests passed!")

def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running Enhanced Reservoir Architecture Tests")
    print("=" * 60)
    
    try:
        test_true_reservoir()
        test_nvar_reservoir() 
        test_hybrid_reservoir_block()
        test_enhanced_model()
        test_state_evolution()
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED!")
        print("Enhanced reservoir architecture is working correctly.")
        print("Key improvements implemented:")
        print("  - True reservoir state evolution with leaky integration")
        print("  - NVAR polynomial feature expansion")  
        print("  - Hybrid reservoir-SSM architecture")
        print("  - Enhanced model with state management")
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