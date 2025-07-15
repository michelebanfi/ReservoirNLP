import torch
import torch.nn as nn
import numpy as np
from reservoirpy.nodes import Reservoir

class ReservoirBlock(nn.Module):
    """A single block containing parallel reservoirs and a readout."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.reservoirs = []
        
        # Dynamically create reservoir instances based on config
        for res_config in config['reservoirs_per_block']:
            self.reservoirs.append(
                Reservoir(
                    units=res_config['reservoir_size'],
                    sr=res_config['spectral_radius'],
                    lr=res_config['leaking_rate']
                )
            )
        
        # The readout's input size is the sum of all reservoir sizes
        total_reservoir_size = sum(res['reservoir_size'] for res in config['reservoirs_per_block'])
        
        # The readout now outputs an update vector of size embedding_dim
        self.readout = nn.Sequential(
            nn.Linear(total_reservoir_size, config['readout_hidden_size']),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(config['readout_hidden_size'], config['embedding_dim']) # Output is an update vector
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        device = x.device
        batch_size, seq_len, _ = x.shape
        
        all_res_states = []
        
        # Process each sequence in the batch
        for i in range(batch_size):
            sequence_np = x[i].detach().cpu().numpy()
            
            # Get states from all parallel reservoirs for this one sequence
            states_for_sequence = []
            for reservoir in self.reservoirs:
                #.run() processes the whole sequence and returns all states
                states = reservoir.run(sequence_np, reset=True)
                states_for_sequence.append(torch.from_numpy(states).float())
            
            # Concatenate the states from the parallel reservoirs
            combined_states = torch.cat(states_for_sequence, dim=1)
            all_res_states.append(combined_states)
            
        # Stack the results for the whole batch
        batch_states = torch.stack(all_res_states).to(device)
        
        # The readout produces the update vector
        update_vector = self.readout(batch_states)
        return update_vector

class DeepReservoirModel(nn.Module):
    """A deep model composed of stacked ReservoirBlocks."""
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        
        # Initial token embedding
        self.embedding = nn.Embedding(vocab_size, config['embedding_dim'])
        
        # Create N stacked ReservoirBlocks
        self.blocks = nn.ModuleList([
            ReservoirBlock(config) for _ in range(config['num_blocks'])
        ])
        
        # Final layer to map from embedding space to vocabulary logits
        self.final_head = nn.Linear(config['embedding_dim'], vocab_size)

    def forward(self, idx):
        # idx shape: (batch_size, seq_len)
        x = self.embedding(idx) # Get initial embeddings
        
        # This is the core loop implementing the deep, residual architecture
        for block in self.blocks:
            update = block(x)
            x = x + update # Residual connection
            
        # Final prediction head
        logits = self.final_head(x)
        
        return logits