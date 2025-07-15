import torch
import numpy as np
import torch.nn as nn
from tokenizers import Tokenizer
from datasets import load_dataset
from reservoirpy.nodes import Reservoir
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

class DataAndStateGenerator:
    """
    Handles all one-time data loading, tokenization, embedding, and
    the pre-computation of reservoir states for a dynamic ensemble.
    """
    def __init__(self, config):
        self.config = config
        self.tokenizer = self._train_tokenizer()
        self.embedding = nn.Embedding(self.get_vocab_size(), config['embedding_dim'])
        
        # Determine the maximum window size needed across all reservoirs
        self.max_window = max(res['window_size'] for res in config['reservoirs'])
        
        # Pre-compute all states and targets
        self.train_states, self.train_targets = self._process_split('train')
        self.val_states, self.val_targets = self._process_split('val')
        self.test_states, self.test_targets = self._process_split('test')

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def _train_tokenizer(self):
        print("Loading dataset and training tokenizer...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        text_data = " ".join([text for text in dataset["text"] if text.strip()]).split()[:self.config['max_words']]
        
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        trainer = WordPieceTrainer(vocab_size=self.config['vocab_size'], special_tokens=["[UNK]"])
        tokenizer.train_from_iterator(text_data, trainer)
        
        self.raw_indices = tokenizer.encode_batch(text_data)
        self.raw_indices = [id for encoding in self.raw_indices for id in encoding.ids]
        print(f"Tokenization complete. Vocab size: {tokenizer.get_vocab_size()}")
        return tokenizer

    def _get_split_indices(self, split_name):
        n = len(self.raw_indices)
        train_end = int(0.9 * n)
        val_end = int(0.95 * n)
        
        if split_name == 'train':
            return self.raw_indices[:train_end]
        elif split_name == 'val':
            return self.raw_indices[train_end:val_end]
        else: # test
            return self.raw_indices[val_end:]

    def _generate_states_from_sequence(self, reservoir, sequence, window_size):
        """Generates states using a sliding window."""
        if len(sequence) < window_size:
            return None
        
        states = []
        for i in range(window_size, len(sequence)):
            window_np = sequence[i - window_size:i].detach().numpy()
            state = reservoir.run(window_np, reset=False)
            states.append(state[-1])
        
        return torch.tensor(np.stack(states), dtype=torch.float32) if states else None

    def _process_split(self, split_name):
        """The main pre-computation function for a data split."""
        print(f"--- Processing {split_name} split ---")
        indices = self._get_split_indices(split_name)
        if not indices or len(indices) <= self.max_window:
            print(f"Not enough data for {split_name} split with max window size {self.max_window}. Skipping.")
            return None, None

        embedded_sequence = self.embedding(torch.tensor(indices))
        
        all_split_states = []
        # Iterate through each reservoir defined in the config
        for res_config in self.config['reservoirs']:
            print(f"  Generating states for reservoir '{res_config['name']}'...")
            reservoir = Reservoir(
                units=res_config['reservoir_size'],
                sr=res_config['spectral_radius'],
                lr=res_config['leaking_rate']
            )
            
            # Generate states using a sliding window
            states = self._generate_states_from_sequence(reservoir, embedded_sequence, res_config['window_size'])
            
            if states is not None:
                all_split_states.append(states)

        if not all_split_states:
            return None, None
            
        # Align all state sequences to the same length
        min_len = min(s.shape[0] for s in all_split_states)
        aligned_states = [s[-min_len:] for s in all_split_states]
        
        # Concatenate states from all reservoirs
        combined_states = torch.cat(aligned_states, dim=1)
        
        # Align targets based on the longest window to ensure valid context for all
        target_indices = torch.tensor(indices[self.max_window:self.max_window + min_len], dtype=torch.long)
        
        print(f"Generated {combined_states.shape} state-target pairs for {split_name} split from {len(all_split_states)} reservoirs.")
        return combined_states, target_indices