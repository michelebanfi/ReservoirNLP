# data_handler.py

import torch
import requests
import os

class DataHandler:
    """
    Handles the Tiny Shakespeare dataset using a character-level tokenizer.
    """
    def __init__(self, config):
        self.config = config
        self.batch_size = config['batch_size']
        self.block_size = config['block_size']
        
        # Download and load the dataset text
        text = self._download_and_load_shakespeare()
        
        # --- Character-level Tokenizer ---
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        
        self._stoi = { ch:i for i,ch in enumerate(chars) }
        self._itos = { i:ch for i,ch in enumerate(chars) }
        
        # Public encoder and decoder methods
        self.encode = lambda s: [self._stoi[c] for c in s]
        self.decode = lambda l: ''.join([self._itos[i] for i in l])
        
        # --- Data Splits ---
        all_data = torch.tensor(self.encode(text), dtype=torch.long)
        n = len(all_data)
        train_end = int(0.9 * n)
        
        self.train_data = all_data[:train_end]
        self.val_data = all_data[train_end:]
        # Use a subset of validation for the test set
        self.test_data = self.val_data[:len(self.val_data) // 2]
        
        print(f"Tiny Shakespeare loaded. Vocab size: {self.vocab_size} (character-level)")
        print(f"Train tokens: {len(self.train_data):,}, Val tokens: {len(self.val_data):,}")

    def _download_and_load_shakespeare(self):
        file_path = "tinyshakespeare.txt"
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        
        if not os.path.exists(file_path):
            print(f"Downloading {file_path}...")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(requests.get(url).text)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def get_vocab_size(self):
        return self.vocab_size

    def _get_split_indices(self, split_name):
        if split_name == 'train': return self.train_data
        if split_name == 'val': return self.val_data
        return self.test_data

    def get_batch(self, split):
        data = self._get_split_indices(split)
        max_start_index = len(data) - self.block_size - 1
        ix = torch.randint(max_start_index, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y