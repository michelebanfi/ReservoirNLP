import torch
from tokenizers import Tokenizer
from datasets import load_dataset
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer

class DataHandler:
    """Handles dataset loading, tokenization, and batching."""
    def __init__(self, config):
        self.config = config
        self.batch_size = config['batch_size']
        self.block_size = config['block_size']
        self.tokenizer = self._train_tokenizer()
        
        # Tokenize and split data
        n = len(self.raw_indices)
        train_end = int(0.9 * n)
        val_end = int(0.95 * n)
        
        self.train_data = torch.tensor(self.raw_indices[:train_end], dtype=torch.long)
        self.val_data = torch.tensor(self.raw_indices[train_end:val_end], dtype=torch.long)
        self.test_data = torch.tensor(self.raw_indices[val_end:], dtype=torch.long)
        
        print(f"Train tokens: {len(self.train_data)}, Val tokens: {len(self.val_data)}, Test tokens: {len(self.test_data)}")

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
        if split_name == 'train': return self.train_data
        if split_name == 'val': return self.val_data
        return self.test_data

    def get_batch(self, split):
        """Generates a random batch of data of size (batch_size, block_size)."""
        data = self._get_split_indices(split)
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y