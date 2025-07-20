import torch
from tokenizers import Tokenizer
from datasets import load_dataset
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.decoders import WordPiece as WordPieceDecoder

class DataHandler:
    """Handles dataset loading, tokenization, and batching."""
    def __init__(self, config):
        self.config = config
        self.batch_size = config['batch_size']
        self.block_size = config['block_size']
        
        # This method is now corrected
        self.tokenizer = self._get_or_train_tokenizer()
        
        # Load the full dataset splits
        train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        val_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

        # Tokenize each split separately on the full text
        # Filter out empty lines
        train_text = [text for text in train_dataset['text'] if text.strip()]
        val_text = [text for text in val_dataset['text'] if text.strip()]
        test_text = [text for text in test_dataset['text'] if text.strip()]

        # Encode the text and flatten into a single list of token IDs
        self.train_data = torch.tensor(self._flatten_tokens(self.tokenizer.encode_batch(train_text)), dtype=torch.long)
        self.val_data = torch.tensor(self._flatten_tokens(self.tokenizer.encode_batch(val_text)), dtype=torch.long)
        self.test_data = torch.tensor(self._flatten_tokens(self.tokenizer.encode_batch(test_text)), dtype=torch.long)
        
        print(f"Data loaded. Train tokens: {len(self.train_data):,}, Val tokens: {len(self.val_data):,}, Test tokens: {len(self.test_data):,}")

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def _flatten_tokens(self, encoded_batch):
        """Helper to flatten the output of tokenizer.encode_batch."""
        return [id for encoding in encoded_batch for id in encoding.ids]

    def _get_or_train_tokenizer(self):
        print("Loading dataset and training tokenizer...")
        # We only use the training split to train the tokenizer
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
        # Define a generator to feed text to the trainer efficiently,
        # filtering out empty strings. This is the correct way.
        def text_iterator():
            for text in dataset['text']:
                if text.strip():
                    yield text
        
        tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
        tokenizer.decoder = WordPieceDecoder(prefix="##")
        trainer = WordPieceTrainer(vocab_size=self.config['vocab_size'], special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"])
        
        # Train the tokenizer on the full text iterator
        tokenizer.train_from_iterator(text_iterator(), trainer)
        
        print(f"Tokenization complete. Vocab size: {tokenizer.get_vocab_size()}")
        return tokenizer

    def _get_split_indices(self, split_name):
        if split_name == 'train': return self.train_data
        if split_name == 'val': return self.val_data
        return self.test_data

    def get_batch(self, split):
        """Generates a random batch of data of size (batch_size, block_size)."""
        data = self._get_split_indices(split)
        # Ensure we don't go out of bounds
        max_start_index = len(data) - self.block_size - 1
        # Handle cases where the dataset split is too small
        if max_start_index <= 0:
            raise ValueError(f"'{split}' split is too small for the given block_size.")
            
        ix = torch.randint(max_start_index, (self.batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        return x, y