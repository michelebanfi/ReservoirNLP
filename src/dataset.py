import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from .config import Config

class SQuADDataset(Dataset):
    def __init__(self, tokenizer, split='train', max_samples=None):
        self.tokenizer = tokenizer
        try:
            self.data = load_dataset('rajpurkar/squad', split=split)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Fallback to local/dummy or check internet connection.")
            self.data = []

        if max_samples and len(self.data) > 0:
            indices = list(range(min(max_samples, len(self.data))))
            self.data = self.data.select(indices)
        
        self.max_src_len = Config.MAX_SRC_LEN
        self.max_tgt_len = Config.MAX_TGT_LEN
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context'][:1500]
        question = item['question']
        answers = item['answers']['text']
        answer = answers[0] if answers else ""
        
        input_text = f"question: {question} context: {context}"
        
        source = self.tokenizer(
            input_text,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            answer,
            max_length=self.max_tgt_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source.input_ids.squeeze(),
            'attention_mask': source.attention_mask.squeeze(),
            'labels': labels,
            'raw_question': question,
            'raw_answer': answer,
        }

def get_dataloaders(tokenizer, config):
    train_dataset = SQuADDataset(tokenizer, 'train', config.TRAIN_SIZE)
    val_dataset = SQuADDataset(tokenizer, 'validation', config.VAL_SIZE)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader
