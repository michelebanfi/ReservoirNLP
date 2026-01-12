"""
Pure Reasoning Architecture - Dataset

Loads QA datasets formatted for text generation.
Returns encoder inputs and decoder targets (answer text).
"""
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer
from datasets import load_dataset


class GenerativeQADataset(Dataset):
    """
    QA dataset formatted for text generation.
    
    Returns:
        input_ids: [L] tokenized [CLS] context [SEP] question [SEP]
        attention_mask: [L] 1=valid, 0=padding
        decoder_input_ids: [T] answer tokens shifted right (starts with [CLS])
        labels: [T] answer tokens for loss (ends with [SEP])
        source: dataset name
        raw_question: original question text
        raw_answer: original answer text
    """
    def __init__(self, tokenizer, dataset_name, split, config, max_samples=None):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.config = config
        self.max_len = config.MAX_CONTEXT_LEN
        self.max_answer_len = config.MAX_ANSWER_LEN
        
        # Special tokens
        self.bos_id = tokenizer.cls_token_id  # [CLS] as BOS
        self.eos_id = tokenizer.sep_token_id  # [SEP] as EOS
        self.pad_id = tokenizer.pad_token_id
        
        # Load dataset
        self.items = []
        self._load_dataset(dataset_name, split, max_samples)
    
    def _load_dataset(self, name, split, max_samples):
        """Load and preprocess dataset"""
        if name == 'squad':
            ds = load_dataset('squad', split=split)
            for item in ds:
                if max_samples and len(self.items) >= max_samples:
                    break
                answer_text = item['answers']['text'][0] if item['answers']['text'] else ""
                if answer_text:  # Skip empty answers
                    self.items.append({
                        'context': item['context'],
                        'question': item['question'],
                        'answer_text': answer_text,
                        'source': 'squad',
                    })
        
        elif name == 'hotpotqa':
            ds = load_dataset('hotpot_qa', 'distractor', split=split, trust_remote_code=True)
            for item in ds:
                if max_samples and len(self.items) >= max_samples:
                    break
                
                # Concatenate supporting contexts
                context_parts = []
                for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                    context_parts.append(f"{title}: {' '.join(sentences)}")
                context = " ".join(context_parts)
                
                answer = item['answer']
                if answer:  # Skip empty
                    self.items.append({
                        'context': context[:4000],  # Truncate long contexts
                        'question': item['question'],
                        'answer_text': answer,
                        'source': 'hotpotqa',
                    })
        
        elif name == 'drop':
            ds = load_dataset('drop', split=split, trust_remote_code=True)
            for item in ds:
                if max_samples and len(self.items) >= max_samples:
                    break
                
                # Get first answer
                answers = item['answers_spans']
                if answers['spans']:
                    answer_text = answers['spans'][0]
                elif item['answer']['number']:
                    answer_text = item['answer']['number']
                else:
                    continue  # Skip if no answer
                
                self.items.append({
                    'context': item['passage'],
                    'question': item['question'],
                    'answer_text': str(answer_text),
                    'source': 'drop',
                })
        
        print(f"  Loaded {len(self.items)} samples from {name} ({split})")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        context = item['context']
        question = item['question']
        answer_text = item['answer_text']
        
        # Encode input: [CLS] context [SEP] question [SEP]
        encoding = self.tokenizer(
            context,
            question,
            max_length=self.max_len,
            padding='max_length',
            truncation='only_first',
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # Encode answer for decoder
        # Tokenizer returns: [CLS] answer_tokens [SEP] [PAD]...
        answer_encoding = self.tokenizer(
            answer_text,
            max_length=self.max_answer_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            add_special_tokens=False,  # Don't add [CLS]/[SEP], we handle them manually
        )
        answer_ids = answer_encoding['input_ids'].squeeze(0)  # Just the answer tokens
        answer_mask = answer_encoding['attention_mask'].squeeze(0)
        answer_len = answer_mask.sum().item()  # Actual answer token count
        
        # decoder_input_ids: [BOS] answer_tokens [PAD]...
        # labels:            answer_tokens [EOS] [PAD=-100]...
        # This is standard teacher forcing: predict next token given previous
        
        decoder_input_ids = torch.full((self.max_answer_len,), self.pad_id, dtype=torch.long)
        labels = torch.full((self.max_answer_len,), -100, dtype=torch.long)  # -100 = ignore in loss
        
        # decoder_input_ids starts with BOS, then answer tokens
        decoder_input_ids[0] = self.bos_id
        if answer_len > 0:
            copy_len = min(answer_len, self.max_answer_len - 1)
            decoder_input_ids[1:1+copy_len] = answer_ids[:copy_len]
        
        # labels are answer tokens shifted left (predict answer from BOS, then EOS)
        if answer_len > 0:
            copy_len = min(answer_len, self.max_answer_len - 1)
            labels[:copy_len] = answer_ids[:copy_len]
            labels[copy_len] = self.eos_id  # Final token is EOS
        else:
            labels[0] = self.eos_id  # Empty answer: just predict EOS
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'decoder_input_ids': decoder_input_ids,
            'labels': labels,
            'source': item['source'],
            'raw_question': question,
            'raw_answer': answer_text,
        }


def get_dataloaders(config):
    """Create train and validation dataloaders for generative QA"""
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    train_datasets = []
    val_datasets = []
    
    samples_per_ds = config.SAMPLES_PER_DATASET
    val_per_ds = config.NUM_VAL_SAMPLES // len(config.DATASETS)
    
    for ds_name in config.DATASETS:
        print(f"Loading {ds_name}...")
        train_ds = GenerativeQADataset(tokenizer, ds_name, 'train', config, max_samples=samples_per_ds)
        val_ds = GenerativeQADataset(tokenizer, ds_name, 'validation', config, max_samples=val_per_ds)
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
    
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    
    print(f"\nTotal: {len(combined_train)} train, {len(combined_val)} val samples")
    
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'decoder_input_ids': torch.stack([x['decoder_input_ids'] for x in batch]),
            'labels': torch.stack([x['labels'] for x in batch]),
            'sources': [x['source'] for x in batch],
            'raw_questions': [x['raw_question'] for x in batch],
            'raw_answers': [x['raw_answer'] for x in batch],
        }
    
    train_loader = DataLoader(
        combined_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        combined_val,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    
    return train_loader, val_loader, tokenizer
