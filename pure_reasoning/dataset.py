"""
Pure Reasoning Architecture - Dataset

Loads QA datasets formatted for span prediction.
Returns start/end positions instead of labels for generation.
"""
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer
from datasets import load_dataset


class SpanQADataset(Dataset):
    """
    QA dataset formatted for span prediction.
    
    Returns:
        input_ids: [L] tokenized [CLS] context [SEP] question [SEP]
        attention_mask: [L] 1=valid, 0=padding
        start_position: int, start of answer span in input_ids
        end_position: int, end of answer span in input_ids
        answer_type: 0=span, 1=yes, 2=no
        source: dataset name
    """
    def __init__(self, tokenizer, dataset_name, split, config, max_samples=None):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.config = config
        self.max_len = config.MAX_CONTEXT_LEN
        
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
                self.items.append({
                    'context': item['context'],
                    'question': item['question'],
                    'answer_text': item['answers']['text'][0] if item['answers']['text'] else "",
                    'answer_start': item['answers']['answer_start'][0] if item['answers']['answer_start'] else -1,
                    'answer_type': 0,  # span
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
                
                # Determine answer type
                answer = item['answer']
                if answer.lower() == 'yes':
                    answer_type = 1
                elif answer.lower() == 'no':
                    answer_type = 2
                else:
                    answer_type = 0  # span
                
                self.items.append({
                    'context': context[:4000],  # Truncate long contexts
                    'question': item['question'],
                    'answer_text': answer,
                    'answer_start': -1,  # Will find in tokenization
                    'answer_type': answer_type,
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
                    continue  # Skip if no extractable answer
                
                self.items.append({
                    'context': item['passage'],
                    'question': item['question'],
                    'answer_text': str(answer_text),
                    'answer_start': -1,
                    'answer_type': 0,  # span
                    'source': 'drop',
                })
        
        print(f"  Loaded {len(self.items)} samples from {name} ({split})")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        item = self.items[idx]
        
        # Tokenize: [CLS] context [SEP] question [SEP]
        context = item['context']
        question = item['question']
        answer_text = item['answer_text']
        
        # Encode with special tokens
        encoding = self.tokenizer(
            context,
            question,
            max_length=self.max_len,
            padding='max_length',
            truncation='only_first',  # Truncate context, keep question
            return_tensors='pt',
            return_offsets_mapping=True,
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        offset_mapping = encoding['offset_mapping'].squeeze(0)
        
        # Find answer span in tokenized input
        start_position = -1
        end_position = -1
        
        if item['answer_type'] == 0 and answer_text:  # Span answer
            # Use provided answer_start if available (SQuAD), otherwise search
            if item['answer_start'] >= 0:
                answer_start_char = item['answer_start']
            else:
                answer_start_char = context.lower().find(answer_text.lower())

            if answer_start_char != -1:
                answer_end_char = answer_start_char + len(answer_text)
                
                # Map character positions to token positions
                for i, (start_char, end_char) in enumerate(offset_mapping.tolist()):
                    if start_char is None or end_char is None:
                        continue
                    if start_char <= answer_start_char < end_char:
                        start_position = i
                    if start_char < answer_end_char <= end_char:
                        end_position = i
                        break
        
        # For span-type answers, clamp to valid range
        # For non-span (yes/no), keep -1 to mask in loss
        if item['answer_type'] == 0:  # span answer
            if start_position < 0:
                start_position = 0
            if end_position < 0 or end_position < start_position:
                end_position = start_position
        else:
            # Non-span: mark as -1 to ignore in span loss
            start_position = -1
            end_position = -1
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'start_position': torch.tensor(start_position),
            'end_position': torch.tensor(end_position),
            'answer_type': torch.tensor(item['answer_type']),
            'source': item['source'],
            'raw_question': question,
            'raw_answer': answer_text,
        }


def get_span_dataloaders(config):
    """Create train and validation dataloaders for span QA"""
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    
    train_datasets = []
    val_datasets = []
    
    samples_per_ds = config.SAMPLES_PER_DATASET
    val_per_ds = config.NUM_VAL_SAMPLES // len(config.DATASETS)
    
    for ds_name in config.DATASETS:
        print(f"Loading {ds_name}...")
        train_ds = SpanQADataset(tokenizer, ds_name, 'train', config, max_samples=samples_per_ds)
        val_ds = SpanQADataset(tokenizer, ds_name, 'validation', config, max_samples=val_per_ds)
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)
    
    combined_train = ConcatDataset(train_datasets)
    combined_val = ConcatDataset(val_datasets)
    
    print(f"\nTotal: {len(combined_train)} train, {len(combined_val)} val samples")
    
    def collate_fn(batch):
        return {
            'input_ids': torch.stack([x['input_ids'] for x in batch]),
            'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
            'start_positions': torch.stack([x['start_position'] for x in batch]),
            'end_positions': torch.stack([x['end_position'] for x in batch]),
            'answer_types': torch.stack([x['answer_type'] for x in batch]),
            'sources': [x['source'] for x in batch],
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
