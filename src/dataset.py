import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from datasets import load_dataset
from .config import Config


class UnifiedQADataset(Dataset):
    """
    Unified dataset that normalizes different QA datasets into a common format.
    Tracks source dataset for per-dataset metrics.
    """
    def __init__(self, tokenizer, dataset_name, split='train', max_samples=None):
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.max_src_len = Config.MAX_SRC_LEN
        self.max_tgt_len = Config.MAX_TGT_LEN
        
        # Load and normalize data based on dataset type
        self.data = self._load_dataset(dataset_name, split, max_samples)
    
    def _load_dataset(self, name, split, max_samples):
        """Load dataset and return list of normalized dicts."""
        try:
            if name == 'squad':
                return self._load_squad(split, max_samples)
            elif name == 'hotpotqa':
                return self._load_hotpotqa(split, max_samples)
            elif name == 'drop':
                return self._load_drop(split, max_samples)
            else:
                raise ValueError(f"Unknown dataset: {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
            return []
    
    def _load_squad(self, split, max_samples):
        """Load SQuAD dataset."""
        ds = load_dataset('rajpurkar/squad', split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        data = []
        for item in ds:
            context = item['context'][:2000]  # Truncate long contexts
            question = item['question']
            answers = item['answers']['text']
            answer = answers[0] if answers else ""
            
            data.append({
                'question': question,
                'context': context,
                'answer': answer,
                'source': 'squad',
                'difficulty': 'easy'
            })
        return data
    
    def _load_hotpotqa(self, split, max_samples):
        """
        Load HotpotQA dataset (distractor setting).
        Context is nested: context['sentences'] is list of lists, context['title'] is list.
        """
        # HotpotQA uses 'train' and 'validation' splits
        ds = load_dataset('hotpotqa/hotpot_qa', 'distractor', split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        data = []
        for item in ds:
            question = item['question']
            answer = item['answer']
            
            # Flatten nested context: list of (title, sentences) pairs
            context_parts = []
            titles = item['context']['title']
            sentences_list = item['context']['sentences']
            
            for title, sentences in zip(titles, sentences_list):
                # sentences is a list of strings
                paragraph = f"{title}: {' '.join(sentences)}"
                context_parts.append(paragraph)
            
            context = ' '.join(context_parts)[:2000]  # Truncate
            
            data.append({
                'question': question,
                'context': context,
                'answer': answer,
                'source': 'hotpotqa',
                'difficulty': item.get('level', 'medium')
            })
        return data
    
    def _load_drop(self, split, max_samples):
        """
        Load DROP dataset.
        Has passage, question, answers_spans.spans (list of answer strings).
        """
        ds = load_dataset('ucinlp/drop', split=split)
        if max_samples:
            ds = ds.select(range(min(max_samples, len(ds))))
        
        data = []
        for item in ds:
            question = item['question']
            context = item['passage'][:2000]  # Truncate
            
            # answers_spans is a dict with 'spans' key containing list of answers
            spans = item['answers_spans']['spans']
            answer = spans[0] if spans else ""
            
            data.append({
                'question': question,
                'context': context,
                'answer': answer,
                'source': 'drop',
                'difficulty': 'hard'
            })
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        input_text = f"question: {item['question']} context: {item['context']}"
        
        source = self.tokenizer(
            input_text,
            max_length=self.max_src_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target = self.tokenizer(
            item['answer'],
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
            'raw_question': item['question'],
            'raw_answer': item['answer'],
            'source': item['source'],
            'difficulty': item['difficulty'],
        }


def get_dataloaders(tokenizer, config):
    """
    Create dataloaders with balanced sampling from all configured datasets.
    """
    train_datasets = []
    val_datasets = []
    
    samples_per_dataset = config.SAMPLES_PER_DATASET
    
    for dataset_name in config.DATASETS:
        print(f"Loading {dataset_name}...")
        
        # For validation, use fewer samples
        val_samples = min(samples_per_dataset // 10, 500)
        
        train_ds = UnifiedQADataset(
            tokenizer, 
            dataset_name, 
            split='train', 
            max_samples=samples_per_dataset
        )
        
        # HotpotQA and DROP use 'validation', SQuAD uses 'validation' too
        val_ds = UnifiedQADataset(
            tokenizer, 
            dataset_name, 
            split='validation', 
            max_samples=val_samples
        )
        
        if len(train_ds) > 0:
            train_datasets.append(train_ds)
            print(f"  Loaded {len(train_ds)} train samples from {dataset_name}")
        
        if len(val_ds) > 0:
            val_datasets.append(val_ds)
            print(f"  Loaded {len(val_ds)} val samples from {dataset_name}")
    
    # Combine all datasets
    combined_train = ConcatDataset(train_datasets) if train_datasets else []
    combined_val = ConcatDataset(val_datasets) if val_datasets else []
    
    print(f"\nTotal: {len(combined_train)} train, {len(combined_val)} val samples")
    
    train_loader = DataLoader(
        combined_train,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        combined_val, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False
    )
    
    return train_loader, val_loader


# Legacy function for backward compatibility with validation baseline
def get_baseline_validation_samples(tokenizer, config, num_samples=10):
    """
    Get a fixed set of validation samples for baseline comparison.
    Returns samples from all datasets for comprehensive evaluation.
    """
    samples = []
    samples_per_ds = num_samples // len(config.DATASETS) + 1
    
    for dataset_name in config.DATASETS:
        ds = UnifiedQADataset(tokenizer, dataset_name, 'validation', samples_per_ds)
        for i in range(min(samples_per_ds, len(ds))):
            samples.append(ds.data[i])
        
        if len(samples) >= num_samples:
            break
    
    return samples[:num_samples]
