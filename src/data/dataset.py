import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class TextDataset(Dataset):
    """
    Simple text dataset for language modeling
    """
    def __init__(self, tokenizer, data, max_length=1024):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()
        }


def get_dataloaders(
    dataset_name='wikitext',
    dataset_config='wikitext-103-v1',
    tokenizer_name='gpt2',
    max_length=1024,
    batch_size=8,
    num_workers=4
):
    """
    Create train/val dataloaders
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config)
    
    # Create datasets
    train_dataset = TextDataset(
        tokenizer,
        dataset['train'],
        max_length=max_length
    )
    
    val_dataset = TextDataset(
        tokenizer,
        dataset['validation'],
        max_length=max_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, tokenizer