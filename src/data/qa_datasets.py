"""
Dataset loaders for NarrativeQA and Natural Questions (closed-book)

These datasets are better for evaluation than WikiText because they test:
- Reading comprehension (NarrativeQA)
- Question answering (Natural Questions)
- Long-context understanding (both)
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Optional, Tuple, Dict
import random

import os

os.environ["HF_DATASETS_CACHE"] = "/projectnb/cs523aw/students/waqar/hf_cache"
os.environ["HF_HOME"] = "/projectnb/cs523aw/students/waqar/hf_home"
os.environ["TRANSFORMERS_CACHE"] = "/projectnb/cs523aw/students/waqar/transformers_cache"
os.environ["TMPDIR"] = "/projectnb/cs523aw/students/waqar/tmp"

class NarrativeQADataset(Dataset):
    """
    NarrativeQA dataset for long-form reading comprehension
    
    Dataset: https://huggingface.co/datasets/narrativeqa
    
    Format:
    - Story summaries (long context)
    - Questions about the story
    - Answers to questions
    
    Good for testing:
    - Long-context understanding
    - Reading comprehension
    - Routing on important tokens (questions vs context)
    """
    def __init__(
        self,
        tokenizer,
        split='train',
        max_length=2048,
        use_summary=True,
        max_samples=None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_summary = use_summary
        
        # Load dataset
        print(f"Loading NarrativeQA ({split})...")
        dataset = load_dataset('narrativeqa', split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.data = []
        for example in dataset:
            # Get context (summary or full document)
            if use_summary:
                context = example['document']['summary']['text']
            else:
                context = example['document']['text']
            
            # Get question and answer
            question = example['question']['text']
            # NarrativeQA has multiple answers, use the first one
            answer = example['answers'][0]['text'] if example['answers'] else ""
            
            # Format: Context + Question + Answer
            text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
            
            self.data.append({
                'text': text,
                'context': context,
                'question': question,
                'answer': answer
            })
        
        print(f"✓ Loaded {len(self.data)} NarrativeQA examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'context': item['context'],
            'question': item['question'],
            'answer': item['answer']
        }


class NaturalQuestionsDataset(Dataset):
    """
    Natural Questions dataset (closed-book QA)
    
    Dataset: https://huggingface.co/datasets/natural_questions
    
    Format:
    - Question from real Google searches
    - Short answer (closed-book: no context provided)
    
    Good for testing:
    - Question answering without context
    - Knowledge retention
    - Compact answer generation
    """
    def __init__(
        self,
        tokenizer,
        split='train',
        max_length=512,
        max_samples=None,
        include_context=False  # For open-book variant
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_context = include_context
        
        # Load dataset (using the smaller validation set for faster loading)
        print(f"Loading Natural Questions ({split})...")
        
        # Natural Questions is very large, so we use a subset
        if split == 'train':
            dataset = load_dataset('natural_questions', split='train')
        else:
            dataset = load_dataset('natural_questions', split='validation')
        
        if max_samples:
            # Random sample for diversity
            indices = random.sample(range(len(dataset)), min(max_samples, len(dataset)))
            dataset = dataset.select(indices)
        
        self.data = []
        for example in dataset:
            question = example['question']['text']
            
            # Get short answer
            annotations = example['annotations']
            if annotations and len(annotations) > 0:
                short_answers = annotations[0]['short_answers']
                if short_answers and len(short_answers) > 0:
                    # Extract answer text from document tokens
                    start_token = short_answers[0]['start_token']
                    end_token = short_answers[0]['end_token']
                    
                    # Get tokens
                    tokens = example['document']['tokens']
                    answer_tokens = tokens['token'][start_token:end_token]
                    answer = ' '.join(answer_tokens)
                    
                    # Optionally include context for open-book QA
                    if self.include_context:
                        # Get surrounding context
                        context_start = max(0, start_token - 50)
                        context_end = min(len(tokens['token']), end_token + 50)
                        context_tokens = tokens['token'][context_start:context_end]
                        context = ' '.join(context_tokens)
                        
                        text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
                    else:
                        # Closed-book: just question and answer
                        text = f"Question: {question}\n\nAnswer: {answer}"
                    
                    self.data.append({
                        'text': text,
                        'question': question,
                        'answer': answer
                    })
        
        print(f"✓ Loaded {len(self.data)} Natural Questions examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'question': item['question'],
            'answer': item['answer']
        }


class SquadDataset(Dataset):
    """
    SQuAD 2.0 dataset (easier to use, faster loading)
    
    Dataset: https://huggingface.co/datasets/squad_v2
    
    Good alternative if NQ/NarrativeQA are too slow to load
    """
    def __init__(
        self,
        tokenizer,
        split='train',
        max_length=1024,
        max_samples=None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load dataset
        print(f"Loading SQuAD 2.0 ({split})...")
        dataset = load_dataset('squad_v2', split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.data = []
        for example in dataset:
            context = example['context']
            question = example['question']
            
            # SQuAD 2.0 includes unanswerable questions
            if example['answers']['text']:
                answer = example['answers']['text'][0]
            else:
                answer = "[No answer]"
            
            text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: {answer}"
            
            self.data.append({
                'text': text,
                'context': context,
                'question': question,
                'answer': answer
            })
        
        print(f"✓ Loaded {len(self.data)} SQuAD examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        encoding = self.tokenizer(
            item['text'],
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone(),
            'context': item['context'],
            'question': item['question'],
            'answer': item['answer']
        }


def get_qa_dataloaders(
    dataset_name: str = 'narrativeqa',
    tokenizer_name: str = 'gpt2',
    max_length: int = 2048,
    batch_size: int = 4,
    num_workers: int = 4,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    """
    Get dataloaders for QA datasets
    
    Args:
        dataset_name: 'narrativeqa', 'natural_questions', or 'squad'
        tokenizer_name: HuggingFace tokenizer name
        max_length: Maximum sequence length
        batch_size: Batch size
        num_workers: Number of data loading workers
        max_train_samples: Limit training samples (for faster loading)
        max_val_samples: Limit validation samples
        **dataset_kwargs: Additional dataset-specific arguments
    
    Returns:
        train_loader, val_loader, tokenizer
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Select dataset class
    if dataset_name == 'narrativeqa':
        DatasetClass = NarrativeQADataset
        default_kwargs = {'use_summary': True}
    elif dataset_name == 'natural_questions':
        DatasetClass = NaturalQuestionsDataset
        default_kwargs = {'include_context': False}  # Closed-book by default
    elif dataset_name == 'squad':
        DatasetClass = SquadDataset
        default_kwargs = {}
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Merge kwargs
    dataset_kwargs = {**default_kwargs, **dataset_kwargs}
    
    # Create datasets
    print(f"\nLoading {dataset_name} dataset...")
    
    train_dataset = DatasetClass(
        tokenizer=tokenizer,
        split='train',
        max_length=max_length,
        max_samples=max_train_samples,
        **dataset_kwargs
    )
    
    val_dataset = DatasetClass(
        tokenizer=tokenizer,
        split='validation',
        max_length=max_length,
        max_samples=max_val_samples,
        **dataset_kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None
    )
    
    print(f"\n✓ Dataset loaded:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {max_length}")
    
    return train_loader, val_loader, tokenizer


# Example usage
if __name__ == '__main__':
    # Test loading datasets
    
    print("\n" + "="*60)
    print("Testing NarrativeQA")
    print("="*60)
    train_loader, val_loader, tokenizer = get_qa_dataloaders(
        dataset_name='narrativeqa',
        max_length=1024,
        batch_size=2,
        max_train_samples=100,
        max_val_samples=50
    )
    
    # Check a batch
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Sample question: {batch['question'][0][:100]}...")
    print(f"Sample answer: {batch['answer'][0][:100]}...")
    
    print("\n" + "="*60)
    print("Testing Natural Questions (closed-book)")
    print("="*60)
    train_loader, val_loader, tokenizer = get_qa_dataloaders(
        dataset_name='natural_questions',
        max_length=512,
        batch_size=4,
        max_train_samples=100,
        max_val_samples=50,
        include_context=False  # Closed-book
    )
    
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Sample question: {batch['question'][0][:100]}...")
    print(f"Sample answer: {batch['answer'][0][:50]}...")
    
    print("\n" + "="*60)
    print("Testing SQuAD 2.0 (faster alternative)")
    print("="*60)
    train_loader, val_loader, tokenizer = get_qa_dataloaders(
        dataset_name='squad',
        max_length=1024,
        batch_size=4,
        max_train_samples=100,
        max_val_samples=50
    )
    
    batch = next(iter(train_loader))
    print(f"\nBatch keys: {batch.keys()}")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Sample question: {batch['question'][0][:100]}...")
    
    print("\n✓ All datasets loaded successfully!")