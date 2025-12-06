#!/usr/bin/env python3
"""
Quick test of QA dataset loading

This script tests that all QA datasets can be loaded properly.
Run this before starting full training to catch any issues early.

Usage:
    python tests/test_qa_datasets.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.qa_datasets import get_qa_dataloaders


def test_dataset(dataset_name, **kwargs):
    """Test loading a specific dataset"""
    print(f"\n{'='*60}")
    print(f"Testing {dataset_name.upper()}")
    print(f"{'='*60}")
    
    try:
        train_loader, val_loader, tokenizer = get_qa_dataloaders(
            dataset_name=dataset_name,
            max_length=kwargs.get('max_length', 512),
            batch_size=2,
            num_workers=0,  # Use 0 for testing
            max_train_samples=10,
            max_val_samples=5,
            **kwargs
        )
        
        # Get a batch
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))
        
        print(f"\n✓ Dataset loaded successfully!")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        print(f"  Batch keys: {list(train_batch.keys())}")
        print(f"  Input shape: {train_batch['input_ids'].shape}")
        
        # Show example
        if 'question' in train_batch:
            print(f"\n  Example question: {train_batch['question'][0][:80]}...")
        if 'answer' in train_batch:
            print(f"  Example answer: {train_batch['answer'][0][:80]}...")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Failed to load {dataset_name}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("QA DATASET LOADING TEST")
    print("="*60)
    print("\nThis will test loading small samples from each dataset.")
    print("Full datasets will be loaded during actual training.\n")
    
    results = {}
    
    # Test SQuAD (should be fastest)
    print("\n" + "="*60)
    print("1/3: Testing SQuAD 2.0 (fastest)")
    print("="*60)
    results['squad'] = test_dataset('squad', max_length=512)
    
    # Test Natural Questions
    print("\n" + "="*60)
    print("2/3: Testing Natural Questions (closed-book)")
    print("="*60)
    results['natural_questions'] = test_dataset(
        'natural_questions',
        max_length=512,
        include_context=False
    )
    
    # Test NarrativeQA
    print("\n" + "="*60)
    print("3/3: Testing NarrativeQA (long-context)")
    print("="*60)
    results['narrativeqa'] = test_dataset(
        'narrativeqa',
        max_length=1024,
        use_summary=True
    )
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for dataset, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {dataset}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL DATASETS LOADED SUCCESSFULLY!")
        print("="*60)
        print("\nYou're ready to start training!")
        print("\nNext steps:")
        print("  1. Quick test: python scripts/train_qa.py --config configs/squad.yaml --max-train-samples 100")
        print("  2. Full training: python scripts/train_qa.py --config configs/narrativeqa.yaml")
        print("="*60 + "\n")
        return 0
    else:
        print("\n" + "="*60)
        print("✗ SOME DATASETS FAILED")
        print("="*60)
        print("\nCheck the errors above and install missing dependencies:")
        print("  pip install datasets transformers --break-system-packages")
        print("="*60 + "\n")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)