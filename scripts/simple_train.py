"""
Simple Training Script - No Optimizations
Use this to debug CUDA errors

Disables:
- torch.compile
- Mixed precision (AMP)
- Gradient accumulation
- Router curriculum
- Complex monitoring
"""

import yaml
import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.routed_model import RoutedHybridModel
from src.data.qa_datasets import get_qa_dataloaders


def simple_train():
    """Simple training with no optimizations"""
    
    print("\n" + "="*60)
    print("SIMPLE TRAINING (DEBUGGING MODE)")
    print("="*60)
    
    # Hardcoded simple config
    config = {
        'dataset_name': 'squad',
        'tokenizer_name': 'gpt2',
        'max_length': 128,
        'batch_size': 2,
        'num_workers': 0,
        'max_train_samples': 50,
        'max_val_samples': 10,
        
        'd_model': 256,
        'n_layers': 2,
        'n_heads': 4,
        'vocab_size': 50257,
        'learning_rate': 0.001,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, tokenizer = get_qa_dataloaders(
        dataset_name=config['dataset_name'],
        tokenizer_name=config['tokenizer_name'],
        max_length=config['max_length'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        max_train_samples=config['max_train_samples'],
        max_val_samples=config['max_val_samples']
    )
    print(f"✓ Loaded {len(train_loader)} train batches, {len(val_loader)} val batches")
    
    # Create model
    print("\nCreating model...")
    model = RoutedHybridModel(
        vocab_size=len(tokenizer),
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        use_query_aware_routing=False,  # Disable for debugging
        use_checkpoint=False,
        use_gradient_balancing=False,  # Disable
        use_position_invariance=False,  # Disable
    ).to(device)
    
    print(f"✓ Model created: {model.get_num_params() / 1e6:.1f}M params")
    
    # Simple optimizer (no fused, no fancy stuff)
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    print("✓ Optimizer created")
    
    # Training loop
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    model.train()
    
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
        try:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Check inputs are valid
            if input_ids.max() >= len(tokenizer):
                print(f"\n✗ Invalid token ID: {input_ids.max()} >= {len(tokenizer)}")
                print(f"  Token range: [{input_ids.min()}, {input_ids.max()}]")
                break
            
            if input_ids.shape[1] > model.pos_embedding.shape[1]:
                print(f"\n✗ Sequence too long: {input_ids.shape[1]} > {model.pos_embedding.shape[1]}")
                break
            
            # Forward pass (NO autocast, NO router loss)
            optimizer.zero_grad()
            
            loss, logits, _ = model(input_ids, labels=labels, deterministic=True)
            
            # Check for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n✗ NaN/Inf loss detected!")
                break
            
            # Backward
            loss.backward()
            
            # Clip grads
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Step
            optimizer.step()
            
            # Log
            if batch_idx % 5 == 0:
                print(f"\nBatch {batch_idx}: loss={loss.item():.4f}")
            
            # Stop after a few batches for testing
            if batch_idx >= 10:
                print(f"\n✓ Completed 10 batches successfully!")
                print("="*60)
                print("NO CUDA ERRORS - The model works!")
                print("="*60)
                print("\nThe error must be in:")
                print("1. Mixed precision (AMP)")
                print("2. Router loss computation")
                print("3. Curriculum learning")
                print("4. Gradient accumulation")
                print("5. torch.compile")
                return
                
        except RuntimeError as e:
            print(f"\n" + "="*60)
            print("✗ ERROR FOUND")
            print("="*60)
            print(f"Batch {batch_idx} failed")
            print(f"Error: {e}")
            print(f"Input shape: {input_ids.shape}")
            print(f"Label shape: {labels.shape}")
            import traceback
            traceback.print_exc()
            print("="*60)
            return
    
    print("\n✓ Training completed successfully!")


if __name__ == '__main__':
    simple_train()