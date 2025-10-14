import torch
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_progress(checkpoint_dir):
    """Plot training curves from checkpoints"""
    checkpoint = torch.load(Path(checkpoint_dir) / 'checkpoint_latest.pt')
    
    # Extract metrics (you'll need to save these in checkpoints)
    # For now, just check what's in the checkpoint
    print("Checkpoint contents:")
    for key in checkpoint.keys():
        print(f"  {key}")
    
    print(f"\nBest val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")

if __name__ == '__main__':
    import sys
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else 'outputs/baseline-v1'
    plot_training_progress(checkpoint_dir)