import torch
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse
from datetime import datetime

def load_training_log(log_file):
    """Load training log from JSONL file"""
    logs = []
    with open(log_file, 'r') as f:
        for line in f:
            logs.append(json.loads(line))
    return logs

def analyze_losses(logs):
    """Analyze loss patterns from logs"""
    train_losses = []
    val_losses = []
    timestamps = []
    significant_changes = []
    
    for entry in logs:
        if 'loss' in entry and entry['split'] == 'train':
            train_losses.append({
                'step': entry['step'],
                'loss': entry['loss'],
                'epoch': entry['epoch'],
                'timestamp': entry['timestamp']
            })
            if entry.get('significant_change', False):
                significant_changes.append(entry)
        
        elif entry['split'] == 'validation':
            val_losses.append({
                'epoch': entry['epoch'],
                'loss': entry['val_loss'],
                'perplexity': entry['val_perplexity'],
                'timestamp': entry['timestamp']
            })
    
    return train_losses, val_losses, significant_changes

def plot_training_progress(checkpoint_dir):
    """Plot training curves from logs and checkpoints"""
    checkpoint_dir = Path(checkpoint_dir)
    log_file = checkpoint_dir / 'training_log.jsonl'
    
    if not log_file.exists():
        print(f"Error: Log file not found at {log_file}")
        return
    
    print(f"\n{'='*60}")
    print("TRAINING PROGRESS ANALYSIS")
    print(f"{'='*60}\n")
    
    # Load logs
    logs = load_training_log(log_file)
    train_losses, val_losses, significant_changes = analyze_losses(logs)
    
    print(f"Total training steps: {len(train_losses)}")
    print(f"Validation points: {len(val_losses)}")
    print(f"Significant loss changes detected: {len(significant_changes)}")
    
    # Print significant changes
    if significant_changes:
        print(f"\nSignificant Loss Changes:")
        for change in significant_changes[:10]:  # Show first 10
            print(f"  Step {change['step']}: Loss {change['loss']:.4f} "
                  f"(change: {change['change_ratio']*100:.1f}%)")
    
    # Check for checkpoint
    checkpoint_path = checkpoint_dir / 'checkpoint_latest.pt'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"\nLatest Checkpoint:")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Global Step: {checkpoint.get('global_step', 'N/A')}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")
        print(f"  Training Time: {checkpoint.get('training_time', 0) / 3600:.2f} hours")
    
    # Plotting
    if train_losses or val_losses:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Training loss over steps
        if train_losses:
            steps = [x['step'] for x in train_losses]
            losses = [x['loss'] for x in train_losses]
            axes[0, 0].plot(steps, losses, alpha=0.6, linewidth=0.5)
            axes[0, 0].set_xlabel('Training Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].set_title('Training Loss vs Steps')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Highlight significant changes
            sig_steps = [x['step'] for x in significant_changes]
            sig_losses = [x['loss'] for x in significant_changes]
            axes[0, 0].scatter(sig_steps, sig_losses, color='red', 
                              s=50, alpha=0.7, label='Significant Change')
            axes[0, 0].legend()
        
        # Plot 2: Validation loss over epochs
        if val_losses:
            epochs = [x['epoch'] for x in val_losses]
            losses = [x['loss'] for x in val_losses]
            axes[0, 1].plot(epochs, losses, 'o-', linewidth=2, markersize=8)
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Validation Loss')
            axes[0, 1].set_title('Validation Loss vs Epochs')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Validation perplexity
        if val_losses:
            epochs = [x['epoch'] for x in val_losses]
            perplexities = [x['perplexity'] for x in val_losses]
            axes[1, 0].plot(epochs, perplexities, 'o-', linewidth=2, markersize=8, color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Perplexity')
            axes[1, 0].set_title('Validation Perplexity vs Epochs')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Loss distribution
        if train_losses:
            losses = [x['loss'] for x in train_losses]
            axes[1, 1].hist(losses, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(sum(losses)/len(losses), color='red', 
                              linestyle='--', linewidth=2, label='Mean')
            axes[1, 1].set_xlabel('Loss Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Training Loss Distribution')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = checkpoint_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Training progress plot saved to: {plot_path}")
        
        # Show plot
        plt.show()
    
    print(f"\n{'='*60}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('checkpoint_dir', type=str, nargs='?', 
                       default='outputs/baseline-v1',
                       help='Directory containing checkpoints and logs')
    args = parser.parse_args()
    
    plot_training_progress(args.checkpoint_dir)
