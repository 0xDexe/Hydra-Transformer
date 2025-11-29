"""
Resume Training Script

This script resumes training from the last saved checkpoint.
It automatically loads the model configuration, weights, optimizer state, and scheduler state.

Usage:
    python resume_training.py --checkpoint outputs/baseline/checkpoint_epoch_5.pt --num_epochs 10
    
    or auto-detect latest:
    python resume_training.py --checkpoint_dir outputs/baseline --num_epochs 10
"""

import torch
import argparse
from pathlib import Path
import glob

from src.train import Trainer, TrainConfig


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint in a directory
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to the latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for checkpoint files
    checkpoint_files = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    
    if not checkpoint_files:
        return None
    
    # Extract epoch numbers and find the latest
    latest_checkpoint = None
    latest_epoch = -1
    
    for ckpt in checkpoint_files:
        try:
            # Extract epoch number from filename: checkpoint_epoch_5.pt -> 5
            epoch_str = ckpt.stem.split('_')[-1]
            epoch = int(epoch_str)
            
            if epoch > latest_epoch:
                latest_epoch = epoch
                latest_checkpoint = ckpt
        except (ValueError, IndexError):
            continue
    
    return latest_checkpoint


def load_checkpoint(checkpoint_path):
    """
    Load a checkpoint file
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        checkpoint dictionary
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"\nCheckpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Global step: {checkpoint['global_step']}")
    print(f"  Val loss: {checkpoint['val_loss']:.4f}")
    print(f"  Training time: {checkpoint['training_time'] / 3600:.2f} hours")
    print(f"  Dtype: {checkpoint.get('dtype', 'N/A')}")
    
    return checkpoint


def resume_training(checkpoint_path, num_additional_epochs=None, new_output_dir=None):
    """
    Resume training from a checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        num_additional_epochs: Number of additional epochs to train (if None, uses config value)
        new_output_dir: New output directory (if None, uses config value)
        
    Returns:
        Trained trainer object
    """
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Get config from checkpoint
    config = checkpoint['config']
    
    # Update config if needed
    if num_additional_epochs is not None:
        original_epochs = config.num_epochs
        config.num_epochs = checkpoint['epoch'] + 1 + num_additional_epochs
        print(f"\nUpdating num_epochs: {original_epochs} -> {config.num_epochs}")
        print(f"  (Resuming from epoch {checkpoint['epoch']}, training for {num_additional_epochs} more)")
    
    if new_output_dir is not None:
        print(f"Updating output_dir: {config.output_dir} -> {new_output_dir}")
        config.output_dir = new_output_dir
    
    # Create trainer
    print(f"\n{'='*60}")
    print("INITIALIZING TRAINER")
    print(f"{'='*60}\n")
    trainer = Trainer(config)
    
    # Load model state
    print("Loading model state...")
    trainer.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state
    print("Loading optimizer state...")
    trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state
    print("Loading scheduler state...")
    trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load scaler state if it exists
    if checkpoint.get('scaler_state_dict') is not None and trainer.use_amp:
        print("Loading gradient scaler state...")
        trainer.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # Restore global step and training start time
    trainer.global_step = checkpoint['global_step']
    
    # Adjust training start time to account for previous training
    # This keeps the total training time accurate
    import time
    trainer.training_start_time = time.time() - checkpoint['training_time']
    
    print(f"\nResuming from epoch {checkpoint['epoch'] + 1}")
    print(f"Global step: {trainer.global_step}")
    print(f"{'='*60}\n")
    
    # Resume training
    resume_from_epoch = checkpoint['epoch'] + 1
    
    print(f"\n{'='*60}")
    print("RESUMING TRAINING")
    print(f"{'='*60}")
    print(f"Device: {trainer.device}")
    print(f"Training dtype: {trainer.dtype}")
    print(f"Mixed precision: {trainer.use_amp}")
    print(f"Output directory: {trainer.output_dir}")
    print(f"Logging to: {trainer.loss_logger.log_file}")
    print(f"Starting from epoch: {resume_from_epoch}")
    print(f"Total epochs: {config.num_epochs}")
    print(f"{'='*60}\n")
    
    for epoch in range(resume_from_epoch, config.num_epochs):
        # Train
        train_loss, epoch_time = trainer.train_epoch(epoch)
        
        # Validate
        val_loss, perplexity = trainer.validate()
        
        # Get current learning rate
        current_lr = trainer.scheduler.get_last_lr()[0]
        
        # Log all metrics for this epoch
        trainer.loss_logger.log_epoch(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_perplexity=perplexity,
            epoch_time=epoch_time,
            lr=current_lr
        )
        
        # WandB logging (per epoch)
        if config.use_wandb:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'val/loss': val_loss,
                'val/perplexity': perplexity,
                'train/lr': current_lr,
                'train/epoch_time': epoch_time,
            })
        
        # Save checkpoint every epoch
        trainer.save_checkpoint(epoch, val_loss)
    
    # Training summary
    import time
    total_training_time = time.time() - trainer.training_start_time
    hours = int(total_training_time / 3600)
    minutes = int((total_training_time % 3600) / 60)
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total training time: {hours}h {minutes}m")
    print(f"Loss log saved to: {trainer.loss_logger.log_file}")
    print(f"{'='*60}\n")
    
    return trainer


def main():
    parser = argparse.ArgumentParser(description='Resume training from checkpoint')
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to checkpoint file (e.g., outputs/baseline/checkpoint_epoch_5.pt)'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        help='Directory containing checkpoints (will auto-detect latest)'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of additional epochs to train (optional, uses config default if not specified)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='New output directory (optional, uses checkpoint config if not specified)'
    )
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    elif args.checkpoint_dir:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path is None:
            print(f"Error: No checkpoints found in {args.checkpoint_dir}")
            return
        print(f"Auto-detected latest checkpoint: {checkpoint_path}")
    else:
        print("Error: Must specify either --checkpoint or --checkpoint_dir")
        parser.print_help()
        return
    
    # Resume training
    resume_training(
        checkpoint_path=checkpoint_path,
        num_additional_epochs=args.num_epochs,
        new_output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()