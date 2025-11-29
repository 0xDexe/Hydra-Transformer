import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
import wandb
from tqdm import tqdm
import os
from pathlib import Path
import time
import json
from datetime import datetime

from src.model.routed_model import RoutedHybridSSMTransformer
from src.data.dataset import get_dataloaders


class LossLogger:
    """Logger for tracking and saving losses"""
    
    def __init__(self, log_dir, log_interval=10):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / 'training_log.jsonl'
        self.loss_history = []
        self.log_interval = log_interval
        
        # Track significant changes
        self.last_logged_loss = None
        self.significant_change_threshold = 0.1  # 10% change
        
        print(f"Loss logger initialized: {self.log_file}")
    
    def log(self, step, epoch, loss, lr, split='train', force=False):
        """
        Log loss value
        
        Args:
            step: Current training step
            epoch: Current epoch
            loss: Loss value
            lr: Learning rate
            split: 'train' or 'val'
            force: Force logging regardless of interval
        """
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'step': step,
            'epoch': epoch,
            'split': split,
            'loss': float(loss),
            'lr': float(lr),
        }
        
        # Check if we should log based on interval or significant change
        should_log = force
        
        if not should_log and step % self.log_interval == 0:
            should_log = True
        
        # Check for significant change
        if not should_log and self.last_logged_loss is not None:
            change_ratio = abs(loss - self.last_logged_loss) / (self.last_logged_loss + 1e-8)
            if change_ratio >= self.significant_change_threshold:
                should_log = True
                log_entry['significant_change'] = True
                log_entry['change_ratio'] = float(change_ratio)
        
        if should_log:
            # Append to history
            self.loss_history.append(log_entry)
            
            # Write to file
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            self.last_logged_loss = loss
    
    def log_validation(self, epoch, val_loss, val_perplexity):
        """Log validation metrics"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'epoch': epoch,
            'split': 'validation',
            'val_loss': float(val_loss),
            'val_perplexity': float(val_perplexity),
        }
        
        self.loss_history.append(log_entry)
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def log_checkpoint(self, epoch, checkpoint_type, checkpoint_path):
        """Log checkpoint save"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'epoch': epoch,
            'event': 'checkpoint_saved',
            'checkpoint_type': checkpoint_type,
            'checkpoint_path': str(checkpoint_path),
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_recent_losses(self, n=100):
        """Get recent loss values"""
        return [entry['loss'] for entry in self.loss_history[-n:] if 'loss' in entry]


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Determine dtype for training
        if config.use_mixed_precision and torch.cuda.is_available():
            # Use bf16 if available, otherwise fp16
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
                print("Using mixed precision training with bfloat16")
            else:
                self.dtype = torch.float16
                print("Using mixed precision training with float16")
        else:
            self.dtype = torch.float32
            print("Using float32 training")
        
        # Setup gradient scaler for mixed precision
        self.use_amp = config.use_mixed_precision and torch.cuda.is_available()
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        
        # Setup wandb
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                config=config.__dict__,
                name=config.run_name
            )
        
        # Create model
        self.model = RoutedHybridSSMTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dropout=config.dropout,
            max_seq_len=config.max_length,
            routing_topk_ratio=config.routing_topk_ratio,
            router_hidden_dim=config.router_hidden_dim,
        ).to(self.device)
        
        print(f"\n{'='*60}")
        print("MODEL CONFIGURATION")
        print(f"{'='*60}")
        print(f"Total parameters: {self.model.get_num_params() / 1e6:.2f}M")
        print(f"Non-embedding parameters: {self.model.get_num_params(non_embedding=True) / 1e6:.2f}M")
        print(f"Routing: topk_ratio={config.routing_topk_ratio}, "
            f"context_mode={self.model.context_layer.mode}")
        print(f"Training dtype: {self.dtype}")
        print("\nLayer structure:")
        for layer_info in self.model.get_layer_info():
            print(f"  {layer_info}")
        print(f"{'='*60}\n")
        
        # Data loaders
        self.train_loader, self.val_loader, self.tokenizer = get_dataloaders(
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            tokenizer_name=config.tokenizer_name,
            max_length=config.max_length,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        # Update vocab size in config if needed
        if len(self.tokenizer) != config.vocab_size:
            print(f"Updating vocab_size: {config.vocab_size} -> {len(self.tokenizer)}")
            config.vocab_size = len(self.tokenizer)
        
        # Optimizer with proper weight decay configuration
        self.optimizer = self.model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=(0.9, 0.95),
            device_type='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * len(self.train_loader),
            eta_min=config.learning_rate * 0.1
        )
        
        # Loss logger
        self.loss_logger = LossLogger(
            log_dir=config.output_dir,
            log_interval=config.log_interval
        )
        
        # For checkpointing
        self.best_val_loss = float('inf')
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Time-based checkpointing (every 4 hours)
        self.checkpoint_interval_seconds = 4 * 60 * 60  # 4 hours
        self.last_checkpoint_time = time.time()
        self.training_start_time = time.time()
        
        # Global step counter
        self.global_step = 0
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with autocast for mixed precision
            with autocast('cuda', enabled=self.use_amp, dtype=self.dtype):
                loss, logits = self.model(input_ids, labels=labels)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping (unscale first for accurate clipping)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log loss
            self.loss_logger.log(
                step=self.global_step,
                epoch=epoch,
                loss=loss.item(),
                lr=current_lr,
                split='train'
            )
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # WandB logging
            if self.config.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': current_lr,
                    'train/step': self.global_step,
                    'train/epoch': epoch,
                })
            
            # Time-based checkpoint (every 2 hours)
            current_time = time.time()
            time_since_last_checkpoint = current_time - self.last_checkpoint_time
            
            if time_since_last_checkpoint >= self.checkpoint_interval_seconds:
                print(f"\n 2-hour checkpoint triggered")
                val_loss, perplexity = self.validate()
                self.save_checkpoint(
                    epoch=epoch,
                    val_loss=val_loss,
                    is_best=False,
                    checkpoint_type='time_based'
                )
                self.last_checkpoint_time = current_time
                self.model.train()  # Return to training mode
        
        avg_loss = total_loss / len(self.train_loader)
        epoch_time = time.time() - epoch_start_time
        
        return avg_loss, epoch_time
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Use autocast for validation too
            with autocast('cuda', enabled=self.use_amp, dtype=self.dtype):
                loss, logits = self.model(input_ids, labels=labels)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return avg_loss, perplexity.item()
    
    def save_checkpoint(self, epoch, val_loss, is_best=False, checkpoint_type='regular'):
        """
        Save model checkpoint
        
        Args:
            epoch: Current epoch
            val_loss: Validation loss
            is_best: Whether this is the best model so far
            checkpoint_type: Type of checkpoint ('regular', 'best', 'time_based')
        """
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'training_time': time.time() - self.training_start_time,
            'dtype': str(self.dtype),
        }
        
        # Save latest
        path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
            self.loss_logger.log_checkpoint(epoch, 'best', path)
        
        # Time-based checkpoint
        if checkpoint_type == 'time_based':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            hours = int((time.time() - self.training_start_time) / 3600)
            path = self.output_dir / f'checkpoint_time_{hours}h_{timestamp}.pt'
            torch.save(checkpoint, path)
            print(f" Saved time-based checkpoint ({hours}h): {path.name}")
            self.loss_logger.log_checkpoint(epoch, f'time_based_{hours}h', path)
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Training dtype: {self.dtype}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Output directory: {self.output_dir}")
        print(f"Logging to: {self.loss_logger.log_file}")
        print(f"Checkpoint interval: {self.checkpoint_interval_seconds / 3600:.1f} hours")
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, epoch_time = self.train_epoch(epoch)
            print(f"Train loss: {train_loss:.4f} (time: {epoch_time:.1f}s)")
            
            # Validate
            val_loss, perplexity = self.validate()
            print(f"Val loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
            
            # Log validation metrics
            self.loss_logger.log_validation(epoch, val_loss, perplexity)
            
            # WandB logging
            if self.config.use_wandb:
                wandb.log({
                    'val/loss': val_loss,
                    'val/perplexity': perplexity,
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                })
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        total_training_time = time.time() - self.training_start_time
        hours = int(total_training_time / 3600)
        minutes = int((total_training_time % 3600) / 60)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"{'='*60}")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total training time: {hours}h {minutes}m")
        print(f"Loss log saved to: {self.loss_logger.log_file}")
        print(f"{'='*60}\n")


# Config class with all parameters
class TrainConfig:
    def __init__(self):
        # Model architecture
        self.d_model = 512
        self.n_layers = 6
        self.n_heads = 8
        self.d_state = 16
        self.d_conv = 4
        self.expand = 2
        self.dropout = 0.1
        self.vocab_size = 50257  # Will be updated from tokenizer
        self.use_routing = True             
        self.routing_topk_ratio = 0.20       # 20% tokens to attention
        self.router_hidden_dim = self.d_model
        
        # Data
        self.dataset_name = 'wikitext'
        self.dataset_config = 'wikitext-2-raw-v1'
        self.tokenizer_name = 'gpt2'
        self.max_length = 512
        self.batch_size = 8
        self.num_workers = 4
        
        # Training
        self.num_epochs = 10
        self.learning_rate = 3e-4
        self.weight_decay = 0.01
        self.grad_clip = 1.0
        self.use_mixed_precision = True  # Enable mixed precision training
        
        # Logging
        self.use_wandb = True
        self.project_name = 'routed-hybrid-ssm-transformer'
        self.run_name = 'token_routing-v1'
        self.output_dir = 'outputs/token_routing-v1'
        self.log_interval = 10  # Log every N steps


if __name__ == '__main__':
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()