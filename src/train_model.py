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
    """Logger for tracking losses per epoch"""
    
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = self.log_dir / 'training_log.jsonl'
        print(f"Loss logger initialized: {self.log_file}")
    
    def log_epoch(self, epoch, train_loss, val_loss, val_perplexity, epoch_time, lr):
        """Log all metrics for an epoch"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            'timestamp': timestamp,
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_perplexity': float(val_perplexity),
            'epoch_time_seconds': float(epoch_time),
            'learning_rate': float(lr),
        }
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        # Print to console
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch} SUMMARY")
        print(f"{'='*60}")
        print(f"Train Loss:       {train_loss:.4f}")
        print(f"Val Loss:         {val_loss:.4f}")
        print(f"Val Perplexity:   {val_perplexity:.2f}")
        print(f"Epoch Time:       {epoch_time:.1f}s")
        print(f"Learning Rate:    {lr:.2e}")
        print(f"{'='*60}\n")


class Trainer:
    def __init__(self, config, model=None):
        """
        Initialize trainer
        
        Args:
            config: Training configuration
            model: Pre-instantiated model (optional). If None, uses default from config.
        """
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

        # Use provided model or create default
        if model is not None:
            self.model = model.to(self.device)
            print("Using provided model instance")
        else:
            # Fallback: create default model for backward compatibility
            print("No model provided, creating default HybridSSMTransformer")
            from src.model.hybrid_model import HybridSSMTransformer
            self.model = HybridSSMTransformer(
                vocab_size=config.vocab_size,
                d_model=config.d_model,
                n_layers=config.n_layers,
                pattern=config.layer_pattern,
                n_heads=config.n_heads,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                dropout=config.dropout,
            ).to(self.device)

        
        # Print model info
        print(f"\n{'='*60}")
        print("MODEL CONFIGURATION")
        print(f"{'='*60}")


        print(f"Model type: {type(self.model).__name__}")
        
        # Try to get model info if methods exist
        if hasattr(self.model, 'get_num_params'):
            print(f"Total parameters: {self.model.get_num_params() / 1e6:.2f}M")
            if hasattr(self.model, 'get_num_params'):
                try:
                    non_emb_params = self.model.get_num_params(non_embedding=True)
                    print(f"Non-embedding parameters: {non_emb_params / 1e6:.2f}M")
                except:
                    pass
        else:
            # Count parameters manually
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Total parameters: {total_params / 1e6:.2f}M")
        
        if hasattr(config, 'layer_pattern'):
            print(f"Pattern: {config.layer_pattern}")
        
        print(f"Training dtype: {self.dtype}")
        
        if hasattr(self.model, 'get_layer_info'):
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
        # Try to use model's configure_optimizers if available
        if hasattr(self.model, 'configure_optimizers'):
            self.optimizer = self.model.configure_optimizers(
                weight_decay=config.weight_decay,
                learning_rate=config.learning_rate,
                betas=(0.9, 0.95),
                device_type='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            # Default optimizer
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.95)
            )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * len(self.train_loader),
            eta_min=config.learning_rate * 0.1
        )
        
        # Loss logger
        self.loss_logger = LossLogger(log_dir=config.output_dir)
        
        # For checkpointing
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        

        # Global step counter
        self.global_step = 0
        self.training_start_time = time.time()
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        total_steps = len(self.train_loader)
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar, 1):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass with autocast
            with autocast('cuda', enabled=self.use_amp, dtype=self.dtype):
                loss, logits = self.model(input_ids, labels=labels)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'step': f"{batch_idx}/{total_steps}",
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
        
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
    
    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint for this epoch"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'val_loss': val_loss,
            'config': self.config,
            'training_time': time.time() - self.training_start_time,
            'dtype': str(self.dtype),
            'model_type': type(self.model).__name__,  # Save model type for reference
        }
        
        # Save checkpoint with epoch number
        path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path.name}")
    
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
        print(f"{'='*60}\n")
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss, epoch_time = self.train_epoch(epoch)
            
            # Validate
            val_loss, perplexity = self.validate()
            
            # Get current learning rate
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log all metrics for this epoch
            self.loss_logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                val_perplexity=perplexity,
                epoch_time=epoch_time,
                lr=current_lr
            )
            
            # WandB logging (per epoch)
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'val/perplexity': perplexity,
                    'train/lr': current_lr,
                    'train/epoch_time': epoch_time,
                })
            
            # Save checkpoint every epoch
            self.save_checkpoint(epoch, val_loss)
        
        # Training summary
        total_training_time = time.time() - self.training_start_time
        hours = int(total_training_time / 3600)
        minutes = int((total_training_time % 3600) / 60)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"{'='*60}")
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
        self.learning_rate = float(0.0003)
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