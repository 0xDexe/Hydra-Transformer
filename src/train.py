import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import os
from pathlib import Path

from model.hybrid_model import HybridSSMTransformer
from data.dataset import get_dataloaders

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup wandb
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                config=config,
                name=config.run_name
            )
        
        # Create model
        self.model = HybridSSMTransformer(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            pattern=config.layer_pattern,
        ).to(self.device)
        
        print(f"Model parameters: {self.model.get_num_params() / 1e6:.2f}M")
        
        # Data loaders
        self.train_loader, self.val_loader, self.tokenizer = get_dataloaders(
            dataset_name=config.dataset_name,
            dataset_config=config.dataset_config,
            tokenizer_name=config.tokenizer_name,
            max_length=config.max_length,
            batch_size=config.batch_size,
            num_workers=config.num_workers
        )
        
        # Update vocab size in config
        config.vocab_size = len(self.tokenizer)
        
        # Optimizer
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
        
        # For checkpointing
        self.best_val_loss = float('inf')
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            loss, logits = self.model(input_ids, labels=labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Logging
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if self.config.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/step': epoch * len(self.train_loader) + batch_idx
                })
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            loss, logits = self.model(input_ids, labels=labels)
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return avg_loss, perplexity.item()
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Save latest
        path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*50}")
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Train loss: {train_loss:.4f}")
            
            # Validate
            val_loss, perplexity = self.validate()
            print(f"Val loss: {val_loss:.4f}, Perplexity: {perplexity:.2f}")
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'val/loss': val_loss,
                    'val/perplexity': perplexity,
                    'epoch': epoch
                })
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")


# Simple config class
class TrainConfig:
    def __init__(self):
        # Model
        self.d_model = 512
        self.n_layers = 6
        self.layer_pattern = 'alternating'
        self.vocab_size = 50257  # Will be updated from tokenizer
        
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
        
        # Logging
        self.use_wandb = True
        self.project_name = 'hybrid-ssm-transformer'
        self.run_name = 'baseline-hybrid'
        self.output_dir = 'outputs/baseline'


if __name__ == '__main__':
    config = TrainConfig()
    trainer = Trainer(config)
    trainer.train()