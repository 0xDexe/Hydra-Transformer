"""
Optimized Training Script for QA Datasets (NarrativeQA, Natural Questions, SQuAD)

Includes all performance optimizations:
- Mixed precision training (AMP)
- Gradient accumulation
- torch.compile support
- Fused optimizer
- Efficient data loading
- Reduced monitoring overhead

Usage:
    python scripts/train_qa.py --config configs/narrativeqa.yaml
    python scripts/train_qa.py --config configs/natural_questions.yaml
    python scripts/train_qa.py --config configs/squad.yaml
    
Quick test:
    python scripts/train_qa.py --config configs/squad.yaml --max-train-samples 100
"""

import yaml
import argparse
import sys
import time
import json
from pathlib import Path
from typing import Dict
from datetime import datetime
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.routed_hybrid_model import RoutedHybridModel
from src.model.token_router import RouterLoss, RouterMonitor, RouterCurriculum
from src.data.qa_datasets import get_qa_dataloaders


def load_config_from_yaml(yaml_path):
    """Load configuration from YAML file"""
    with open(yaml_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # Create config dict with defaults
    config = {
        # Model defaults
        'd_model': 768,
        'n_layers': 12,
        'n_heads': 12,
        'd_state': 16,
        'd_conv': 4,
        'expand': 2,
        'd_ff': None,
        'dropout': 0.1,
        'tie_weights': True,
        'vocab_size': 50257,  # Will be updated from tokenizer
        
        # Router defaults
        'router_hidden_dim': 64,
        'target_ratio': 0.15,
        'use_gradient_balancing': True,
        'use_position_invariance': True,
        'use_checkpoint': False,
        'use_query_aware_routing': False,  # NEW: Query-aware routing
        
        # Curriculum defaults
        'router_warmup_steps': 2000,
        'curriculum_heuristic': 'uniform',
        
        # Router loss defaults
        'router_loss_weight': 0.01,
        'load_loss_weight': 0.01,
        'entropy_loss_weight': 0.01,
        'variance_loss_weight': 0.01,
        'position_inv_weight': 0.005,
        'diversity_loss_weight': 0.005,
        
        # Data defaults
        'dataset_name': 'squad',
        'tokenizer_name': 'gpt2',
        'max_length': 1024,
        'batch_size': 8,
        'num_workers': 4,
        'dataset_kwargs': {},
        'max_train_samples': None,
        'max_val_samples': None,
        
        # Training defaults
        'num_epochs': 10,
        'learning_rate': 0.007,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'use_amp': True,
        'gradient_accumulation_steps': 2,
        'use_fused_optimizer': True,
        'compile_model': True,
        
        # Logging defaults
        'use_wandb': False,
        'project_name': 'hydra-qa',
        'run_name': 'qa-experiment',
        'output_dir': 'outputs/qa',
        'log_interval': 50,
        'monitor_interval': 200,
        'save_every': 5,
    }
    
    # Update from YAML
    if 'model' in cfg_dict:
        config.update(cfg_dict['model'])
    
    if 'router' in cfg_dict:
        config.update(cfg_dict['router'])
    
    if 'curriculum' in cfg_dict:
        if 'warmup_steps' in cfg_dict['curriculum']:
            config['router_warmup_steps'] = cfg_dict['curriculum']['warmup_steps']
        if 'heuristic_type' in cfg_dict['curriculum']:
            config['curriculum_heuristic'] = cfg_dict['curriculum']['heuristic_type']
    
    if 'router_loss' in cfg_dict:
        loss_cfg = cfg_dict['router_loss']
        config['router_loss_weight'] = loss_cfg.get('total_weight', config['router_loss_weight'])
        config['load_loss_weight'] = loss_cfg.get('load_balance', config['load_loss_weight'])
        config['entropy_loss_weight'] = loss_cfg.get('entropy', config['entropy_loss_weight'])
        config['variance_loss_weight'] = loss_cfg.get('variance', config['variance_loss_weight'])
        config['position_inv_weight'] = loss_cfg.get('position_invariance', config['position_inv_weight'])
        config['diversity_loss_weight'] = loss_cfg.get('layer_diversity', config['diversity_loss_weight'])
    
    if 'data' in cfg_dict:
        data_cfg = cfg_dict['data']
        config['dataset_name'] = data_cfg.get('dataset_name', config['dataset_name'])
        config['tokenizer_name'] = data_cfg.get('tokenizer_name', config['tokenizer_name'])
        config['max_length'] = data_cfg.get('max_length', config['max_length'])
        config['batch_size'] = data_cfg.get('batch_size', config['batch_size'])
        config['num_workers'] = data_cfg.get('num_workers', config['num_workers'])
        config['max_train_samples'] = data_cfg.get('max_train_samples', config['max_train_samples'])
        config['max_val_samples'] = data_cfg.get('max_val_samples', config['max_val_samples'])
        
        # Dataset-specific kwargs
        if 'use_summary' in data_cfg:
            config['dataset_kwargs']['use_summary'] = data_cfg['use_summary']
        if 'include_context' in data_cfg:
            config['dataset_kwargs']['include_context'] = data_cfg['include_context']
    
    if 'training' in cfg_dict:
        config.update(cfg_dict['training'])
    
    if 'logging' in cfg_dict:
        config.update(cfg_dict['logging'])
    
    return config


class OptimizedQATrainer:
    """
    High-performance QA trainer with all optimizations
    
    Features:
    - Mixed precision (AMP)
    - Gradient accumulation
    - torch.compile
    - Fused optimizer
    - Efficient monitoring
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print("OPTIMIZED QA TRAINER")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Dataset: {config['dataset_name']}")
        
        # Load data first to get vocab size
        print(f"\n{'='*60}")
        print("Loading Data")
        print(f"{'='*60}")
        
        self.train_loader, self.val_loader, self.tokenizer = get_qa_dataloaders(
            dataset_name=config['dataset_name'],
            tokenizer_name=config['tokenizer_name'],
            max_length=config['max_length'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            max_train_samples=config['max_train_samples'],
            max_val_samples=config['max_val_samples'],
            **config['dataset_kwargs']
        )
        
        config['vocab_size'] = len(self.tokenizer)
        
        # Create model
        print(f"\n{'='*60}")
        print("Creating Model")
        print(f"{'='*60}")
        
        self.model = RoutedHybridModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_state=config['d_state'],
            d_conv=config['d_conv'],
            expand=config['expand'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            router_hidden_dim=config['router_hidden_dim'],
            target_ratio=config['target_ratio'],
            use_gradient_balancing=config['use_gradient_balancing'],
            use_position_invariance=config['use_position_invariance'],
            use_checkpoint=config['use_checkpoint'],
            tie_weights=config['tie_weights'],
            use_query_aware_routing=config['use_query_aware_routing']  # NEW
        ).to(self.device)
        
        total_params = self.model.get_num_params() / 1e6
        print(f"✓ Model: {total_params:.2f}M parameters")
        
        # torch.compile
        if config['compile_model'] and hasattr(torch, 'compile'):
            print("✓ Compiling model...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
        
        # Optimizer
        optimizer_kwargs = {
            'lr': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'betas': (0.9, 0.95)
        }
        
        if config['use_fused_optimizer'] and self.device.type == 'cuda':
            try:
                optimizer_kwargs['fused'] = True
                self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
                print("✓ Using fused optimizer")
            except:
                self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
                print("⚠ Fused optimizer unavailable")
        else:
            self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
        
        # Mixed precision
        self.use_amp = config['use_amp'] and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("✓ Mixed precision enabled")
        
        # Gradient accumulation
        self.grad_accum_steps = config['gradient_accumulation_steps']
        if self.grad_accum_steps > 1:
            print(f"✓ Gradient accumulation: {self.grad_accum_steps} steps")
            print(f"  Effective batch: {config['batch_size'] * self.grad_accum_steps}")
        
        # Scheduler
        self.total_steps = config['num_epochs'] * len(self.train_loader) // self.grad_accum_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps,
            eta_min=config['learning_rate'] * 0.1
        )
        
        # Router curriculum
        self.curriculum = RouterCurriculum(
            total_steps=self.total_steps,
            warmup_steps=config['router_warmup_steps'],
            heuristic_type=config['curriculum_heuristic']
        )
        
        # Router loss
        self.router_loss_fn = RouterLoss(
            target_ratio=config['target_ratio'],
            load_weight=config['load_loss_weight'],
            entropy_weight=config['entropy_loss_weight'],
            position_inv_weight=config['position_inv_weight'],
            diversity_weight=config['diversity_loss_weight'],
            variance_weight=config['variance_loss_weight']
        )
        
        # Monitor
        self.monitor = RouterMonitor(num_layers=config['n_layers'])
        
        # Checkpointing
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        
        # State
        self.global_step = 0
        self.current_epoch = 0
        
        # Training log (JSON per epoch)
        self.training_log = []
        self.training_log_path = self.output_dir / 'training_log.json'
        
        # Load existing log if resuming
        if self.training_log_path.exists():
            with open(self.training_log_path, 'r') as f:
                self.training_log = json.load(f)
            print(f"✓ Loaded existing training log with {len(self.training_log)} epochs")
        
        # Wandb
        if config['use_wandb']:
            import wandb
            wandb.init(
                project=config['project_name'],
                name=config['run_name'],
                config=config
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        print(f"{'='*60}\n")
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        print(f"\n{'='*60}")
        print("LOADING CHECKPOINT")
        print(f"{'='*60}")
        print(f"Path: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model state loaded")
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("✓ Optimizer state loaded")
        
        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("✓ Scheduler state loaded")
        
        # Load scaler state
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            print("✓ Scaler state loaded")
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"✓ Resuming from epoch {self.current_epoch}")
        print(f"✓ Global step: {self.global_step}")
        print(f"✓ Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
    
    def train_step(self, input_ids, labels, accumulation_step) -> Dict[str, float]:
        """Optimized training step"""
        
        # Forward with autocast
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            lm_loss, logits, router_outputs = self.model(
                input_ids, labels=labels, deterministic=False, return_router_outputs=True
            )
            
            # Router loss
            result = self.router_loss_fn(router_outputs, return_components=True)
            router_loss, router_components = result
            
            curriculum_weight = self.curriculum.get_blend_weight()
            router_loss = router_loss * curriculum_weight
            
            total_loss = (lm_loss + self.config['router_loss_weight'] * router_loss)
            total_loss = total_loss / self.grad_accum_steps
        
        # Backward
        self.scaler.scale(total_loss).backward()
        
        # Step optimizer every N accumulations
        if (accumulation_step + 1) % self.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.global_step += 1
        
        self.curriculum.step()
        
        return {
            'lm_loss': lm_loss.item(),
            'router_loss': router_loss.item() * self.grad_accum_steps,
            'total_loss': total_loss.item() * self.grad_accum_steps,
        }
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            metrics = self.train_step(input_ids, labels, batch_idx)
            
            total_loss += metrics['total_loss']
            num_batches += 1
            
            # Update progress
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'step': self.global_step
                })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.config['log_interval'] == 0:
                import wandb
                wandb.log({
                    'train/loss': metrics['total_loss'],
                    'train/lm_loss': metrics['lm_loss'],
                    'train/router_loss': metrics['router_loss'],
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/step': self.global_step,
                })
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start
        
        print(f"\n✓ Epoch {epoch}: {epoch_time:.1f}s, loss={avg_loss:.4f}")
        
        return {
            'avg_loss': avg_loss,
            'epoch_time': epoch_time,
            'num_batches': num_batches,
            'global_step': self.global_step
        }
    
    @torch.no_grad()
    def validate(self):
        """Validate"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss, _, _ = self.model(input_ids, labels=labels, deterministic=True, return_router_outputs=False)
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        # Always save latest
        latest_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        print(f"✓ Saved latest checkpoint")
        
        # Save best
        if is_best:
            best_path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            print(f"✓ Saved best checkpoint: val_loss={val_loss:.4f}")
        
        # Save epoch checkpoint every save_every epochs
        if epoch % self.config['save_every'] == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
            print(f"✓ Saved epoch {epoch} checkpoint")
    
    def save_training_log(self, epoch_metrics):
        """Save training log as JSON"""
        self.training_log.append(epoch_metrics)
        
        with open(self.training_log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        print(f"✓ Saved training log ({len(self.training_log)} epochs)")
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"Starting from epoch: {self.current_epoch}")
        print(f"Steps: {self.total_steps}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_loss, ppl = self.validate()
            
            print(f"Validation: loss={val_loss:.4f}, ppl={ppl:.2f}")
            
            # Prepare epoch log
            epoch_log = {
                'epoch': epoch,
                'timestamp': datetime.now().isoformat(),
                'train': {
                    'loss': train_metrics['avg_loss'],
                    'time_seconds': train_metrics['epoch_time'],
                    'num_batches': train_metrics['num_batches'],
                    'samples_per_sec': train_metrics['num_batches'] * self.config['batch_size'] / train_metrics['epoch_time'],
                },
                'val': {
                    'loss': val_loss,
                    'perplexity': ppl
                },
                'global_step': train_metrics['global_step'],
                'learning_rate': self.scheduler.get_last_lr()[0],
                'is_best': val_loss < self.best_val_loss
            }
            
            # Save training log
            self.save_training_log(epoch_log)
            
            # Wandb logging
            if self.use_wandb:
                import wandb
                wandb.log({
                    'val/loss': val_loss,
                    'val/perplexity': ppl,
                    'epoch': epoch
                })
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Always save (latest always saved, best if improved, epoch periodically)
            self.save_checkpoint(epoch, val_loss, is_best)
        
        print(f"\n{'='*60}")
        print(f"✓ TRAINING COMPLETE!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Training log saved to: {self.training_log_path}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from (e.g., outputs/narrativeqa/checkpoint_best.pt)')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)
    args = parser.parse_args()
    
    config = load_config_from_yaml(args.config)
    
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.run_name:
        config['run_name'] = args.run_name
    if args.no_wandb:
        config['use_wandb'] = False
    if args.max_train_samples:
        config['max_train_samples'] = args.max_train_samples
    if args.max_val_samples:
        config['max_val_samples'] = args.max_val_samples
    
    trainer = OptimizedQATrainer(config)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


if __name__ == '__main__':
    main()