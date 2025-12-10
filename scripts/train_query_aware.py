"""
Complete Training Script for Query-Aware HYDRA

Properly integrates:
- QueryAwareHybridModel
- QuestionExtractor for identifying question spans
- Router losses and monitoring
- Checkpointing and logging

Usage:
    python scripts/train_query_aware_complete.py --config configs/query_aware_naturalqa.yaml
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.query_aware_model import QueryAwareHybridModel
from src.model.question_extractor import QuestionExtractor
from src.model.router import RouterLoss, RouterMonitor, RouterCurriculum
from src.data.qa_datasets import get_qa_dataloaders


def load_config_from_yaml(yaml_path):
    """Load configuration from YAML file"""
    with open(yaml_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # Flatten nested structure
    config = {}
    for section in ['model', 'router', 'data', 'training', 'router_loss', 'curriculum', 'logging']:
        if section in cfg_dict:
            config.update(cfg_dict[section])
    
    return config


class QueryAwareTrainer:
    """Complete trainer for query-aware HYDRA"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*70}")
        print("QUERY-AWARE HYDRA TRAINING")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        # Load data
        print("\nLoading dataset...")
        self.train_loader, self.val_loader, self.tokenizer = get_qa_dataloaders(
            dataset_name=config['dataset_name'],
            tokenizer_name=config['tokenizer_name'],
            max_length=config['max_length'],
            batch_size=config['batch_size'],
            num_workers=config.get('num_workers', 4),
            max_train_samples=config.get('max_train_samples', None),
            max_val_samples=config.get('max_val_samples', None),
        )
        
        config['vocab_size'] = len(self.tokenizer)
        
        print(f"✓ Dataset: {config['dataset_name']}")
        print(f"✓ Train batches: {len(self.train_loader)}")
        print(f"✓ Val batches: {len(self.val_loader)}")
        
        # Create question extractor
        print("\nInitializing question extractor...")
        self.question_extractor = QuestionExtractor(self.tokenizer)
        print("✓ Question extractor ready")
        
        # Create model
        print("\nCreating query-aware model...")
        self.model = QueryAwareHybridModel(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_state=config.get('d_state', 16),
            d_conv=config.get('d_conv', 4),
            expand=config.get('expand', 2),
            d_ff=config.get('d_ff', 3072),
            dropout=config.get('dropout', 0.1),
            target_ratio=config.get('target_ratio', 0.15),
            router_hidden_dim=config.get('hidden_dim', 64),
            use_gradient_balancing=config.get('use_gradient_balancing', True),
            use_position_invariance=config.get('use_position_invariance', True),
            tie_weights=config.get('tie_weights', True),
        ).to(self.device)
        
        print(f"✓ Model: {self.model.get_num_params() / 1e6:.1f}M parameters")
        
        # Optimizer
        self.use_amp = config.get('use_amp', True) and self.device.type == 'cuda'
        
        optimizer_kwargs = {
            'lr': config['learning_rate'],
            'weight_decay': config['weight_decay'],
            'betas': (0.9, 0.95)
        }
        
        if config.get('use_fused_optimizer', True) and self.device.type == 'cuda':
            try:
                optimizer_kwargs['fused'] = True
                self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
                print("✓ Fused AdamW optimizer")
            except:
                self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
                print("✓ Standard AdamW optimizer")
        else:
            self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
        
        # Gradient scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # Gradient accumulation
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)
        
        # Scheduler
        self.total_steps = config['num_epochs'] * len(self.train_loader) // self.grad_accum_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps,
            eta_min=config['learning_rate'] * 0.1
        )
        
        # Router components
        self.curriculum = RouterCurriculum(
            total_steps=self.total_steps,
            warmup_steps=config.get('router_warmup_steps', 1000),
            heuristic_type=config.get('curriculum_heuristic', 'uniform')
        )
        
        self.router_loss_fn = RouterLoss(
            target_ratio=config.get('target_ratio', 0.15),
            load_weight=config.get('load_balance', 0.01),
            entropy_weight=config.get('entropy', 0.01),
            position_inv_weight=config.get('position_invariance', 0.005),
            diversity_weight=config.get('layer_diversity', 0.005),
            variance_weight=config.get('variance', 0.01)
        )
        
        self.monitor = RouterMonitor(num_layers=config['n_layers'])
        
        # Checkpointing
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float('inf')
        
        # State
        self.global_step = 0
        self.current_epoch = 0
        
        # Training log
        self.training_log = []
        self.training_log_path = self.output_dir / 'training_log.json'
        
        # Load existing log if resuming
        if self.training_log_path.exists():
            with open(self.training_log_path, 'r') as f:
                self.training_log = json.load(f)
        
        # Wandb
        if config.get('use_wandb', False):
            import wandb
            wandb.init(
                project=config.get('project_name', 'hydra-query-aware'),
                name=config.get('run_name', 'query-aware-run'),
                config=config
            )
            self.use_wandb = True
        else:
            self.use_wandb = False
        
        print(f"✓ Output directory: {self.output_dir}")
        print(f"{'='*70}\n")
    
    def extract_question_masks(self, batch: Dict) -> torch.Tensor:
        """
        Extract question masks from batch using QuestionExtractor.
        
        This is where question_extractor.py is actually used!
        """
        input_ids = batch['input_ids']
        
        # Try to get text from batch
        if 'text' in batch:
            texts = batch['text']
            question_mask = self.question_extractor.extract_question_mask_from_ids(
                input_ids,
                texts
            )
        elif 'question' in batch and 'context' in batch:
            # Reconstruct full text from question + context
            contexts = batch.get('context', [''] * len(batch['question']))
            questions = batch['question']
            
            full_texts = []
            for ctx, q in zip(contexts, questions):
                if ctx:
                    full_text = f"Context: {ctx}\n\nQuestion: {q}\n\nAnswer:"
                else:
                    full_text = f"Question: {q}\n\nAnswer:"
                full_texts.append(full_text)
            
            question_mask = self.question_extractor.extract_question_mask_from_ids(
                input_ids,
                full_texts
            )
        else:
            # No explicit question info - model will use heuristic
            question_mask = None
        
        return question_mask
    
    def train_step(self, batch: Dict, accumulation_step: int) -> Dict:
        """Single training step"""
        
        input_ids = batch['input_ids'].to(self.device, non_blocking=True)
        labels = batch['labels'].to(self.device, non_blocking=True)
        
        # Extract question masks
        question_mask = self.extract_question_masks(batch)
        if question_mask is not None:
            question_mask = question_mask.to(self.device, non_blocking=True)
        
        # Forward with autocast
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            lm_loss, logits, router_outputs = self.model(
                input_ids,
                labels=labels,
                question_mask=question_mask,  # Pass question mask to model
                deterministic=False,
                return_router_outputs=True
            )
            
            # Router loss
            result = self.router_loss_fn(router_outputs, return_components=True)
            router_loss, router_components = result
            
            curriculum_weight = self.curriculum.get_blend_weight()
            router_loss = router_loss * curriculum_weight
            
            total_loss = (lm_loss + self.config.get('total_weight', 0.01) * router_loss)
            total_loss = total_loss / self.grad_accum_steps
        
        # Backward
        self.scaler.scale(total_loss).backward()
        
        # Step optimizer
        if (accumulation_step + 1) % self.grad_accum_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            self.global_step += 1
        
        self.curriculum.step()
        
        # Extract query weight from router outputs
        query_weight = router_outputs[0].get('query_weight', 0.0) if router_outputs else 0.0
        
        return {
            'lm_loss': lm_loss.item(),
            'router_loss': router_loss.item() * self.grad_accum_steps,
            'total_loss': total_loss.item() * self.grad_accum_steps,
            'query_weight': query_weight,
        }
    
    def train_epoch(self, epoch: int) -> Dict:
        """Train one epoch"""
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0
        num_batches = 0
        total_query_weight = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        epoch_start = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch, batch_idx)
            
            total_loss += metrics['total_loss']
            total_query_weight += metrics['query_weight']
            num_batches += 1
            
            # Update progress
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'qw': f"{metrics['query_weight']:.3f}",
                    'step': self.global_step
                })
            
            # Log to wandb
            if self.use_wandb and batch_idx % self.config.get('log_interval', 50) == 0:
                import wandb
                wandb.log({
                    'train/loss': metrics['total_loss'],
                    'train/lm_loss': metrics['lm_loss'],
                    'train/router_loss': metrics['router_loss'],
                    'train/query_weight': metrics['query_weight'],
                    'train/lr': self.scheduler.get_last_lr()[0],
                    'train/step': self.global_step,
                })
        
        avg_loss = total_loss / num_batches
        avg_query_weight = total_query_weight / num_batches
        epoch_time = time.time() - epoch_start
        
        print(f"\n✓ Epoch {epoch}: {epoch_time:.1f}s, loss={avg_loss:.4f}, query_weight={avg_query_weight:.3f}")
        
        return {
            'avg_loss': avg_loss,
            'epoch_time': epoch_time,
            'num_batches': num_batches,
            'global_step': self.global_step,
            'avg_query_weight': avg_query_weight,
        }
    
    @torch.no_grad()
    def validate(self) -> tuple:
        """Validate"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Extract question masks
            question_mask = self.extract_question_masks(batch)
            if question_mask is not None:
                question_mask = question_mask.to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss, _, _ = self.model(
                    input_ids,
                    labels=labels,
                    question_mask=question_mask,
                    deterministic=True,
                    return_router_outputs=False
                )
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return avg_loss, perplexity
    
    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
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
        
        # Save epoch checkpoint
        if epoch % self.config.get('save_every', 5) == 0:
            epoch_path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)
            print(f"✓ Saved epoch {epoch} checkpoint")
    
    def save_training_log(self, epoch_metrics: Dict):
        """Save training log as JSON"""
        self.training_log.append(epoch_metrics)
        
        with open(self.training_log_path, 'w') as f:
            json.dump(self.training_log, f, indent=2)
        
        print(f"✓ Saved training log ({len(self.training_log)} epochs)")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint for resuming"""
        print(f"\n{'='*70}")
        print("LOADING CHECKPOINT")
        print(f"{'='*70}")
        print(f"Path: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        print(f"✓ Resuming from epoch {self.current_epoch}")
        print(f"✓ Global step: {self.global_step}")
        print(f"✓ Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*70}\n")
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {self.config['num_epochs']}")
        print(f"Starting from epoch: {self.current_epoch}")
        print(f"Total steps: {self.total_steps}")
        print(f"{'='*70}\n")
        
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
                    'query_weight': train_metrics['avg_query_weight'],
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
                    'train/query_weight_epoch': train_metrics['avg_query_weight'],
                    'epoch': epoch
                })
            
            # Checkpointing
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(epoch, val_loss, is_best)
        
        print(f"\n{'='*70}")
        print(f"✓ TRAINING COMPLETE!")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Training log: {self.training_log_path}")
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Train Query-Aware HYDRA")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--no-wandb', action='store_true')
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)
    args = parser.parse_args()
    
    # Load config
    config = load_config_from_yaml(args.config)
    
    # Override with command-line args
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
    
    # Create trainer
    trainer = QueryAwareTrainer(config)
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()