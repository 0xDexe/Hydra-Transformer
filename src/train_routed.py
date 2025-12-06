import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import os
from pathlib import Path
from typing import Optional, Dict, Tuple
import json

from model.routed_model import RoutedHybridModel
from model.blocks.router import RouterLoss, RouterMonitor, RouterCurriculum
from data.dataset import get_dataloaders


class RoutedTrainer:
    """
    Trainer for HYDRA with learned token routing
    
    Features:
    - Curriculum learning for router
    - Comprehensive monitoring
    - Router loss with multiple objectives
    - Gradient balancing
    - Automatic checkpointing
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup wandb
        if config.use_wandb:
            wandb.init(
                project=config.project_name,
                config=vars(config),
                name=config.run_name
            )
        
        # Create model
        self.model = RoutedHybridModel(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            d_ff=config.d_ff,
            dropout=config.dropout,
            router_hidden_dim=config.router_hidden_dim,
            target_ratio=config.target_ratio,
            use_gradient_balancing=config.use_gradient_balancing,
            use_position_invariance=config.use_position_invariance,
            use_checkpoint=config.use_checkpoint,
            tie_weights=config.tie_weights
        ).to(self.device)
        
        print(f"Model parameters: {self.model.get_num_params() / 1e6:.2f}M")
        print(f"Non-embedding params: {self.model.get_num_params(non_embedding=True) / 1e6:.2f}M")
        
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
        
        # Optimizer with parameter groups
        # Separate learning rates for router vs main model (optional)
        if config.router_lr_multiplier != 1.0:
            router_params = []
            model_params = []
            for name, param in self.model.named_parameters():
                if 'router' in name:
                    router_params.append(param)
                else:
                    model_params.append(param)
            
            self.optimizer = AdamW([
                {'params': model_params, 'lr': config.learning_rate},
                {'params': router_params, 'lr': config.learning_rate * config.router_lr_multiplier}
            ], weight_decay=config.weight_decay, betas=(0.9, 0.95))
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                betas=(0.9, 0.95)
            )
        
        # Scheduler
        total_steps = config.num_epochs * len(self.train_loader)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        # Router curriculum
        self.curriculum = RouterCurriculum(
            total_steps=total_steps,
            warmup_steps=config.router_warmup_steps,
            heuristic_type=config.curriculum_heuristic
        )
        
        # Router loss
        self.router_loss_fn = RouterLoss(
            target_ratio=config.target_ratio,
            load_weight=config.load_loss_weight,
            entropy_weight=config.entropy_loss_weight,
            position_inv_weight=config.position_inv_weight,
            diversity_weight=config.diversity_loss_weight,
            variance_weight=config.variance_loss_weight
        )
        
        # Router monitor
        self.router_monitor = RouterMonitor(num_layers=config.n_layers)
        
        # For checkpointing
        self.best_val_loss = float('inf')
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
    
    def train_step(self, batch, step_in_epoch) -> Dict[str, float]:
        """Single training step"""
        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass with router outputs
        lm_loss, logits, router_outputs = self.model(
            input_ids,
            labels=labels,
            deterministic=False,  # Stochastic routing during training
            return_router_outputs=True
        )
        
        # Compute router loss
        result = self.router_loss_fn(router_outputs, return_components=True)
        # Unpack with type safety
        router_loss: torch.Tensor
        router_loss_components: Dict[str, float]
        router_loss, router_loss_components = result
        
        # Apply curriculum to router loss
        curriculum_weight = self.curriculum.get_blend_weight()
        router_loss = router_loss * curriculum_weight
        
        # Total loss
        total_loss = lm_loss + self.config.router_loss_weight * router_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.grad_clip
        )
        
        self.optimizer.step()
        self.scheduler.step()
        self.curriculum.step()
        
        # Collect metrics
        metrics = {
            'train/lm_loss': lm_loss.item(),
            'train/router_loss': router_loss.item(),
            'train/total_loss': total_loss.item(),
            'train/lr': self.scheduler.get_last_lr()[0],
            'train/curriculum_weight': curriculum_weight,
            'train/step': self.global_step
        }
        
        # Add router loss components
        for k, v in router_loss_components.items():
            metrics[f'train/router_{k}'] = v
        
        # Add routing statistics (average across layers)
        if router_outputs:
            avg_routing_ratio = sum(
                out['routing_ratio'] for out in router_outputs
            ) / len(router_outputs)
            metrics['train/avg_routing_ratio'] = avg_routing_ratio
        
        return metrics
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        self.current_epoch = epoch
        
        total_lm_loss = 0
        total_router_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            metrics = self.train_step(batch, batch_idx)
            
            total_lm_loss += metrics['train/lm_loss']
            total_router_loss += metrics['train/router_loss']
            
            # Update progress bar
            pbar.set_postfix({
                'lm_loss': f"{metrics['train/lm_loss']:.4f}",
                'router_loss': f"{metrics['train/router_loss']:.4f}",
                'lr': f"{metrics['train/lr']:.2e}"
            })
            
            # Log to wandb
            if self.config.use_wandb and batch_idx % self.config.log_interval == 0:
                wandb.log(metrics)
            
            # Monitor routing every N steps
            if batch_idx % self.config.monitor_interval == 0:
                self.log_routing_stats(batch, batch_idx)
            
            self.global_step += 1
        
        avg_lm_loss = total_lm_loss / len(self.train_loader)
        avg_router_loss = total_router_loss / len(self.train_loader)
        
        return avg_lm_loss, avg_router_loss
    
    @torch.no_grad()
    def log_routing_stats(self, batch, step):
        """Log detailed routing statistics"""
        input_ids = batch['input_ids'].to(self.device)
        
        # Get routing decisions
        _, _, router_outputs = self.model(
            input_ids,
            deterministic=True,
            return_router_outputs=True
        )
        
        if not router_outputs:
            return
        
        # Log statistics for each layer
        for layer_idx, output in enumerate(router_outputs):
            stats = self.router_monitor.log_layer_stats(
                layer_idx=layer_idx,
                router_probs=output['router_probs'],
                step=self.global_step,
                target_ratio=self.config.target_ratio
            )
            
            if self.config.use_wandb and step % (self.config.monitor_interval * 5) == 0:
                wandb.log(stats)
        
        # Log summary statistics
        if self.config.use_wandb:
            summary = self.router_monitor.get_summary()
            if summary:
                wandb.log({f'routing_summary/{k}': v for k, v in summary.items()})
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Deterministic routing for validation
            loss, logits, _ = self.model(
                input_ids,
                labels=labels,
                deterministic=True
            )
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return avg_loss, perplexity.item()
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint with routing statistics"""
        # Get current routing stats
        routing_stats = self.model.get_routing_stats()
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'config': vars(self.config),
            'routing_stats': routing_stats,
            'curriculum_step': self.curriculum.current_step
        }
        
        # Save latest
        path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, path)
        
        # Save best
        if is_best:
            path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, path)
            print(f"✓ Saved best model with val_loss: {val_loss:.4f}")
        
        # Save periodic checkpoints
        if epoch % self.config.save_every == 0:
            path = self.output_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['val_loss']
        self.curriculum.current_step = checkpoint.get('curriculum_step', 0)
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Global step: {self.global_step}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}")
        print(f"Model: {self.model.get_num_params() / 1e6:.2f}M parameters")
        print(f"Target routing ratio: {self.config.target_ratio:.2%}")
        print(f"Router warmup steps: {self.config.router_warmup_steps}")
        print(f"Total epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Output dir: {self.output_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_lm_loss, train_router_loss = self.train_epoch(epoch)
            print(f"Train LM loss: {train_lm_loss:.4f}")
            print(f"Train Router loss: {train_router_loss:.4f}")
            
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
            
            # Print routing summary
            summary = self.router_monitor.get_summary()
            if summary:
                print(f"\nRouting Summary:")
                print(f"  Avg routing ratio: {summary.get('avg_routing_ratio', 0):.2%}")
                print(f"  Avg entropy: {summary.get('avg_entropy', 0):.4f}")
                print(f"  Avg position correlation: {summary.get('avg_position_corr', 0):.4f}")
                print(f"  Collapsed layers: {summary.get('avg_collapsed', 0):.0f}")
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best validation perplexity: {torch.exp(torch.tensor(self.best_val_loss)):.2f}")
        print("="*60 + "\n")
        
        # Save final routing statistics
        self.save_routing_analysis()
    
    def save_routing_analysis(self):
        """Save detailed routing analysis"""
        analysis = {
            'metrics': dict(self.router_monitor.metrics),
            'summary': self.router_monitor.get_summary(),
            'final_routing_stats': self.model.get_routing_stats(),
            'config': vars(self.config)
        }
        
        output_path = self.output_dir / 'routing_analysis.json'
        with open(output_path, 'w') as f:
            # Convert tensors to lists for JSON serialization
            def convert_values(obj):
                if isinstance(obj, dict):
                    return {k: convert_values(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_values(item) for item in obj]
                elif isinstance(obj, torch.Tensor):
                    return obj.item() if obj.numel() == 1 else obj.tolist()
                else:
                    return obj
            
            json.dump(convert_values(analysis), f, indent=2)
        
        print(f"✓ Saved routing analysis to {output_path}")


class RoutedTrainConfig:
    """Configuration for routed HYDRA training"""
    def __init__(self):
        # Model architecture
        self.d_model = 768
        self.n_layers = 12
        self.n_heads = 12
        self.d_state = 16
        self.d_conv = 4
        self.expand = 2
        self.d_ff = None  # Will default to 4 * d_model
        self.dropout = 0.1
        self.tie_weights = True
        
        # Router configuration
        self.router_hidden_dim = 64
        self.target_ratio = 0.15
        self.use_gradient_balancing = True
        self.use_position_invariance = True
        self.use_checkpoint = False  # Activation checkpointing
        
        # Router curriculum
        self.router_warmup_steps = 2000
        self.curriculum_heuristic = 'uniform'  # 'uniform', 'entropy', 'random'
        
        # Router loss weights
        self.router_loss_weight = 0.01
        self.load_loss_weight = 0.01
        self.entropy_loss_weight = 0.01
        self.variance_loss_weight = 0.01
        self.position_inv_weight = 0.005
        self.diversity_loss_weight = 0.005
        
        # Data
        self.dataset_name = 'wikitext'
        self.dataset_config = 'wikitext-103-v1'
        self.tokenizer_name = 'gpt2'
        self.max_length = 1024
        self.batch_size = 8
        self.num_workers = 4
        self.vocab_size = 50257  # Will be updated from tokenizer
        
        # Training
        self.num_epochs = 20
        self.learning_rate = 3e-4
        self.router_lr_multiplier = 1.0  # Separate LR for router if != 1.0
        self.weight_decay = 0.01
        self.grad_clip = 1.0
        
        # Logging & checkpointing
        self.use_wandb = True
        self.project_name = 'hydra-routed'
        self.run_name = 'routed-hybrid-v1'
        self.output_dir = 'outputs/routed-v1'
        self.log_interval = 10
        self.monitor_interval = 100  # How often to log detailed routing stats
        self.save_every = 5  # Save checkpoint every N epochs


if __name__ == '__main__':
    config = RoutedTrainConfig()
    trainer = RoutedTrainer(config)
    trainer.train()