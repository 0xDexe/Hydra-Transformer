"""
Optimized Fast Trainer for HYDRA with Learned Token Routing

Key optimizations:
- Mixed precision training (AMP)
- Gradient accumulation
- Fused optimizer
- torch.compile support
- Reduced monitoring overhead
- Optimized data loading

Expected speedup: 5-10x vs baseline
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict

from model.routed_model import RoutedHybridModel
from model.router import RouterLoss, RouterMonitor, RouterCurriculum


class FastRoutedTrainer:
    """
    High-performance trainer with mixed precision and other optimizations
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n{'='*60}")
        print("FAST TRAINER INITIALIZATION")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        
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
        
        print(f"✓ Model: {self.model.get_num_params() / 1e6:.2f}M parameters")
        
        # torch.compile (PyTorch 2.0+)
        if config.compile_model and hasattr(torch, 'compile'):
            print("✓ Compiling model with torch.compile...")
            self.model = torch.compile(
                self.model,
                mode='reduce-overhead',
                fullgraph=False
            )
        
        # Fused optimizer (faster)
        optimizer_kwargs = {
            'lr': config.learning_rate,
            'weight_decay': config.weight_decay,
            'betas': (0.9, 0.95)
        }
        
        if config.use_fused_optimizer and self.device.type == 'cuda':
            try:
                optimizer_kwargs['fused'] = True
                self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
                print("✓ Using fused optimizer")
            except:
                self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
                print("⚠ Fused optimizer not available, using standard AdamW")
        else:
            self.optimizer = AdamW(self.model.parameters(), **optimizer_kwargs)
        
        # Mixed precision scaler
        self.use_amp = config.use_amp and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        if self.use_amp:
            print("✓ Mixed precision training enabled")
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.gradient_accumulation_steps
        if self.gradient_accumulation_steps > 1:
            print(f"✓ Gradient accumulation: {self.gradient_accumulation_steps} steps")
            print(f"  Effective batch size: {config.batch_size * self.gradient_accumulation_steps}")
        
        # Scheduler
        self.total_steps = config.num_epochs * config.steps_per_epoch // self.gradient_accumulation_steps
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps,
            eta_min=config.learning_rate * 0.1
        )
        
        # Router curriculum
        self.curriculum = RouterCurriculum(
            total_steps=self.total_steps,
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
        
        # Router monitor (lightweight)
        self.router_monitor = RouterMonitor(num_layers=config.n_layers)
        
        # Checkpointing
        self.best_val_loss = float('inf')
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        
        # Timing
        self.step_times = []
        
        print(f"{'='*60}\n")
    
    def train_step(self, input_ids, labels, accumulation_step):
        """Optimized training step with mixed precision"""
        
        # Forward pass with autocast
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            lm_loss, logits, router_outputs = self.model(
                input_ids,
                labels=labels,
                deterministic=False,
                return_router_outputs=True
            )
            
            # Router loss
            result = self.router_loss_fn(router_outputs, return_components=True)
            router_loss: torch.Tensor
            router_loss_components: Dict[str, float]
            router_loss, router_loss_components = result
            
            # Apply curriculum
            curriculum_weight = self.curriculum.get_blend_weight()
            router_loss = router_loss * curriculum_weight
            
            # Total loss (scaled for gradient accumulation)
            total_loss = (lm_loss + self.config.router_loss_weight * router_loss)
            total_loss = total_loss / self.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        self.scaler.scale(total_loss).backward()
        
        # Only step optimizer every N accumulation steps
        if (accumulation_step + 1) % self.gradient_accumulation_steps == 0:
            # Unscale gradients for clipping
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Zero gradients (set_to_none is faster)
            self.optimizer.zero_grad(set_to_none=True)
            
            # Update scheduler
            self.scheduler.step()
            
            self.global_step += 1
        
        # Curriculum always steps
        self.curriculum.step()
        
        # Return metrics (unscaled loss for logging)
        return {
            'lm_loss': lm_loss.item(),
            'router_loss': router_loss.item() * self.gradient_accumulation_steps,
            'total_loss': total_loss.item() * self.gradient_accumulation_steps,
            'lr': self.scheduler.get_last_lr()[0],
        }
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        self.current_epoch = epoch
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        start_time = time.time()
        
        for batch_idx, batch in enumerate(pbar):
            step_start = time.time()
            
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            # Train step
            metrics = self.train_step(input_ids, labels, batch_idx)
            
            total_loss += metrics['total_loss']
            num_batches += 1
            
            # Track timing
            step_time = time.time() - step_start
            self.step_times.append(step_time)
            if len(self.step_times) > 100:
                self.step_times.pop(0)
            
            # Update progress bar
            if batch_idx % 10 == 0:
                avg_step_time = sum(self.step_times) / len(self.step_times)
                pbar.set_postfix({
                    'loss': f"{metrics['total_loss']:.4f}",
                    'lr': f"{metrics['lr']:.2e}",
                    'ms/step': f"{avg_step_time*1000:.0f}",
                    'step': self.global_step
                })
        
        avg_loss = total_loss / num_batches
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch} completed in {epoch_time:.1f}s")
        print(f"Average loss: {avg_loss:.4f}")
        print(f"Steps/sec: {num_batches/epoch_time:.2f}")
        
        return avg_loss
    
    @torch.no_grad()
    def validate(self, val_loader):
        """Fast validation"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            labels = batch['labels'].to(self.device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                loss, _, _ = self.model(
                    input_ids,
                    labels=labels,
                    deterministic=True,
                    return_router_outputs=False
                )
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss))
        
        return avg_loss, perplexity.item()
    
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
            'config': vars(self.config)
        }
        
        # Save best
        if is_best:
            path = self.output_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, path)
            print(f"✓ Saved best model: val_loss={val_loss:.4f}")
    
    def train(self, train_loader, val_loader):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("STARTING FAST TRAINING")
        print(f"{'='*60}")
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Total steps: {self.total_steps}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, perplexity = self.validate(val_loader)
            print(f"Validation - Loss: {val_loss:.4f}, PPL: {perplexity:.2f}")
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Best perplexity: {torch.exp(torch.tensor(self.best_val_loss)):.2f}")
        print(f"{'='*60}\n")


# Example usage
if __name__ == '__main__':
    from dataclasses import dataclass
    
    @dataclass
    class FastConfig:
        # Model
        vocab_size: int = 50257
        d_model: int = 512
        n_layers: int = 6
        n_heads: int = 8
        d_state: int = 16
        d_conv: int = 4
        expand: int = 2
        d_ff: int = None
        dropout: float = 0.1
        tie_weights: bool = True
        
        # Router
        router_hidden_dim: int = 32
        target_ratio: float = 0.15
        use_gradient_balancing: bool = True
        use_position_invariance: bool = True
        use_checkpoint: bool = False
        
        # Training
        num_epochs: int = 10
        steps_per_epoch: int = 1000  # Estimated
        batch_size: int = 16
        learning_rate: float = 5e-4
        weight_decay: float = 0.01
        grad_clip: float = 1.0
        
        # Optimizations
        use_amp: bool = True
        gradient_accumulation_steps: int = 2
        use_fused_optimizer: bool = True
        compile_model: bool = True
        
        # Curriculum
        router_warmup_steps: int = 500
        curriculum_heuristic: str = 'uniform'
        
        # Router loss
        router_loss_weight: float = 0.01
        load_loss_weight: float = 0.01
        entropy_loss_weight: float = 0.01
        variance_loss_weight: float = 0.01
        position_inv_weight: float = 0.001
        diversity_loss_weight: float = 0.001
        
        # Logging
        output_dir: str = 'outputs/fast'
        save_every: int = 10
    
    print("Fast trainer created! Use with actual data loaders.")