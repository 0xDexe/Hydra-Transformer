import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Union, Any, overload, Literal
from collections import defaultdict


class GradientScaler(torch.autograd.Function):
    """Scale gradients during backward pass without affecting forward pass"""
    @staticmethod
    def forward(ctx, input: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return input
    
    @staticmethod
    def backward(ctx, grad_outputs: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return grad_outputs * ctx.scale, None


class EfficientTokenRouter(nn.Module):
    """
    Efficient learned token router with safeguards against:
    - Positional bias
    - Routing collapse
    - Training instability
    - Gradient imbalance
    
    Design principles:
    - Minimal parameters (~0.05% of model)
    - Fast inference (single linear layer option)
    - JIT-compilable core operations
    - Comprehensive monitoring
    """
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        target_ratio: float = 0.15,
        use_position_invariance: bool = True,
        use_threshold: bool = True,  # threshold vs top-k
        layer_idx: int = 0,
        total_layers: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.target_ratio = target_ratio
        self.use_position_invariance = use_position_invariance
        self.use_threshold = use_threshold
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        
        # Lightweight router network
        if hidden_dim > 0:
            # Two-layer MLP for better expressiveness
            self.router = nn.Sequential(
                nn.LayerNorm(d_model),  # Stabilize inputs
                nn.Linear(d_model, hidden_dim, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1, bias=True)
            )
        else:
            # Single linear layer for maximum speed
            self.router = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, 1, bias=True)
            )
        
        # Learnable threshold for routing decision
        # Initialize based on target ratio: sigmoid^-1(target_ratio)
        init_threshold = torch.log(torch.tensor(target_ratio / (1 - target_ratio)))
        self.threshold = nn.Parameter(init_threshold)
        
        # Layer-specific bias to encourage diversity across layers
        # Earlier layers route less, later layers route more
        layer_bias = (layer_idx / max(total_layers - 1, 1) - 0.5) * 0.2
        self.layer_bias = nn.Parameter(torch.tensor(layer_bias))
        
        # Position bias (small, optional)
        # We want content to dominate, but some position awareness can help
        if not use_position_invariance:
            self.register_buffer('position_bias_scale', torch.tensor(0.05))
            self.position_mlp = nn.Sequential(
                nn.Linear(d_model, 16),
                nn.GELU(),
                nn.Linear(16, 1)
            )
        else:
            self.register_buffer('position_bias_scale', torch.tensor(0.0))
            self.position_mlp = None
        
        # For monitoring
        self.register_buffer('step_count', torch.tensor(0))
        self.training_stats = defaultdict(list)
    
    def compute_content_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute routing scores based on token content
        Args:
            x: (batch, seqlen, d_model)
        Returns:
            scores: (batch, seqlen)
        """
        scores = self.router(x).squeeze(-1)
        return scores
    
    def compute_position_bias(self, x: torch.Tensor) -> torch.Tensor:
        """
        Optional: small position bias
        Args:
            x: (batch, seqlen, d_model)
        Returns:
            bias: (batch, seqlen)
        """
        if self.position_mlp is None or self.position_bias_scale == 0:
            return torch.zeros(x.shape[0], x.shape[1], device=x.device)
        
        pos_bias = self.position_mlp(x).squeeze(-1)
        return pos_bias * self.position_bias_scale
    
    def get_routing_mask(
    self,
    scores: torch.Tensor,
    deterministic: bool = False
) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Convert scores to binary routing decisions"""
        batch, seqlen = scores.shape
        
        # Add threshold to scores
        logits = scores + self.threshold
        
        # Apply sigmoid
        probs = torch.sigmoid(logits)
        
        # CRITICAL: Clamp to [0, 1]
        probs = torch.clamp(probs, min=0.0, max=1.0)
        
        # SAFETY: Handle NaN/Inf
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            probs = torch.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        
        if self.use_threshold:
            if deterministic or not self.training:
                mask = probs > 0.5  # Correct decision boundary
            else:
                mask = torch.bernoulli(probs).bool()  # Now safe!
        else:
            # Top-k routing
            k = max(1, int(seqlen * self.target_ratio))
            top_k_values, top_k_indices = torch.topk(probs, k, dim=1)
            mask = torch.zeros_like(probs, dtype=torch.bool)
            mask.scatter_(1, top_k_indices, True)
        
        # Compute statistics  
        actual_ratio = mask.float().mean().item()
        stats = {
            'routing_ratio': actual_ratio,
            'mean_prob': probs.mean().item(),
            'prob_std': probs.std().item(),
            'threshold': torch.sigmoid(self.threshold).item(),
            'num_routed': mask.sum().item(),
        }
        
        return mask, stats
        
    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False,
        return_probs: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Main forward pass
        Args:
            x: (batch, seqlen, d_model) input representations
            deterministic: if True, use deterministic routing
            return_probs: if True, return routing probabilities
        Returns:
            mask: (batch, seqlen) boolean routing mask
            aux: dictionary with auxiliary outputs (tensors and floats) for loss computation
        """
        # Compute content-based scores
        content_scores = self.compute_content_scores(x)
        
        # Add optional position bias
        pos_bias = self.compute_position_bias(x)
        
        # Add layer-specific bias
        total_scores = content_scores + pos_bias + self.layer_bias
        
        # Get routing mask
        mask, stats = self.get_routing_mask(total_scores, deterministic)
        
        # Compute probabilities for loss
        probs = torch.sigmoid(total_scores + self.threshold)

        
        # Prepare auxiliary outputs
        aux = {
            'router_probs': probs,
            'router_logits': total_scores,
            'routing_mask': mask,
            'stats': stats
        }
        
        if return_probs:
            aux['raw_probs'] = probs
        
        # Update step count
        if self.training:
            self.step_count += 1
        
        return mask, aux


class RouterCurriculum:
    """
    Curriculum learning for router training
    Gradually transitions from heuristic to learned routing
    """
    def __init__(
        self,
        total_steps: int,
        warmup_steps: int = 2000,
        heuristic_type: str = 'uniform'  # 'uniform', 'entropy', 'random'
    ):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.heuristic_type = heuristic_type
        self.current_step = 0
    
    def get_blend_weight(self) -> float:
        """Get current blending weight (0 = heuristic, 1 = learned)"""
        if self.current_step < self.warmup_steps:
            return 0.0
        progress = (self.current_step - self.warmup_steps) / \
                   max(self.total_steps - self.warmup_steps, 1)
        return min(1.0, progress)
    
    def get_heuristic_scores(
        self,
        x: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute heuristic routing scores
        Args:
            x: (batch, seqlen, d_model)
            logits: (batch, seqlen, vocab) optional, for entropy-based
        Returns:
            scores: (batch, seqlen)
        """
        batch, seqlen, d_model = x.shape
        
        if self.heuristic_type == 'uniform':
            # Uniform random
            return torch.rand(batch, seqlen, device=x.device)
        
        elif self.heuristic_type == 'entropy' and logits is not None:
            # Route high-entropy (uncertain) tokens
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
            # Normalize to [0, 1]
            entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-10)
            return entropy_norm
        
        elif self.heuristic_type == 'random':
            # Random but consistent per position
            return torch.rand(seqlen, device=x.device).unsqueeze(0).expand(batch, -1)
        
        else:
            return torch.rand(batch, seqlen, device=x.device)
    
    def blend_scores(
        self,
        learned_scores: torch.Tensor,
        x: torch.Tensor,
        logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Blend learned and heuristic scores based on curriculum
        """
        weight = self.get_blend_weight()
        
        if weight >= 1.0:
            return learned_scores
        
        if weight <= 0.0:
            return self.get_heuristic_scores(x, logits)
        
        heuristic = self.get_heuristic_scores(x, logits)
        return weight * learned_scores + (1 - weight) * heuristic
    
    def step(self):
        """Increment curriculum step"""
        self.current_step += 1


class RouterLoss:
    """
    Comprehensive loss function for router training
    Includes:
    - Load balancing
    - Entropy regularization
    - Position invariance
    - Layer diversity
    """
    def __init__(
        self,
        target_ratio: float = 0.15,
        load_weight: float = 0.01,
        entropy_weight: float = 0.01,
        position_inv_weight: float = 0.005,
        diversity_weight: float = 0.005,
        variance_weight: float = 0.01
    ):
        self.target_ratio = target_ratio
        self.load_weight = load_weight
        self.entropy_weight = entropy_weight
        self.position_inv_weight = position_inv_weight
        self.diversity_weight = diversity_weight
        self.variance_weight = variance_weight
    
    def load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Encourage routing ratio close to target
        Args:
            router_probs: (batch, seqlen)
        """
        mean_prob = router_probs.mean()
        return (mean_prob - self.target_ratio) ** 2
    
    def entropy_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Encourage diverse routing (high entropy)
        Prevents all tokens having same probability
        """
        # Clamp to avoid log(0)
        p = router_probs.clamp(1e-7, 1 - 1e-7)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log()).mean()
        # Maximize entropy (return negative)
        return -entropy
    
    def variance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        """
        Encourage variance in routing probabilities
        Prevents all probabilities being identical
        """
        return -router_probs.var()
    
    def position_invariance_loss(
        self,
        router_probs: torch.Tensor,
        num_permutations: int = 2
    ) -> torch.Tensor:
        """
        Encourage position-invariant routing
        Test that permuting sequence doesn't drastically change routing
        """
        if not self.position_inv_weight > 0:
            return torch.tensor(0.0, device=router_probs.device)
        
        batch, seqlen = router_probs.shape
        
        if seqlen <= 1:
            return torch.tensor(0.0, device=router_probs.device)
        
        loss = 0.0
        sorted_orig = router_probs.sort(dim=1).values
        
        for _ in range(num_permutations):
            # Random permutation
            perm = torch.randperm(seqlen, device=router_probs.device)
            permuted_probs = router_probs[:, perm]
            sorted_perm = permuted_probs.sort(dim=1).values
            
            # Sorted values should be similar regardless of position
            loss += F.mse_loss(sorted_orig, sorted_perm)
        
        return loss / num_permutations
    
    def layer_diversity_loss(
        self,
        all_router_probs: list
    ) -> torch.Tensor:
        """
        Encourage different layers to route different tokens
        Args:
            all_router_probs: list of (batch, seqlen) tensors
        """
        if len(all_router_probs) < 2:
            return torch.tensor(0.0, device=all_router_probs[0].device)
        
        # Stack: (n_layers, batch, seqlen)
        stacked = torch.stack(all_router_probs, dim=0)
        n_layers, batch, seqlen = stacked.shape
        
        # Flatten for correlation: (n_layers, batch * seqlen)
        flat = stacked.reshape(n_layers, -1)
        
        # Compute pairwise correlations
        total_corr = 0.0
        num_pairs = 0
        
        for i in range(n_layers):
            for j in range(i + 1, n_layers):
                # Cosine similarity between routing patterns
                corr = F.cosine_similarity(flat[i], flat[j], dim=0)
                total_corr += corr.abs()
                num_pairs += 1
        
        # Penalize high correlation
        avg_corr = total_corr / max(num_pairs, 1)
        return avg_corr
    
    @overload
    def __call__(
        self,
        router_outputs: list,
        return_components: Literal[False] = False
    ) -> torch.Tensor:
        ...
    
    @overload
    def __call__(
        self,
        router_outputs: list,
        return_components: Literal[True]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        ...
    
    def __call__(
        self,
        router_outputs: list,  # List of aux dicts from each layer
        return_components: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Compute total router loss
        Args:
            router_outputs: list of aux dictionaries from each layer
            return_components: if True, return tuple of (loss, dict of loss components)
        Returns:
            If return_components=False: total_loss (Tensor)
            If return_components=True: (total_loss, components_dict)
        """
        all_probs = [out['router_probs'] for out in router_outputs]
        
        # Compute individual losses
        load_loss = sum(self.load_balance_loss(p) for p in all_probs) / len(all_probs)
        entropy_loss = sum(self.entropy_loss(p) for p in all_probs) / len(all_probs)
        variance_loss = sum(self.variance_loss(p) for p in all_probs) / len(all_probs)
        
        position_inv_loss = sum(
            self.position_invariance_loss(p) for p in all_probs
        ) / len(all_probs)
        
        diversity_loss = self.layer_diversity_loss(all_probs)
        
        # Weighted sum
        total_loss = (
            self.load_weight * load_loss +
            self.entropy_weight * entropy_loss +
            self.variance_weight * variance_loss +
            self.position_inv_weight * position_inv_loss +
            self.diversity_weight * diversity_loss
        )
        
        if return_components:
            # Helper to safely convert tensor or float to float
            def to_float(x):
                if isinstance(x, torch.Tensor):
                    return x.item()
                return float(x)
            
            return total_loss, {
                'load_balance': to_float(load_loss),
                'entropy': to_float(entropy_loss),
                'variance': to_float(variance_loss),
                'position_invariance': to_float(position_inv_loss),
                'layer_diversity': to_float(diversity_loss),
                'total': to_float(total_loss)
            }
        
        return total_loss


class RouterMonitor:
    """
    Comprehensive monitoring for router behavior
    Tracks:
    - Routing ratios per layer
    - Position correlations
    - Entropy metrics
    - Collapse detection
    """
    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.metrics = defaultdict(list)
        self.collapse_threshold = 0.05
    
    def compute_entropy(self, probs: torch.Tensor) -> float:
        """Binary entropy of routing probabilities"""
        p = probs.clamp(1e-10, 1 - 1e-10)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        return entropy.mean().item()
    
    def compute_position_correlation(self, probs: torch.Tensor) -> float:
        """Correlation between position and routing probability"""
        batch, seqlen = probs.shape
        positions = torch.arange(seqlen, device=probs.device).float()
        positions = positions.unsqueeze(0).expand(batch, -1)
        
        # Flatten and compute correlation
        pos_flat = positions.flatten()
        prob_flat = probs.flatten()
        
        # Pearson correlation
        pos_mean = pos_flat.mean()
        prob_mean = prob_flat.mean()
        
        covariance = ((pos_flat - pos_mean) * (prob_flat - prob_mean)).mean()
        pos_std = pos_flat.std()
        prob_std = prob_flat.std()
        
        if pos_std > 0 and prob_std > 0:
            corr = covariance / (pos_std * prob_std)
            return corr.abs().item()
        return 0.0
    
    def detect_collapse(self, probs: torch.Tensor, target_ratio: float) -> Dict:
        """Detect if routing has collapsed"""
        actual_ratio = (probs > 0.5).float().mean().item()
        deviation = abs(actual_ratio - target_ratio)
        entropy = self.compute_entropy(probs)
        
        # Collapse indicators:
        # 1. Large deviation from target ratio
        # 2. Low entropy (all same decision)
        # 3. Very high or very low variance
        
        collapsed = (
            deviation > self.collapse_threshold or
            entropy < 0.1 or
            probs.var() < 0.01
        )
        
        return {
            'collapsed': collapsed,
            'deviation': deviation,
            'entropy': entropy,
            'variance': probs.var().item(),
            'actual_ratio': actual_ratio
        }
    
    def log_layer_stats(
        self,
        layer_idx: int,
        router_probs: torch.Tensor,
        step: int,
        target_ratio: float = 0.15
    ) -> Dict[str, float]:
        """Log statistics for a single layer"""
        stats = {
            f'layer_{layer_idx}/routing_ratio': (router_probs > 0.5).float().mean().item(),
            f'layer_{layer_idx}/mean_prob': router_probs.mean().item(),
            f'layer_{layer_idx}/std_prob': router_probs.std().item(),
            f'layer_{layer_idx}/entropy': self.compute_entropy(router_probs),
            f'layer_{layer_idx}/position_corr': self.compute_position_correlation(router_probs),
        }
        
        # Check for collapse
        collapse_info = self.detect_collapse(router_probs, target_ratio)
        stats[f'layer_{layer_idx}/collapsed'] = float(collapse_info['collapsed'])
        
        # Store metrics
        for k, v in stats.items():
            self.metrics[k].append((step, v))
        
        return stats
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics across all layers"""
        if not self.metrics:
            return {}
        
        summary = {}
        
        # Average across layers for each metric type
        metric_types = ['routing_ratio', 'entropy', 'position_corr', 'collapsed']
        
        for metric_type in metric_types:
            values = []
            for layer_idx in range(self.num_layers):
                key = f'layer_{layer_idx}/{metric_type}'
                if key in self.metrics and self.metrics[key]:
                    values.append(self.metrics[key][-1][1])
            
            if values:
                summary[f'avg_{metric_type}'] = sum(values) / len(values)
                summary[f'max_{metric_type}'] = max(values)
                summary[f'min_{metric_type}'] = min(values)
        
        return summary