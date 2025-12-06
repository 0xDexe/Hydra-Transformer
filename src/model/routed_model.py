import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List, Any
import math

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

from src.model.router import EfficientTokenRouter, GradientScaler


class SSMBlock(nn.Module):
    """SSM block wrapper - imports from existing code"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        from mamba_ssm import Mamba
        self.d_model = d_model
        self.ssm = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        return x + residual


class AttentionBlock(nn.Module):
    """
    Multi-head attention with FULL dtype compatibility
    
    CRITICAL FP32 COMPATIBILITY:
    - FlashAttention requires fp16/bf16
    - Automatically converts and reverts
    - Falls back to standard attention for fp32
    - Supports sliding window attention
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1, window_size=None):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = dropout
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        """
        Args:
            x: (batch, seqlen, d_model) - any dtype
        Returns:
            output: (batch, seqlen, d_model) - same dtype as input
        """
        batch, seqlen, d_model = x.shape
        residual = x
        original_dtype = x.dtype
        
        # Pre-norm
        x = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seqlen, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        
        # Try FlashAttention if available and dtype compatible
        if HAS_FLASH_ATTN and original_dtype in [torch.float16, torch.bfloat16]:
            try:
                # FlashAttention path (fp16/bf16)
                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=True,
                    window_size=(self.window_size, 0) if self.window_size else (-1, -1)
                )
            except Exception as e:
                # Fallback to standard attention
                attn_output = self._standard_attention(q, k, v, original_dtype)
        else:
            # Standard attention for fp32 or when FlashAttention unavailable
            attn_output = self._standard_attention(q, k, v, original_dtype)
        
        # Reshape and project
        attn_output = attn_output.reshape(batch, seqlen, d_model)
        output = self.out_proj(attn_output)
        
        return output + residual
    
    def _standard_attention(self, q, k, v, dtype):
        """
        Standard attention implementation with full dtype compatibility
        
        Args:
            q, k, v: (batch, seqlen, n_heads, head_dim)
            dtype: target output dtype
        """
        batch, seqlen, n_heads, head_dim = q.shape
        
        # Convert to fp32 for numerical stability in attention computation
        q = q.float()
        k = k.float()
        v = v.float()
        
        # Rearrange to (batch, n_heads, seqlen, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Optional window mask
        if self.window_size is not None:
            window_mask = torch.abs(
                torch.arange(seqlen, device=q.device).unsqueeze(0) -
                torch.arange(seqlen, device=q.device).unsqueeze(1)
            ) > self.window_size
            scores = scores.masked_fill(window_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Rearrange back to (batch, seqlen, n_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)
        
        # Convert back to original dtype
        return attn_output.to(dtype)


class FFNBlock(nn.Module):
    """Feed-forward network"""
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
            
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class RoutedHybridLayer(nn.Module):
    """
    Efficient hybrid layer with learned token routing
    
    Architecture:
    - All tokens → SSM (efficient baseline)
    - Selected tokens (~15%) → SSM + Attention (enhanced processing)
    - All tokens → FFN
    
    Efficiency optimizations:
    - Lightweight router (~0.05% params)
    - Efficient token gathering for attention
    - Gradient balancing between paths
    - Optional activation checkpointing
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        router_hidden_dim: int = 64,
        target_ratio: float = 0.15,
        layer_idx: int = 0,
        total_layers: int = 12,
        use_gradient_balancing: bool = True,
        use_position_invariance: bool = True,
        use_checkpoint: bool = False  # Activation checkpointing
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.target_ratio = target_ratio
        self.layer_idx = layer_idx
        self.use_gradient_balancing = use_gradient_balancing
        self.use_checkpoint = use_checkpoint
        
        # Core blocks
        self.ssm_block = SSMBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        self.attn_block = AttentionBlock(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout
        )
        
        self.ffn = FFNBlock(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Token router
        self.router = EfficientTokenRouter(
            d_model=d_model,
            hidden_dim=router_hidden_dim,
            target_ratio=target_ratio,
            use_position_invariance=use_position_invariance,
            use_threshold=True,  # Faster than top-k
            layer_idx=layer_idx,
            total_layers=total_layers,
            dropout=dropout
        )
        
        # For efficient token gathering
        self.register_buffer('_dummy', torch.tensor(0.0))  # For device detection
    
    def gather_routed_tokens(
        self,
        x: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Efficiently gather tokens that should go through attention
        
        Args:
            x: (batch, seqlen, d_model)
            mask: (batch, seqlen) boolean mask
        
        Returns:
            gathered_tokens: (total_routed, d_model)
            indices: tuple of (batch_indices, seq_indices) for scattering back
        """
        # Get indices of True values in mask
        batch_indices, seq_indices = mask.nonzero(as_tuple=True)
        
        if len(batch_indices) == 0:
            # No tokens routed - return empty
            return (
                torch.zeros(0, self.d_model, device=x.device, dtype=x.dtype),
                (batch_indices, seq_indices)
            )
        
        # Gather tokens: this is efficient as it's a single indexing operation
        gathered = x[batch_indices, seq_indices]
        
        return gathered, (batch_indices, seq_indices)
    
    def scatter_routed_tokens(
        self,
        attn_output: torch.Tensor,
        indices: Tuple[torch.Tensor, torch.Tensor],
        original_shape: Tuple[int, int, int]
    ) -> torch.Tensor:
        """
        Scatter attention outputs back to original positions
        
        Args:
            attn_output: (total_routed, d_model)
            indices: tuple of (batch_indices, seq_indices)
            original_shape: (batch, seqlen, d_model)
        
        Returns:
            scattered: (batch, seqlen, d_model) with zeros except at routed positions
        """
        batch, seqlen, d_model = original_shape
        batch_indices, seq_indices = indices
        
        # Create output tensor
        output = torch.zeros(batch, seqlen, d_model, device=attn_output.device, dtype=attn_output.dtype)
        
        if len(batch_indices) > 0:
            # Scatter back
            output[batch_indices, seq_indices] = attn_output
        
        return output
    
    def forward_with_routing(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with learned routing
        
        Args:
            x: (batch, seqlen, d_model)
            deterministic: if True, use deterministic routing
        
        Returns:
            output: (batch, seqlen, d_model)
            aux: auxiliary outputs for loss computation (contains routing stats - tensors and floats)
        """
        batch, seqlen, d_model = x.shape
        
        # Get routing decisions
        routing_mask, router_aux = self.router(x, deterministic=deterministic)
        
        # All tokens through SSM (baseline processing)
        ssm_output = self.ssm_block(x)
        
        # Apply gradient balancing to SSM path if enabled
        if self.use_gradient_balancing and self.training:
            # Scale SSM gradients inversely to usage
            ssm_scale = 1.0 / (1.0 - self.target_ratio)
            ssm_output = GradientScaler.apply(ssm_output, ssm_scale)
        
        # Selected tokens through attention (enhanced processing)
        gathered_tokens, indices = self.gather_routed_tokens(x, routing_mask)
        
        if gathered_tokens.shape[0] > 0:
            # Process gathered tokens through attention
            # Note: AttentionBlock expects (batch, seqlen, d_model)
            # We need to handle variable-length sequences
            
            # Reshape to (1, num_tokens, d_model) for attention processing
            attn_input = gathered_tokens.unsqueeze(0)
            attn_output = self.attn_block(attn_input)
            attn_output = attn_output.squeeze(0)
            
            # Apply gradient balancing to attention path
            if self.use_gradient_balancing and self.training:
                attn_scale = 1.0 / self.target_ratio
                attn_output = GradientScaler.apply(attn_output, attn_scale)
            
            # Scatter back to original positions
            attn_scattered = self.scatter_routed_tokens(
                attn_output,
                indices,
                (batch, seqlen, d_model)
            )
        else:
            # No tokens routed to attention
            attn_scattered = torch.zeros_like(ssm_output)
        
        # Combine: SSM for all, attention additive for selected
        combined = ssm_output + attn_scattered
        
        # FFN for all tokens
        output = self.ffn(combined)
        
        # Prepare auxiliary outputs
        aux = {
            'router_probs': router_aux['router_probs'],
            'router_logits': router_aux['router_logits'],
            'routing_mask': routing_mask,
            'num_routed': routing_mask.sum().item(),
            'routing_ratio': routing_mask.float().mean().item()
        }
        
        return output, aux
    
    def forward(
        self,
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Main forward pass with optional activation checkpointing
        """
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.forward_with_routing,
                x,
                deterministic,
                use_reentrant=False
            )
        else:
            return self.forward_with_routing(x, deterministic)


class RoutedHybridModel(nn.Module):
    """
    Complete HYDRA model with learned token routing
    
    Features:
    - Efficient routed layers
    - Comprehensive monitoring
    - Gradient balancing
    - Position invariance
    - Layer diversity
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        router_hidden_dim: int = 64,
        target_ratio: float = 0.15,
        use_gradient_balancing: bool = True,
        use_position_invariance: bool = True,
        use_checkpoint: bool = False,
        tie_weights: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.target_ratio = target_ratio
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings (learnable)
        self.pos_embedding = nn.Parameter(torch.zeros(1, 8192, d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Routed hybrid layers
        self.layers = nn.ModuleList([
            RoutedHybridLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                d_ff=d_ff,
                dropout=dropout,
                router_hidden_dim=router_hidden_dim,
                target_ratio=target_ratio,
                layer_idx=i,
                total_layers=n_layers,
                use_gradient_balancing=use_gradient_balancing,
                use_position_invariance=use_position_invariance,
                use_checkpoint=use_checkpoint
            )
            for i in range(n_layers)
        ])
        
        # Output layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # LM head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Optionally tie weights
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_router_outputs: bool = False
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[List[Dict[str, Any]]]]:
        """
        Forward pass
        
        Args:
            input_ids: (batch, seqlen)
            labels: (batch, seqlen) optional, for loss computation
            deterministic: if True, use deterministic routing
            return_router_outputs: if True, return routing info
        
        Returns:
            loss: scalar or None
            logits: (batch, seqlen, vocab_size)
            router_outputs: list of aux dicts (with mixed types) if return_router_outputs=True, else None
        """
        batch, seqlen = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)
        
        # Add positional embeddings
        pos_emb = self.pos_embedding[:, :seqlen, :]
        x = x + pos_emb
        
        x = self.dropout(x)
        
        # Process through layers
        router_outputs = []
        for layer in self.layers:
            x, aux = layer(x, deterministic=deterministic)
            if return_router_outputs:
                router_outputs.append(aux)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Compute logits
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for cross entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        if return_router_outputs:
            return loss, logits, router_outputs
        else:
            return loss, logits, None
    
    def get_num_params(self, non_embedding: bool = True):
        """
        Return the number of parameters in the model.
        For non_embedding count, subtract the position and token embeddings.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.pos_embedding.numel()
        return n_params
    
    def get_routing_stats(self) -> Dict[str, float]:
        """Get statistics about current routing behavior"""
        stats = {}
        for i, layer in enumerate(self.layers):
            router = layer.router
            if hasattr(router, 'step_count'):
                stats[f'layer_{i}/steps'] = router.step_count.item()
                stats[f'layer_{i}/threshold'] = torch.sigmoid(router.threshold).item()
        return stats
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        deterministic_routing: bool = True
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively
        
        Args:
            idx: (batch, seqlen) initial sequence
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature
            top_k: if set, only sample from top k logits
            deterministic_routing: use deterministic routing during generation
        
        Returns:
            (batch, seqlen + max_new_tokens) generated sequence
        """
        for _ in range(max_new_tokens):
            # Crop context if needed
            idx_cond = idx if idx.size(1) <= 8192 else idx[:, -8192:]
            
            # Forward pass
            _, logits, _ = self(idx_cond, deterministic=deterministic_routing)
            
            # Focus on last token
            logits = logits[:, -1, :] / temperature
            
            # Optional top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx