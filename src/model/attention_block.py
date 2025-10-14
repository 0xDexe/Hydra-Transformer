import torch
import torch.nn as nn
import math
from flash_attn import flash_attn_func

class AttentionBlock(nn.Module):
    """
    Multi-head attention block with FlashAttention
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Q, K, V projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        self.dropout = dropout
        
    def forward(self, x):
        """
        Args:
            x: (batch, seqlen, d_model)
        Returns:
            output: (batch, seqlen, d_model)
        """
        batch, seqlen, d_model = x.shape
        residual = x
        
        # Pre-norm
        x = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seqlen, 3, self.n_heads, self.head_dim)
        
        # Rearrange for flash attention: (batch, seqlen, 3, n_heads, head_dim)
        q, k, v = qkv.unbind(dim=2)  # Each: (batch, seqlen, n_heads, head_dim)
        
        # FlashAttention expects (batch, seqlen, n_heads, head_dim)
        # Apply flash attention
        attn_output = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,  # For autoregressive LM
        )
        
        # Reshape and project
        attn_output = attn_output.reshape(batch, seqlen, d_model)
        output = self.out_proj(attn_output)
        
        return output + residual