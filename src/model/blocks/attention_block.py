import torch
import torch.nn as nn
import math

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash_attn not available, using standard attention")

class AttentionBlock(nn.Module):
    """
    Multi-head attention block with FlashAttention support
    
    Automatically handles dtype conversion for FlashAttention (requires fp16/bf16)
    Falls back to standard attention if FlashAttention is unavailable.
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1, use_flash_attn=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN
        
        # Q, K, V projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Layer norm (pre-norm)
        self.norm = nn.LayerNorm(d_model)
        
        self.dropout = dropout
        
        if not self.use_flash_attn:
            print(f"AttentionBlock: Using standard attention (FlashAttention not available)")
        
    def forward(self, x):
        """
        Args:
            x: (batch, seqlen, d_model) - can be fp32, fp16, or bf16
        Returns:
            output: (batch, seqlen, d_model) - same dtype as input
        """
        batch, seqlen, d_model = x.shape
        input_dtype = x.dtype
        residual = x
        
        # Pre-norm
        x = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seqlen, 3, self.n_heads, self.head_dim)
        
        # Rearrange for attention
        q, k, v = qkv.unbind(dim=2)  # Each: (batch, seqlen, n_heads, head_dim)
        
        if self.use_flash_attn:
            # FlashAttention requires fp16 or bf16
            # Convert if necessary
            original_dtype = q.dtype
            
            if original_dtype == torch.float32:
                # Convert to bf16 if available, otherwise fp16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    target_dtype = torch.bfloat16
                else:
                    target_dtype = torch.float16
                
                q = q.to(target_dtype)
                k = k.to(target_dtype)
                v = v.to(target_dtype)
            
            # Apply flash attention
            try:
                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=True,  # For autoregressive LM
                )
                
                # Convert back to original dtype if needed
                if original_dtype == torch.float32:
                    attn_output = attn_output.to(original_dtype)
                    
            except Exception as e:
                print(f"FlashAttention failed: {e}, falling back to standard attention")
                # Fall back to standard attention
                attn_output = self._standard_attention(q, k, v)
        else:
            # Use standard attention
            attn_output = self._standard_attention(q, k, v)
        
        # Reshape and project
        attn_output = attn_output.reshape(batch, seqlen, d_model)
        output = self.out_proj(attn_output)
        
        # Residual connection
        return output + residual
    
    def _standard_attention(self, q, k, v):
        """
        Standard scaled dot-product attention with causal mask
        
        Args:
            q, k, v: (batch, seqlen, n_heads, head_dim)
        Returns:
            output: (batch, seqlen, n_heads, head_dim)
        """
        batch, seqlen, n_heads, head_dim = q.shape
        
        # Transpose for attention: (batch, n_heads, seqlen, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(head_dim)
        
        # Compute attention scores: (batch, n_heads, seqlen, seqlen)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        if self.training and self.dropout > 0:
            attn_weights = torch.dropout(attn_weights, self.dropout, train=True)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Transpose back: (batch, seqlen, n_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output