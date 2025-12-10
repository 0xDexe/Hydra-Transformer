import torch
import torch.nn as nn
import math
from typing import Optional

try:
    from flash_attn import flash_attn_func
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash_attn not available, using fallback implementation")


class SlidingWindowAttentionBlock(nn.Module):
    """
    Sliding Window Attention block with FlashAttention support and dtype compatibility.
    
    This implementation handles the dtype mismatch between PyTorch's default fp32
    and FlashAttention's requirement for fp16/bf16. It provides automatic conversion
    and fallback mechanisms.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_size: Size of the sliding window (tokens on each side)
        dropout: Dropout probability
        use_flash: Whether to use FlashAttention (auto-detected if available)
        dtype: Target dtype for computation ('fp16', 'bf16', or 'fp32')
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_size: int = 256,
        dropout: float = 0.1,
        use_flash: bool = True,
        dtype: str = 'bf16'
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.dropout = dropout
        
        # Determine if we can use FlashAttention
        self.use_flash = use_flash and FLASH_ATTN_AVAILABLE
        
        # Set target dtype for computation
        self.target_dtype = self._resolve_dtype(dtype)
        
        # Q, K, V projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Scale factor
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def _resolve_dtype(self, dtype: str) -> torch.dtype:
        """Resolve dtype string to torch dtype"""
        dtype_map = {
            'fp16': torch.float16,
            'float16': torch.float16,
            'bf16': torch.bfloat16,
            'bfloat16': torch.bfloat16,
            'fp32': torch.float32,
            'float32': torch.float32,
        }
        
        if dtype not in dtype_map:
            print(f"Warning: Unknown dtype '{dtype}', defaulting to bfloat16")
            return torch.bfloat16
            
        resolved = dtype_map[dtype]
        
        # Check if bf16 is available
        if resolved == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print("Warning: bfloat16 not supported on this device, using float16")
            resolved = torch.float16
            
        return resolved
    
    def _create_sliding_window_mask(
        self,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Create a causal sliding window attention mask.
        
        Returns a mask where:
        - True/1 = positions that CAN be attended to
        - False/0 = positions that CANNOT be attended to
        """
        # Create causal mask (lower triangular)
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        )
        
        # Create sliding window mask
        if self.window_size > 0:
            # Create a band matrix for the window
            row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
            col_idx = torch.arange(seq_len, device=device).unsqueeze(0)
            
            # Token i can attend to tokens in range [max(0, i-window_size), i]
            window_mask = (col_idx >= row_idx - self.window_size) & (col_idx <= row_idx)
            
            # Combine causal and window masks
            mask = causal_mask & window_mask
        else:
            # No window restriction, just causal
            mask = causal_mask
        
        return mask
    
    def _flash_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        original_dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Forward pass using FlashAttention with dtype conversion.
        
        Args:
            q, k, v: Query, Key, Value tensors in (batch, seqlen, n_heads, head_dim)
            original_dtype: Original dtype of input for conversion back
            
        Returns:
            attention output in original dtype
        """
        batch, seqlen, n_heads, head_dim = q.shape
        
        # Convert to target dtype if needed
        if q.dtype != self.target_dtype:
            q = q.to(self.target_dtype)
            k = k.to(self.target_dtype)
            v = v.to(self.target_dtype)
        
        try:
            # FlashAttention with sliding window
            # window_size parameter: (left, right) - for causal, right=0
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                window_size=(self.window_size, 0) if self.window_size > 0 else (-1, -1)
            )
            
            # Convert back to original dtype
            if attn_output.dtype != original_dtype:
                attn_output = attn_output.to(original_dtype)
                
            return attn_output
            
        except Exception as e:
            print(f"FlashAttention failed: {e}, falling back to manual implementation")
            # Fall back to manual implementation
            return self._manual_attention_forward(q, k, v, original_dtype)
    
    def _manual_attention_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        original_dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Manual attention implementation with sliding window.
        
        This is used as a fallback when FlashAttention is not available
        or fails, and supports fp32 computation.
        """
        batch, seqlen, n_heads, head_dim = q.shape
        
        # Ensure we're in fp32 for numerical stability in fallback
        if q.dtype != torch.float32:
            q = q.to(torch.float32)
            k = k.to(torch.float32)
            v = v.to(torch.float32)
        
        # Transpose for batch matrix multiply: (batch, n_heads, seqlen, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        # Shape: (batch, n_heads, seqlen, seqlen)
        
        # Create and apply sliding window mask
        mask = self._create_sliding_window_mask(seqlen, q.device, torch.bool)
        # Expand mask for batch and heads
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seqlen, seqlen)
        
        # Apply mask (set masked positions to -inf)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply dropout
        if self.training and self.dropout > 0:
            attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        # Shape: (batch, n_heads, seqlen, head_dim)
        
        # Transpose back: (batch, seqlen, n_heads, head_dim)
        attn_output = attn_output.transpose(1, 2)
        
        # Convert back to original dtype
        if attn_output.dtype != original_dtype:
            attn_output = attn_output.to(original_dtype)
        
        return attn_output
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic dtype handling.
        
        Args:
            x: Input tensor of shape (batch, seqlen, d_model)
            
        Returns:
            Output tensor of shape (batch, seqlen, d_model)
        """
        batch, seqlen, d_model = x.shape
        original_dtype = x.dtype
        residual = x
        
        # Pre-norm
        x = self.norm(x)
        
        # QKV projection
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch, seqlen, 3, self.n_heads, self.head_dim)
        
        # Split into Q, K, V
        q, k, v = qkv.unbind(dim=2)
        # Each: (batch, seqlen, n_heads, head_dim)
        
        # Apply attention (with appropriate dtype handling)
        if self.use_flash:
            attn_output = self._flash_attention_forward(q, k, v, original_dtype)
        else:
            attn_output = self._manual_attention_forward(q, k, v, original_dtype)
        
        # Reshape and project
        attn_output = attn_output.reshape(batch, seqlen, d_model)
        output = self.out_proj(attn_output)
        
        # Residual connection
        return output + residual
    
    def get_effective_context(self, position: int, seq_len: int) -> int:
        """
        Get the effective context length at a given position.
        
        Args:
            position: Current position in sequence (0-indexed)
            seq_len: Total sequence length
            
        Returns:
            Number of tokens this position can attend to
        """
        if self.window_size <= 0:
            # Full causal attention
            return position + 1
        else:
            # Sliding window: can see window_size tokens back + self
            return min(self.window_size + 1, position + 1)


class MultiWindowAttentionBlock(nn.Module):
    """
    Multi-scale sliding window attention with different window sizes.
    
    This allows the model to capture both local and semi-global patterns
    by using multiple attention heads with different window sizes.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        window_sizes: List of window sizes for different head groups
        dropout: Dropout probability
        use_flash: Whether to use FlashAttention
        dtype: Target dtype for computation
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_sizes: list = [128, 256, 512],
        dropout: float = 0.1,
        use_flash: bool = True,
        dtype: str = 'bf16'
    ):
        super().__init__()
        assert d_model % n_heads == 0
        assert len(window_sizes) <= n_heads
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_sizes = window_sizes
        
        # Create multiple SWA blocks with different window sizes
        # Distribute heads across windows
        heads_per_window = n_heads // len(window_sizes)
        remaining_heads = n_heads % len(window_sizes)
        
        self.swa_blocks = nn.ModuleList()
        self.head_groups = []
        
        current_head = 0
        for i, window_size in enumerate(window_sizes):
            # Allocate heads
            num_heads = heads_per_window + (1 if i < remaining_heads else 0)
            
            if num_heads > 0:
                block = SlidingWindowAttentionBlock(
                    d_model=d_model,
                    n_heads=num_heads,
                    window_size=window_size,
                    dropout=dropout,
                    use_flash=use_flash,
                    dtype=dtype
                )
                self.swa_blocks.append(block)
                self.head_groups.append((current_head, current_head + num_heads))
                current_head += num_heads
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining multiple window sizes.
        
        Args:
            x: Input tensor of shape (batch, seqlen, d_model)
            
        Returns:
            Output tensor of shape (batch, seqlen, d_model)
        """
        # Each block will handle its own residual connection
        # We just need to combine their outputs
        outputs = []
        for block in self.swa_blocks:
            output = block(x)
            outputs.append(output)
        
        # Average the outputs (you could also concatenate or use other strategies)
        combined = torch.stack(outputs).mean(dim=0)
        
        return combined


if __name__ == '__main__':
    # Test the implementation
    print("Testing SlidingWindowAttentionBlock...")
    
    # Test parameters
    batch_size = 2
    seq_len = 512
    d_model = 768
    n_heads = 12
    window_size = 256
    
    # Create model
    model = SlidingWindowAttentionBlock(
        d_model=d_model,
        n_heads=n_heads,
        window_size=window_size,
        dropout=0.1,
        use_flash=True,
        dtype='bf16'
    )
    
    # Test with different dtypes
    for test_dtype in [torch.float32, torch.float16, torch.bfloat16]:
        if test_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            print(f"Skipping {test_dtype} (not supported)")
            continue
            
        print(f"\nTesting with input dtype: {test_dtype}")
        
        # Create test input
        x = torch.randn(batch_size, seq_len, d_model, dtype=test_dtype)
        
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
        
        # Forward pass
        try:
            output = model(x)
            print(f" Input shape: {x.shape}, dtype: {x.dtype}")
            print(f" Output shape: {output.shape}, dtype: {output.dtype}")
            print(f" Output dtype matches input: {output.dtype == test_dtype}")
            
            # Check effective context
            for pos in [0, 128, 256, 511]:
                ctx = model.get_effective_context(pos, seq_len)
                print(f"  Position {pos}: effective context = {ctx}")
                
        except Exception as e:
            print(f" Failed with {test_dtype}: {e}")
    
    print("\n" + "="*60)
    print("Testing MultiWindowAttentionBlock...")
    
    multi_model = MultiWindowAttentionBlock(
        d_model=d_model,
        n_heads=n_heads,
        window_sizes=[128, 256, 512],
        dropout=0.1,
        use_flash=True,
        dtype='bf16'
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    if torch.cuda.is_available():
        multi_model = multi_model.cuda()
        x = x.cuda()
    
    output = multi_model(x)
    print(f"Multi-window output shape: {output.shape}")
    print("\nAll tests completed!")