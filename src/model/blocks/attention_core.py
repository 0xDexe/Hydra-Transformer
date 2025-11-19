import torch
import torch.nn as nn
import math

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
    print("Warning: flash_attn not available — AttentionCore will use standard attention")

class AttentionCore(nn.Module):
    """
    Core multi-head self-attention module with FlashAttention support.
    
    Differences vs full AttentionBlock:
      - No LayerNorm (assumes pre-norm provided by caller)
      - No residual connection
      - Returns only the delta update: (B, T, D)
    """

    def __init__(self, d_model, n_heads=8, dropout=0.1, use_flash_attn=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.use_flash_attn = use_flash_attn and HAS_FLASH_ATTN
        self.dropout = dropout
        
        # Q, K, V projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        if not self.use_flash_attn:
            print("AttentionCore: Using standard attention (FlashAttention unavailable)")

    def forward(self, x):
        """
        Args:
            x: (B, T, D) pre-normalized hidden states
        Returns:
            delta: (B, T, D)
        """
        B, T, D = x.shape
        input_dtype = x.dtype

        # ---- QKV Projection ----
        qkv = self.qkv(x)                               # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                     # Each: (B, T, H, Hd)

        # =====================================================
        # FlashAttention path
        # =====================================================
        if self.use_flash_attn:
            original_dtype = q.dtype

            # FlashAttention requires fp16 or bf16
            if original_dtype == torch.float32:
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    target_dtype = torch.bfloat16
                else:
                    target_dtype = torch.float16

                q = q.to(target_dtype)
                k = k.to(target_dtype)
                v = v.to(target_dtype)

            try:
                attn_output = flash_attn_func(
                    q, k, v,
                    dropout_p=self.dropout if self.training else 0.0,
                    causal=True,
                )

                # convert back
                if original_dtype == torch.float32:
                    attn_output = attn_output.to(original_dtype)

            except Exception as e:
                print(f"FlashAttention failed: {e} — falling back to standard attention")
                attn_output = self._standard_attention(q, k, v)

        # =====================================================
        # Standard attention path
        # =====================================================
        else:
            attn_output = self._standard_attention(q, k, v)

        # Merge heads: (B, T, H, Hd) → (B, T, D)
        attn_output = attn_output.reshape(B, T, D)

        # Output projection → delta update
        delta = self.out_proj(attn_output)
        return delta

    # ---------------------------------------------------------
    # Standard attention implementation (causal)
    # ---------------------------------------------------------
    def _standard_attention(self, q, k, v):
        """
        Args:
            q, k, v: (B, T, H, Hd)
        Returns:
            out: (B, T, H, Hd)
        """
        B, T, H, Hd = q.shape

        # (B, H, T, Hd)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(Hd)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T)

        # causal mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=q.device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

        # softmax
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # dropout
        if self.training and self.dropout > 0:
            attn_weights = torch.dropout(attn_weights, self.dropout, train=True)

        # attention output
        out = torch.matmul(attn_weights, v)  # (B, H, T, Hd)

        # back to (B, T, H, Hd)
        out = out.transpose(1, 2)
        return out
