import torch
import torch.nn as nn

from src.model.blocks.ssm_block import SSMBlock
from src.model.blocks.sliding_window_attention import SlidingWindowAttention


class LightweightContextLayer(nn.Module):
    """Add local context to embeddings without full O(n²) attention."""
    
    def __init__(self, d_model, mode="conv", window_size=128):
        super().__init__()
        self.mode = mode
        
        if mode == "conv":
            self.layer = nn.Conv1d(
                d_model, d_model,
                kernel_size=7,
                padding=3,
                groups=1
            )
        elif mode == "ssm":
            self.layer = SSMBlock(d_model)
        elif mode == "local_attn":
            self.layer = SlidingWindowAttention(
                d_model=d_model,
                window_size=window_size
            )
        else:
            raise ValueError("mode must be one of: conv, ssm, local_attn")

    def forward(self, x):
        """
        x: (B, T, D)
        """
        if self.mode == "conv":
            # Conv1d expects (B, D, T)
            out = self.layer(x.transpose(1, 2)).transpose(1, 2)
            return out
        
        # SSMBlock or Local Attention already expect (B, T, D)
        return self.layer(x)
