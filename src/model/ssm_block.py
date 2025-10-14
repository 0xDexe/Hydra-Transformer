import torch
import torch.nn as nn
from mamba_ssm import Mamba

class SSMBlock(nn.Module):
    """
    Wrapper around Mamba SSM for our hybrid architecture
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        
        # Use Mamba's SSM implementation
        self.ssm = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seqlen, d_model)
        Returns:
            output: (batch, seqlen, d_model)
        """
        # Pre-norm + residual
        residual = x
        x = self.norm(x)
        x = self.ssm(x)
        return x + residual