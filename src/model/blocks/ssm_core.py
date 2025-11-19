import torch
import torch.nn as nn
from mamba_ssm import Mamba

class SSMCore(nn.Module):
    """
    Core Mamba SSM.
    No LayerNorm, no residual. Assumes pre-normalized input.

    This is the low-level building block used inside RoutedHybridBlock,
    analogous to AttentionCore.

    Input:
        x: (batch, seqlen, d_model)
    Output:
        delta: (batch, seqlen, d_model)
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

    def forward(self, x):
        """
        Args:
            x: (B, T, D) - assumed already normalized by caller
        Returns:
            delta: (B, T, D)
        """
        # Direct SSM transform, no residual, no LayerNorm
        delta = self.ssm(x)
        return delta
