import torch
import torch.nn as nn

class TokenRouter_MLP(nn.Module):
    """
    MLP that scores each token for routing.

    Input:
        x: (batch, seqlen, d_model)
    Output:
        scores: (batch, seqlen)
    """
    def __init__(self, d_model, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = d_model

        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)   # scalar per token
        )

    def forward(self, x):
        # x: (B, T, D)
        scores = self.net(x)          # (B, T, 1)
        scores = scores.squeeze(-1)   # (B, T)
        return scores