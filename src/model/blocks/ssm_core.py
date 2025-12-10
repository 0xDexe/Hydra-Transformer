import torch
import torch.nn as nn


class MambaLikeSSM(nn.Module):
    def __init__(self, d_model: int, expansion: int = 2, kernel_size: int = 7, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model * expansion)
        self.gate_proj = nn.Linear(d_model, d_model * expansion)
        self.conv = nn.Conv1d(
            d_model * expansion,
            d_model * expansion,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model * expansion,
        )
        self.out_proj = nn.Linear(d_model * expansion, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        h = self.in_proj(x)              # (B, T, De)
        g = torch.sigmoid(self.gate_proj(x))
        h = h.transpose(1, 2)            # (B, De, T)
        h = self.conv(h).transpose(1, 2) # (B, T, De)
        h = h * g
        h = self.out_proj(h)
        return self.dropout(h)
