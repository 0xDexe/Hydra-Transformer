import torch
import torch.nn as nn

from .token_router_mlp import TokenRouterMLP
from .attention_core import AttentionCore
from .ssm_core import MambaLikeSSM
from .global_context_block import GlobalContextBlock


class RoutedHybridBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        k_top: int,
        chunk_size: int,
        attn_dropout: float = 0.0,
        ssm_expansion: int = 2,
        local_kernel: int = 7,
        global_kernel: int = 5,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.k_top = k_top

        self.norm_router = nn.LayerNorm(d_model)
        self.router = TokenRouterMLP(d_model)

        self.attn_core = AttentionCore(d_model, n_heads, dropout=attn_dropout)
        self.local_ssm = MambaLikeSSM(
            d_model=d_model,
            expansion=ssm_expansion,
            kernel_size=local_kernel,
            dropout=dropout,
        )

        self.global_context = GlobalContextBlock(
            d_model=d_model,
            chunk_size=chunk_size,
            ssm_expansion=ssm_expansion,
            kernel_size=global_kernel,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        B, T, D = x.shape
        residual = x

        x_norm = self.norm_router(x)
        scores = self.router(x_norm)            # (B, T)
        k = min(self.k_top, T)
        topk_idx = scores.topk(k, dim=1).indices  # (B, k)

        attn_mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        attn_mask.scatter_(1, topk_idx, True)  # True = use attention

        attn_mask_exp = attn_mask.unsqueeze(-1)  # (B, T, 1)

        attn_out = self.attn_core(x_norm)
        ssm_out = self.local_ssm(x_norm)

        local_mixed = torch.where(attn_mask_exp, attn_out, ssm_out)

        global_ctx = self.global_context(local_mixed)

        delta = local_mixed + global_ctx
        delta = self.dropout(delta)

        return residual + delta
