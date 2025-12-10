import torch
import torch.nn as nn

from src.model.blocks.scoring.token_router_mlp import TokenRouter_MLP
from src.model.blocks.attention_core import AttentionCore
from src.model.blocks.ssm_core import SSMCore
from src.model.blocks.hierarchical.global_context_block import GlobalContextBlock


class HierarchicalRoutedHybridBlock(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        router,
        topk_ratio=0.2,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        chunk_size=64,
        global_kernel=5,
    ):
        super().__init__()

        self.topk_ratio = topk_ratio
        self.router = router

        self.attn = AttentionCore(d_model, n_heads, dropout=dropout)

        self.local_ssm = SSMCore(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )

        self.global_context = GlobalContextBlock(
            d_model=d_model,
            chunk_size=chunk_size,
            ssm_expansion=expand,
            kernel_size=global_kernel,
            dropout=dropout,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        residual = x

        x_norm = self.norm(x)
        scores = self.router(x_norm)

        k = max(1, int(self.topk_ratio * T))
        idx = scores.topk(k, dim=1).indices

        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        mask.scatter_(1, idx, True)
        mask = mask.unsqueeze(-1)

        attn_out = self.attn(x_norm)
        ssm_out = self.local_ssm(x_norm)

        local_mix = torch.where(mask, attn_out, ssm_out)

        global_ctx = self.global_context(local_mix)

        delta = local_mix + global_ctx
        delta = self.dropout(delta)

        return residual + delta
