import torch
import torch.nn as nn

from src.model.blocks.attention_core import AttentionCore
from src.model.blocks.ssm_core import SSMCore
from src.model.blocks.token_router import TokenRouter

class RoutedHybridBlock(nn.Module):
    """
    Hybrid block with learned token routing between Attention and SSM.

    Pipeline:
        h_in
          ─ LN -> h_norm
               ─ router(h_norm) -> scores -> top-k mask
               ─ AttentionCore(h_norm) -> delta_attn
               ─ SSMCore(h_norm)      -> delta_ssm
        per-token: delta = attn or ssm
        h_out = h_in + delta

    Args:
        d_model: hidden size
        n_heads: attention heads
        d_state, d_conv, expand: Mamba SSM hyperparams
        dropout: attention dropout
        topk_ratio: fraction of tokens routed to attention (e.g., 0.2)
        router: shared TokenRouter instance
    """
    def __init__(
        self,
        d_model,
        n_heads,
        d_state,
        d_conv,
        expand,
        dropout=0.1,
        topk_ratio=0.2,
        router=None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

        self.attn_core = AttentionCore(d_model, n_heads=n_heads, dropout=dropout)
        self.ssm_core = SSMCore(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        self.topk_ratio = topk_ratio
        if router is None:
            # Local router (not shared); in the big model we pass a shared one.
            self.router = TokenRouter(d_model)
        else:
            self.router = router

    def forward(self, x):
        """
        x: (batch, seqlen, d_model)
        return: (batch, seqlen, d_model)
        """
        B, T, D = x.shape
        residual = x

        # 1) Pre-norm
        x_norm = self.norm(x)  # (B, T, D)

        # 2) Router scores per token
        scores = self.router(x_norm)  # (B, T)

        # 3) Compute top-k mask per batch element
        k = max(1, int(self.topk_ratio * T))  # at least 1 token
        k = min(k, T)                         # safety

        _, topk_idx = torch.topk(scores, k=k, dim=1)  # (B, k)

        # Boolean mask: True → use attention, False → use SSM
        attn_mask = torch.zeros(B, T, device=x.device, dtype=torch.bool)
        attn_mask.scatter_(1, topk_idx, True)  # mark top-k positions
        attn_mask = attn_mask.unsqueeze(-1)    # (B, T, 1) for broadcasting

        # 4) Compute both branches on normalized input
        delta_attn = self.attn_core(x_norm)  # (B, T, D)
        delta_ssm = self.ssm_core(x_norm)    # (B, T, D)

        # 5) Per-token selection between attention and SSM deltas
        delta = torch.where(attn_mask, delta_attn, delta_ssm)  # (B, T, D)

        # 6) Single residual connection
        out = residual + delta
        return out