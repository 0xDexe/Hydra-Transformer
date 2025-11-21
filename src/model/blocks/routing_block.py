import torch
import torch.nn as nn

from src.model.blocks.attention_core import AttentionCore
from src.model.blocks.ssm_core import SSMCore
from model.blocks.scoring.token_router_mlp import TokenRouter_MLP


class RoutedHybridBlock(nn.Module):
    """
    Hybrid block with learned token routing between Attention and SSM.

    Now optimized:
    - AttentionCore is computed ONLY on the top-k selected tokens.
    - SSMCore is computed on all tokens.
    - Final delta is merged per token.

    Args:
        d_model: hidden size
        n_heads: attention heads
        d_state, d_conv, expand: SSM (Mamba) hyperparameters
        dropout: dropout for attention
        topk_ratio: fraction of tokens routed to attention
        router: shared TokenRouter module
        compute_full_ssm: bool, compute SSM for all tokens
        use_sparse_attention: bool, use sparse attention for selected tokens only
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
        compute_full_ssm=True,
        use_sparse_attention=True,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(d_model)

        # Cores
        self.attn_core = AttentionCore(d_model, n_heads=n_heads, dropout=dropout)
        self.ssm_core = SSMCore(d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        # Routing parameters
        self.topk_ratio = topk_ratio
        self.compute_full_ssm = compute_full_ssm
        self.use_sparse_attention = use_sparse_attention

        # Router network
        if router is None:
            self.router = TokenRouter_MLP(d_model)
        else:
            self.router = router

    def forward(self, x):
        """
        Inputs:
            x: (B, T, D)

        Returns:
            out: (B, T, D)
        """
        B, T, D = x.shape
        residual = x

        # 1) Pre-norm 
        x_norm = self.norm(x)   # (B, T, D)

        # 2) Router: token importance scores 
        scores = self.router(x_norm)  # (B, T)

        # 3) Top-k tokens per batch 
        k = max(1, int(self.topk_ratio * T))
        k = min(k, T)

        _, topk_idx = torch.topk(scores, k=k, dim=1)  # (B, k)

        # Boolean routing mask
        attn_mask = torch.zeros(B, T, device=x.device, dtype=torch.bool)
        attn_mask.scatter_(1, topk_idx, True)  # mark top-k positions
        attn_mask = attn_mask.unsqueeze(-1)    # (B, T, 1)

        # 4A) Compute SSM for ALL tokens
        delta_ssm = self.ssm_core(x_norm) if self.compute_full_ssm else None

        # 4B) Compute attention ONLY for selected tokens
        # Extract top-k tokens
        x_attn_tokens = torch.gather(
            x_norm, 
            dim=1, 
            index=topk_idx.unsqueeze(-1).expand(B, k, D)
        )   # (B, k, D)

        # Apply attention to the selected tokens
        delta_attn_tokens = self.attn_core(x_attn_tokens)   # (B, k, D)

        # Place attention deltas back to (B, T, D) shape
        delta_attn = torch.zeros_like(x_norm)  # (B, T, D)
        delta_attn.scatter_(
            1,
            topk_idx.unsqueeze(-1).expand(B, k, D),
            delta_attn_tokens
        )

        # 5) Merge deltas per token
        if delta_ssm is None:
            # No SSM case
            delta = delta_attn
        else:
            delta = torch.where(attn_mask, delta_attn, delta_ssm)

        # 6) Residual 
        out = residual + delta
        return out
