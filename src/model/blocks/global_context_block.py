import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.blocks.ssm_core import SSMCore


class GlobalContextBlock(nn.Module):
    def __init__(
        self,
        d_model,
        chunk_size=64,
        ssm_expansion=2,
        kernel_size=5,
        dropout=0.1
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.global_ssm = SSMCore(
            d_model=d_model,
            d_state=ssm_expansion,
            d_conv=kernel_size,
            expand=ssm_expansion,
            dropout=dropout
        )

    def forward(self, x):
        B, T, D = x.shape
        cs = self.chunk_size

        pad_len = (cs - T % cs) % cs
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x

        T_pad = x_pad.size(1)
        C = T_pad // cs

        chunks = x_pad.view(B, C, cs, D)
        chunk_repr = chunks.mean(dim=2)

        g = self.global_ssm(chunk_repr)

        g_expand = (
            g.unsqueeze(2)
            .expand(B, C, cs, D)
            .reshape(B, T_pad, D)
        )
        return g_expand[:, :T, :]
