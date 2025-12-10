import torch
import torch.nn as nn

from src.model.blocks.scoring.token_router_mlp import TokenRouter_MLP
from src.model.blocks.hierarchical.hierarchical_routing_block import HierarchicalRoutedHybridBlock
from src.model.blocks.ffn_block import FFNBlock
from src.model.blocks.lightweight_context_layer import LightweightContextLayer


class HierarchicalHybridTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads=8,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        max_seq_len=8192,
        topk_ratio=0.2,
        router_hidden_dim=None,
        context_mode="conv",
        chunk_size=64,
        global_kernel=5,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        self.emb_drop = nn.Dropout(dropout)

        self.context_layer = LightweightContextLayer(d_model, mode=context_mode)

        self.router = TokenRouter_MLP(d_model, router_hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(
                HierarchicalRoutedHybridBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    router=self.router,
                    topk_ratio=topk_ratio,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    chunk_size=chunk_size,
                    global_kernel=global_kernel,
                )
            )
            self.layers.append(FFNBlock(d_model, dropout=dropout))

        self.final_norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape

        x = self.token_embedding(input_ids)
        x = x + self.pos_embedding[:, :T, :]
        x = self.emb_drop(x)

        x = self.context_layer(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        if labels is None:
            return logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.reshape(-1, self.vocab_size),
            shift_labels.reshape(-1),
            ignore_index=-100,
        )
        return loss, logits
