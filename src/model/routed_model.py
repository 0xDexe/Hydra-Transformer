import torch
import torch.nn as nn

from src.model.blocks.attention_core import AttentionCore
from src.model.blocks.ssm_core import SSMCore
from model.blocks.scoring.token_router_mlp import TokenRouter_MLP
from src.model.blocks.routing_block import RoutedHybridBlock
from src.model.blocks.ffn_block import FFNBlock
from src.model.blocks.lightweight_context_layer import LightweightContextLayer

class RoutedHybridSSMTransformer(nn.Module):
    """
    Hybrid language model with learned token routing between Attention and SSM.

    - Every block is a RoutedHybridBlock + FFNBlock.
    - A single shared TokenRouter decides per token whether to use
      attention or SSM in each block.
    - Designed as a drop-in LM for next-token prediction.

    Args:
        vocab_size: vocabulary size
        d_model: hidden dimension
        n_layers: number of hybrid blocks (each = RoutedHybridBlock + FFN)
        n_heads: attention heads
        d_state, d_conv, expand: Mamba hyperparameters
        dropout: embedding & attention dropout
        max_seq_len: maximum context length for position embeddings
        routing_topk_ratio: fraction of tokens routed to attention
        router_hidden_dim: hidden size in the router MLP (default = d_model)
    """
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
        routing_topk_ratio=0.2,
        router_hidden_dim=None, 
        context_mode="conv"
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.routing_topk_ratio = routing_topk_ratio

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )

        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(dropout)

        # NEW: lightweight contextualizer
        self.context_layer = LightweightContextLayer(
            d_model,
            mode=context_mode
        )

        # Shared token router (single routing mechanism)
        self.router = TokenRouter_MLP(d_model, hidden_dim=router_hidden_dim)

        # Build stack of RoutedHybridBlocks + FFNBlocks
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            block = RoutedHybridBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                topk_ratio=self.routing_topk_ratio,
                router=self.router,  # shared
            )
            self.layers.append(block)
            self.layers.append(FFNBlock(d_model, dropout=dropout))

        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights similar to GPT-style models."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def forward(self, input_ids, labels=None):
        """
        Args:
            input_ids: (batch, seqlen) token IDs
            labels: (batch, seqlen), optional, for next-token loss

        Returns:
            If labels is not None: (loss, logits)
            Else: logits
        """
        B, T = input_ids.shape

        # 1) Embeddings
        # x = self.token_embedding(input_ids)           # (B, T, D)
        # x = x + self.pos_embedding[:, :T, :]          # add positional encodings
        # x = self.embed_dropout(x)                     # (B, T, D)

        # 1) Embeddings
        x = self.token_embedding(input_ids)            # (B, T, D)
        x = x + self.pos_embedding[:, :T, :]           # (B, T, D)
        x = self.embed_dropout(x)

        # 2) Lightweight context layer 
        x = self.context_layer(x)                      # (B, T, D) enriched

        # 2) Stacked routed hybrid blocks + FFNs
        for layer in self.layers:
            x = layer(x)                              # always (B, T, D)

        # 3) Final norm
        x = self.final_norm(x)                        # (B, T, D)

        # 4) Project to vocabulary
        logits = self.lm_head(x)                      # (B, T, V)

        # 5) Optional next-token loss
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()   # (B, T-1, V)
            shift_labels = labels[..., 1:].contiguous()       # (B, T-1)

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            return loss, logits

        return logits

    def get_num_params(self, non_embedding=False):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.pos_embedding.numel()
        return n_params
