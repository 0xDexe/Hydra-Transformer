"""
Query-Aware Routing for HYDRA

This module implements question-conditioned token routing for QA tasks.
The key insight: route tokens based on relevance to the question, not just content.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from src.model.router import EfficientTokenRouter


class QueryAwareRouter(nn.Module):
    """
    Routes tokens based on both content and relevance to a query/question.
    
    Args:
        d_model: Model dimension
        hidden_dim: Router hidden dimension
        target_ratio: Target routing ratio
        use_content_routing: If True, combine with content-based routing
        query_weight: Weight for query-relevance vs content (learnable)
    
    Example:
        Context: "Harry Potter lived with the Dursleys. Ron was his best friend."
        Question: "Who is Harry's best friend?"
        
        Content scores:     [0.3, 0.4, 0.2, 0.1, 0.5, 0.3, 0.8, 0.6]
        Relevance scores:   [0.1, 0.2, 0.1, 0.0, 0.9, 0.2, 0.8, 0.9]
                                                    ↑        ↑    ↑
                                                  "Ron"  "best" "friend"
        
        Combined → Routes "Ron", "best", "friend" to attention
    """
    
    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 64,
        target_ratio: float = 0.15,
        use_content_routing: bool = True,
        n_heads: int = 4,
        layer_idx: int = 0,
        total_layers: int = 12,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.target_ratio = target_ratio
        self.use_content_routing = use_content_routing
        self.layer_idx = layer_idx
        
        # Content-based router (from existing implementation)
        if use_content_routing:
            self.content_router = EfficientTokenRouter(
                d_model=d_model,
                hidden_dim=hidden_dim,
                target_ratio=target_ratio,
                layer_idx=layer_idx,
                total_layers=total_layers
            )
        
        # Query-relevance scorer
        # Uses cross-attention to compute token-query relevance
        self.query_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Learnable weight for combining content vs relevance
        # Start at 0.5 (equal weight), let it learn optimal balance
        self.query_weight = nn.Parameter(torch.tensor(0.5))
        
        # Learnable threshold (same as EfficientTokenRouter)
        init_threshold = torch.log(torch.tensor(target_ratio / (1 - target_ratio)))
        self.threshold = nn.Parameter(init_threshold)
        
        # Layer-specific bias (encourage different routing per layer)
        layer_bias = (layer_idx / max(total_layers - 1, 1) - 0.5) * 0.2
        self.layer_bias = nn.Parameter(torch.tensor(layer_bias))
    
    def extract_question_embedding(
        self,
        input_ids: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract question representation from input.
        
        For QA datasets with format: "Context: ... Question: ... Answer: ..."
        We want to extract the question part.
        
        Args:
            input_ids: (batch, seqlen, d_model) token embeddings
            question_mask: (batch, seqlen) boolean mask for question tokens
        
        Returns:
            question_emb: (batch, d_model) question representation
        """
        if question_mask is not None:
            # User provided explicit question mask
            # Take mean of question tokens
            question_tokens = input_ids * question_mask.unsqueeze(-1)
            question_emb = question_tokens.sum(dim=1) / (question_mask.sum(dim=1, keepdim=True) + 1e-8)
        else:
            # Heuristic: Use attention pooling over all tokens
            # This lets the model learn which parts are "question-like"
            batch, seqlen, d_model = input_ids.shape
            
            # Self-attention pooling
            query = input_ids.mean(dim=1, keepdim=True)  # (batch, 1, d_model)
            scores = torch.bmm(query, input_ids.transpose(1, 2))  # (batch, 1, seqlen)
            weights = F.softmax(scores, dim=-1)
            question_emb = torch.bmm(weights, input_ids).squeeze(1)  # (batch, d_model)
        
        return question_emb
    
    def compute_query_relevance(
        self,
        tokens: torch.Tensor,
        query: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute relevance of each token to the query using cross-attention.
        
        Args:
            tokens: (batch, seqlen, d_model) context tokens
            query: (batch, d_model) query representation
        
        Returns:
            relevance: (batch, seqlen) relevance scores [0, 1]
        """
        batch, seqlen, d_model = tokens.shape
        
        # Expand query for cross-attention
        # query: (batch, 1, d_model)
        query_expanded = query.unsqueeze(1)
        
        # Cross-attend: tokens attend to query
        # attn_output: (batch, seqlen, d_model)
        # attn_weights: (batch, seqlen, 1)
        attn_output, attn_weights = self.query_attention(
            tokens,           # query in attention terminology
            query_expanded,   # key
            query_expanded,   # value
            need_weights=True,
            average_attn_weights=True
        )
        
        # Relevance = attention weights (already normalized)
        # attn_weights: (batch, n_heads, seqlen, 1)
        # We want: (batch, seqlen)
        relevance = attn_weights.squeeze(-1).mean(1)  # Average over heads
        
        return relevance
    
    def forward(
        self,
        x: torch.Tensor,
        question_embedding: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Route tokens based on content and query relevance.
        
        Args:
            x: (batch, seqlen, d_model) token embeddings
            question_embedding: (batch, d_model) optional pre-computed question embedding
            question_mask: (batch, seqlen) optional mask indicating question tokens
            deterministic: if True, use deterministic routing
        
        Returns:
            routing_mask: (batch, seqlen) boolean mask
            aux: auxiliary outputs for monitoring/loss
        """
        batch, seqlen, d_model = x.shape
        
        # Extract question representation if not provided
        if question_embedding is None:
            question_embedding = self.extract_question_embedding(x, question_mask)
        
        # Compute query relevance
        relevance_scores = self.compute_query_relevance(x, question_embedding)
        
        # Compute content scores
        if self.use_content_routing:
            content_mask, content_aux = self.content_router(x, deterministic=False)
            content_scores = content_aux['router_probs']  # (batch, seqlen)
        else:
            content_scores = torch.zeros_like(relevance_scores)
            content_aux = {}
        
        # Combine content and relevance scores
        # Use learnable weight (clamped to [0, 1])
        alpha = torch.sigmoid(self.query_weight)
        combined_scores = (
            (1 - alpha) * content_scores + 
            alpha * relevance_scores
        )
        
        # Apply threshold with layer bias
        threshold_value = torch.sigmoid(self.threshold + self.layer_bias)
        
        if deterministic:
            # Deterministic: hard threshold
            routing_mask = combined_scores > threshold_value
        else:
            # Stochastic: Bernoulli sampling
            routing_probs = torch.sigmoid(
                (combined_scores - threshold_value) * 10.0  # Temperature
            )
            routing_mask = torch.bernoulli(routing_probs).bool()
        
        # Prepare auxiliary outputs
        aux = {
            'router_probs': combined_scores,
            'content_scores': content_scores,
            'relevance_scores': relevance_scores,
            'query_weight': alpha.item(),
            'routing_ratio': routing_mask.float().mean().item(),
            **content_aux
        }
        
        return routing_mask, aux
    
    def get_routing_stats(self, aux: Dict) -> Dict[str, float]:
        """Extract statistics for logging"""
        return {
            'routing_ratio': aux['routing_ratio'],
            'query_weight': aux['query_weight'],
            'avg_content_score': aux['content_scores'].mean().item(),
            'avg_relevance_score': aux['relevance_scores'].mean().item(),
        }


# TODO: Integration into RoutedHybridModel
# 
# In routed_hybrid_model.py:
# 
# class RoutedHybridLayer:
#     def __init__(self, ..., use_query_routing=False):
#         if use_query_routing:
#             self.router = QueryAwareRouter(d_model, ...)
#         else:
#             self.router = EfficientTokenRouter(d_model, ...)
# 
#     def forward(self, x, question_embedding=None):
#         if isinstance(self.router, QueryAwareRouter):
#             mask, aux = self.router(x, question_embedding=question_embedding)
#         else:
#             mask, aux = self.router(x)
#         ...


if __name__ == '__main__':
    # Test
    print("Testing QueryAwareRouter...")
    
    batch, seqlen, d_model = 2, 128, 768
    
    router = QueryAwareRouter(
        d_model=d_model,
        hidden_dim=64,
        target_ratio=0.15
    )
    
    # Simulate tokens
    tokens = torch.randn(batch, seqlen, d_model)
    
    # Simulate question (last 20 tokens)
    question_mask = torch.zeros(batch, seqlen, dtype=torch.bool)
    question_mask[:, -20:] = True
    
    # Forward
    mask, aux = router(tokens, question_mask=question_mask, deterministic=True)
    
    print(f"Input shape: {tokens.shape}")
    print(f"Routing mask shape: {mask.shape}")
    print(f"Routing ratio: {mask.float().mean().item():.3f}")
    print(f"Query weight: {aux['query_weight']:.3f}")
    print(f"Avg content score: {aux['avg_content_score']:.3f}")
    print(f"Avg relevance score: {aux['avg_relevance_score']:.3f}")
    print("\n✓ QueryAwareRouter test passed!")