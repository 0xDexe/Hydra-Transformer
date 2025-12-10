"""
Query-Aware HYDRA Model

Complete implementation that properly uses question extraction.
Routes tokens based on content AND question relevance.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from src.model.routed_model import RoutedHybridModel
from src.model.query_aware_router import QueryAwareRouter
from src.model.question_extractor import QuestionExtractor


class QueryAwareHybridModel(nn.Module):
    """
    HYDRA with query-aware routing for QA tasks.
    
    Key difference from base model:
    - Extracts question representation from input
    - Routes tokens based on relevance to question
    - Uses QuestionExtractor to identify question spans
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        d_ff: int = 3072,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        target_ratio: float = 0.15,
        router_hidden_dim: int = 64,
        use_gradient_balancing: bool = True,
        use_position_invariance: bool = True,
        tie_weights: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.target_ratio = target_ratio
        
        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Query-aware routers (one per layer)
        self.routers = nn.ModuleList([
            QueryAwareRouter(
                d_model=d_model,
                hidden_dim=router_hidden_dim,
                target_ratio=target_ratio,
                use_content_routing=True,
                n_heads=4,
                layer_idx=i,
                total_layers=n_layers,
            )
            for i in range(n_layers)
        ])
        
        # Transformer layers (SSM + Attention + FFN)
        # Import from your existing implementation
        from src.model.routed_hybrid_model import RoutedHybridLayer
        
        self.layers = nn.ModuleList([
            RoutedHybridLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                d_ff=d_ff,
                dropout=dropout,
                use_gradient_balancing=use_gradient_balancing,
                layer_idx=i,
            )
            for i in range(n_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        if tie_weights:
            self.lm_head.weight = self.embedding.weight
        
        print(f"✓ QueryAwareHybridModel created")
        print(f"  Layers: {n_layers}")
        print(f"  d_model: {d_model}")
        print(f"  Query-aware routing: ENABLED")
    
    def extract_question_embedding(
        self,
        x: torch.Tensor,
        question_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract question representation from embeddings.
        
        Args:
            x: (batch, seqlen, d_model) token embeddings
            question_mask: (batch, seqlen) boolean mask for question tokens
        
        Returns:
            question_emb: (batch, d_model)
        """
        if question_mask is not None and question_mask.any():
            # Use provided question mask
            # Average pool over question tokens
            question_tokens = x * question_mask.unsqueeze(-1).float()
            question_sum = question_tokens.sum(dim=1)
            question_count = question_mask.sum(dim=1, keepdim=True).float() + 1e-8
            question_emb = question_sum / question_count
        else:
            # Heuristic: Use last 20% of sequence as "question region"
            # (Questions often appear at end in QA datasets)
            batch, seqlen, d_model = x.shape
            question_start = max(1, int(seqlen * 0.8))
            question_region = x[:, question_start:, :]
            question_emb = question_region.mean(dim=1)
        
        return question_emb
    
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        question_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        return_router_outputs: bool = False
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[List[Dict]]]:
        """
        Forward pass with query-aware routing.
        
        Args:
            input_ids: (batch, seqlen) token IDs
            labels: (batch, seqlen) labels for loss computation
            question_mask: (batch, seqlen) optional mask indicating question tokens
            deterministic: if True, use deterministic routing
            return_router_outputs: if True, return router auxiliary outputs
        
        Returns:
            loss: scalar loss (if labels provided)
            logits: (batch, seqlen, vocab_size)
            router_outputs: list of router aux dicts (if requested)
        """
        batch, seqlen = input_ids.shape
        
        # Embed tokens
        x = self.embedding(input_ids)  # (batch, seqlen, d_model)
        
        # Add positional embeddings
        positions = torch.arange(seqlen, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        # Extract question representation
        question_emb = self.extract_question_embedding(x, question_mask)
        
        # Process through layers with query-aware routing
        router_outputs = [] if return_router_outputs else None
        
        for i, (router, layer) in enumerate(zip(self.routers, self.layers)):
            # Get routing mask using query-aware router
            routing_mask, router_aux = router(
                x,
                question_embedding=question_emb,
                question_mask=question_mask,
                deterministic=deterministic
            )
            
            if return_router_outputs:
                router_outputs.append(router_aux)
            
            # Apply layer with routing
            x = layer(x, routing_mask=routing_mask)
        
        # Output
        x = self.norm(x)
        logits = self.lm_head(x)  # (batch, seqlen, vocab_size)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        
        return loss, logits, router_outputs
    
    def get_num_params(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())