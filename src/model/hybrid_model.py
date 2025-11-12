import torch
import torch.nn as nn
from src.model.attention_block import AttentionBlock
from src.model.ssm_block import SSMBlock
from src.model.ffn_block import FFNBlock


class HybridSSMTransformer(nn.Module):
    """
    Hybrid model combining SSM (Mamba) and Transformer layers
    
    This architecture alternates between efficient SSM layers and powerful
    Transformer attention layers to balance computational efficiency with
    modeling capability.
    
    Args:
        vocab_size: Size of the vocabulary
        d_model: Model dimension
        n_layers: Number of layers (will be doubled as each block has compute + FFN)
        pattern: Layer arrangement pattern - 'alternating', 'attention_first', or 'ssm_first'
        n_heads: Number of attention heads (for attention layers)
        d_state: SSM state dimension (for Mamba layers)
        d_conv: SSM convolution dimension (for Mamba layers)
        expand: SSM expansion factor (for Mamba layers)
        dropout: Dropout probability
        max_seq_len: Maximum sequence length for position embeddings
    """
    def __init__(
        self, 
        vocab_size,
        d_model,
        n_layers,
        pattern='alternating',
        n_heads=8,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        max_seq_len=8192
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.pattern = pattern
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # Dropout for embeddings
        self.embed_dropout = nn.Dropout(dropout)
        
        # Build layers based on pattern
        self.layers = nn.ModuleList()
        
        for i in range(n_layers):
            # Determine layer type based on pattern
            if pattern == 'alternating':
                # Alternate between attention and SSM
                if i % 2 == 0:
                    layer = AttentionBlock(d_model, n_heads, dropout)
                else:
                    layer = SSMBlock(d_model, d_state, d_conv, expand)
            
            elif pattern == 'attention_first':
                # First half attention, second half SSM
                if i < n_layers // 2:
                    layer = AttentionBlock(d_model, n_heads, dropout)
                else:
                    layer = SSMBlock(d_model, d_state, d_conv, expand)
            
            elif pattern == 'ssm_first':
                # First half SSM, second half attention
                if i < n_layers // 2:
                    layer = SSMBlock(d_model, d_state, d_conv, expand)
                else:
                    layer = AttentionBlock(d_model, n_heads, dropout)
            
            else:
                raise ValueError(f"Unknown pattern: {pattern}. Use 'alternating', 'attention_first', or 'ssm_first'")
            
            self.layers.append(layer)
            
            # Add FFN after each main layer
            self.layers.append(FFNBlock(d_model, dropout=dropout))
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings (weight sharing between input and output embeddings)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following best practices"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, labels=None):
        """
        Forward pass through the model
        
        Args:
            input_ids: (batch, seqlen) - Input token IDs
            labels: (batch, seqlen) - Optional labels for computing loss
            
        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # (batch, seqlen, d_model)
        
        # Add position embeddings
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Apply dropout to embeddings
        x = self.embed_dropout(x)
        
        # Apply all layers
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (batch, seqlen, vocab_size)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next-token prediction
            # We want to predict token at position i+1 given tokens 0...i
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten for cross entropy
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            
            return loss, logits
        
        return logits
    
    def get_num_params(self, non_embedding=False):
        """
        Return the number of parameters in the model
        
        Args:
            non_embedding: If True, subtract embedding parameters
        """
        n_params = sum(p.numel() for p in self.parameters())
        
        if non_embedding:
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.pos_embedding.numel()
        
        return n_params
    
    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS
        
        Args:
            fwdbwd_per_iter: Number of forward-backward passes per iteration
            dt: Time taken for iteration in seconds
        """
        # A100 GPU peak flops (bfloat16)
        N = self.get_num_params()
        cfg = self
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.d_model // cfg.n_heads, 1024
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops
        mfu = flops_achieved / flops_promised
        return mfu
    
    def get_layer_info(self):
        """Get information about layer composition"""
        layer_types = []
        for i, layer in enumerate(self.layers):
            if isinstance(layer, AttentionBlock):
                layer_types.append(f"Layer {i//2}: Attention")
            elif isinstance(layer, SSMBlock):
                layer_types.append(f"Layer {i//2}: SSM")
            elif isinstance(layer, FFNBlock):
                layer_types.append(f"  └─ FFN")
        return layer_types
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Configure optimizer with weight decay applied selectively
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Adam betas
            device_type: 'cuda' or 'cpu'
        """
        # Start with all parameters that require grad
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        
        # Separate into decay and no_decay groups
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"Optimizer groups: {len(decay_params)} tensors with decay, {len(nodecay_params)} without")
        print(f"  Decay params: {num_decay_params:,}")
        print(f"  No decay params: {num_nodecay_params:,}")
        
        # Use fused AdamW if on CUDA
        use_fused = (device_type == 'cuda')
        extra_args = dict(fused=True) if use_fused else dict()
        
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            **extra_args
        )
        
        return optimizer
