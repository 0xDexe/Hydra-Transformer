import torch
from src.model.routed_model import RoutedHybridSSMTransformer


def test_model_forward():
    """Test basic forward pass for RoutedHybridSSMTransformer"""
    vocab_size = 1000
    batch_size = 2
    seq_len = 128
    d_model = 256

    # Instantiate routed hybrid model
    model = RoutedHybridSSMTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=4,              # 4 (block + FFN) pairs
        n_heads=8,               # 256 / 8 = 32 dim per head
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        max_seq_len=seq_len,
        routing_topk_ratio=0.2,  # 20% of tokens to attention
        router_hidden_dim=None,  # defaults to d_model
        context_mode="conv"      # or "ssm" / "local_attn"
    )

    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    loss, logits = model(input_ids, labels=labels)

    print(f" ==== Forward pass successful ==== ")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected shape: ({batch_size}, {seq_len}, {vocab_size})")

    # Basic sanity checks
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss.item() > 0

    print(f" ==== All tests passed! ==== ")


if __name__ == '__main__':
    test_model_forward()
