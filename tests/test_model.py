import torch
from src.model.hybrid_model import HybridSSMTransformer

def test_model_forward():
    """Test basic forward pass"""
    vocab_size = 1000
    batch_size = 2
    seq_len = 128
    d_model = 256
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HybridSSMTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=4,
        pattern='alternating'
    ).to(device)
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    
    # Forward pass
    loss, logits = model(input_ids, labels=labels)
    
    print(f" ==== Forward pass successful ==== ")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected shape: ({batch_size}, {seq_len}, {vocab_size})")
    
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert loss.item() > 0

    print(f" ==== All tests passed! ==== ")

if __name__ == '__main__':
    test_model_forward()