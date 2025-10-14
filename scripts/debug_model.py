import torch
from src.model.hybrid_model import HybridSSMTransformer

def print_model_summary(model):
    """Print detailed model summary"""
    print(f"\n{'='*60}")
    print("MODEL SUMMARY")
    print(f"{'='*60}")
    
    total_params = 0
    for name, param in model.named_parameters():
        params = param.numel()
        total_params += params
        print(f"{name:50s} {params:>15,}")
    
    print(f"{'='*60}")
    print(f"{'TOTAL PARAMETERS':50s} {total_params:>15,}")
    print(f"{'SIZE (MB)':50s} {total_params * 4 / 1024 / 1024:>15.2f}")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    model = HybridSSMTransformer(
        vocab_size=50257,
        d_model=768,
        n_layers=12,
        pattern='alternating'
    )
    
    print_model_summary(model)
    
    # Test shapes
    dummy_input = torch.randint(0, 50257, (1, 512))
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")