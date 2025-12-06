import torch
import time
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.routed_model import RoutedHybridModel, RoutedHybridLayer
from model.router import EfficientTokenRouter, RouterLoss, RouterMonitor
from src.model.hybrid_model import HybridSSMTransformer  # Baseline for comparison


def test_router_basic():
    """Test basic router functionality"""
    print("\n" + "="*60)
    print("TEST: Router Basic Functionality")
    print("="*60)
    
    d_model = 256
    batch_size = 4
    seqlen = 128
    
    router = EfficientTokenRouter(
        d_model=d_model,
        hidden_dim=64,
        target_ratio=0.15,
        layer_idx=0,
        total_layers=6
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seqlen, d_model)
    
    # Forward pass
    mask, aux = router(x, deterministic=True)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Mask shape: {mask.shape}")
    print(f"✓ Routing ratio: {aux['stats']['routing_ratio']:.2%}")
    print(f"✓ Mean prob: {aux['stats']['mean_prob']:.4f}")
    print(f"✓ Threshold: {aux['stats']['threshold']:.4f}")
    
    # Check mask properties
    assert mask.shape == (batch_size, seqlen)
    assert mask.dtype == torch.bool
    assert 0.05 <= aux['stats']['routing_ratio'] <= 0.25  # Within reasonable range
    
    print("✓ All router tests passed!")


def test_routed_layer():
    """Test routed hybrid layer"""
    print("\n" + "="*60)
    print("TEST: Routed Hybrid Layer")
    print("="*60)
    
    d_model = 256
    batch_size = 2
    seqlen = 64
    
    layer = RoutedHybridLayer(
        d_model=d_model,
        n_heads=8,
        router_hidden_dim=64,
        target_ratio=0.15,
        layer_idx=0,
        total_layers=6
    )
    
    # Create dummy input
    x = torch.randn(batch_size, seqlen, d_model)
    
    # Forward pass
    output, aux = layer(x, deterministic=True)
    
    print(f"✓ Input shape: {x.shape}")
    print(f"✓ Output shape: {output.shape}")
    print(f"✓ Tokens routed: {aux['num_routed']}/{batch_size * seqlen}")
    print(f"✓ Routing ratio: {aux['routing_ratio']:.2%}")
    
    # Check properties
    assert output.shape == x.shape
    assert aux['num_routed'] > 0
    assert aux['routing_ratio'] > 0
    
    print("✓ All layer tests passed!")


def test_routed_model():
    """Test complete routed model"""
    print("\n" + "="*60)
    print("TEST: Routed HYDRA Model")
    print("="*60)
    
    vocab_size = 1000
    batch_size = 2
    seqlen = 64
    
    model = RoutedHybridModel(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=8,
        target_ratio=0.15
    )
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seqlen))
    labels = input_ids.clone()
    
    # Forward pass
    loss, logits, router_outputs = model(
        input_ids,
        labels=labels,
        return_router_outputs=True
    )
    
    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Logits shape: {logits.shape}")
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Number of layers with routing: {len(router_outputs)}")
    
    # Check routing statistics
    avg_ratio = sum(out['routing_ratio'] for out in router_outputs) / len(router_outputs)
    print(f"✓ Average routing ratio: {avg_ratio:.2%}")
    
    # Check properties
    assert logits.shape == (batch_size, seqlen, vocab_size)
    assert loss.item() > 0
    assert len(router_outputs) == 4
    
    print("✓ All model tests passed!")


def test_gradient_flow():
    """Test that gradients flow correctly through router"""
    print("\n" + "="*60)
    print("TEST: Gradient Flow")
    print("="*60)
    
    vocab_size = 1000
    batch_size = 2
    seqlen = 32
    
    model = RoutedHybridModel(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=4,
        target_ratio=0.15,
        use_gradient_balancing=True
    )
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seqlen))
    labels = input_ids.clone()
    
    # Forward pass
    loss, logits, router_outputs = model(
        input_ids,
        labels=labels,
        return_router_outputs=True
    )
    
    # Backward pass
    loss.backward()
    
    # Check that router has gradients
    router_has_grad = False
    for name, param in model.named_parameters():
        if 'router' in name and param.grad is not None:
            router_has_grad = True
            grad_norm = param.grad.norm().item()
            print(f"✓ {name}: grad norm = {grad_norm:.4f}")
    
    assert router_has_grad, "Router parameters should have gradients!"
    
    print("✓ Gradient flow test passed!")


def test_router_loss():
    """Test router loss computation"""
    print("\n" + "="*60)
    print("TEST: Router Loss")
    print("="*60)
    
    # Create dummy router outputs
    batch_size = 4
    seqlen = 64
    n_layers = 3
    
    router_outputs = []
    for i in range(n_layers):
        probs = torch.rand(batch_size, seqlen) * 0.3 + 0.1  # Range [0.1, 0.4]
        router_outputs.append({
            'router_probs': probs,
            'router_logits': torch.logit(probs),
            'routing_mask': probs > 0.2,
        })
    
    # Compute loss
    loss_fn = RouterLoss(
        target_ratio=0.15,
        load_weight=0.01,
        entropy_weight=0.01,
        diversity_weight=0.005
    )
    
    # With return_components=True, type checker should know this returns tuple
    total_loss, components = loss_fn(router_outputs, return_components=True)
    
    print(f"✓ Total loss: {total_loss.item():.6f}")
    print(f"✓ Load balance loss: {components['load_balance']:.6f}")
    print(f"✓ Entropy loss: {components['entropy']:.6f}")
    print(f"✓ Diversity loss: {components['layer_diversity']:.6f}")
    
    assert total_loss.item() > 0
    assert all(v >= 0 for v in components.values() if v != components['total'])
    
    print("✓ Router loss test passed!")


def test_router_monitoring():
    """Test router monitoring"""
    print("\n" + "="*60)
    print("TEST: Router Monitoring")
    print("="*60)
    
    monitor = RouterMonitor(num_layers=3)
    
    # Log some fake statistics
    for layer_idx in range(3):
        for step in range(5):
            probs = torch.rand(4, 64) * 0.3 + 0.05 * layer_idx
            stats = monitor.log_layer_stats(layer_idx, probs, step)
            
            if step == 4:  # Last step
                print(f"✓ Layer {layer_idx} stats:")
                print(f"  Routing ratio: {stats[f'layer_{layer_idx}/routing_ratio']:.2%}")
                print(f"  Entropy: {stats[f'layer_{layer_idx}/entropy']:.4f}")
                print(f"  Position corr: {stats[f'layer_{layer_idx}/position_corr']:.4f}")
    
    # Get summary
    summary = monitor.get_summary()
    print(f"\n✓ Summary statistics:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    
    assert len(summary) > 0
    print("✓ Monitoring test passed!")


def benchmark_efficiency():
    """Benchmark efficiency of routed vs baseline model"""
    print("\n" + "="*60)
    print("BENCHMARK: Efficiency Comparison")
    print("="*60)
    
    vocab_size = 50257
    batch_size = 4
    seqlen = 512
    d_model = 768
    n_layers = 6
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Sequence length: {seqlen}")
    
    # Create models
    routed_model = RoutedHybridModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=12,
        target_ratio=0.15
    ).to(device)
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seqlen)).to(device)
    
    # Warmup
    for _ in range(3):
        with torch.no_grad():
            _ = routed_model(input_ids, deterministic=True)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark forward pass
    num_runs = 10
    start = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            loss, logits, router_outputs = routed_model(
                input_ids,
                labels=input_ids,
                deterministic=True,
                return_router_outputs=True
            )
        if device.type == 'cuda':
            torch.cuda.synchronize()
    forward_time = (time.time() - start) / num_runs
    
    print(f"\n✓ Routed Model:")
    print(f"  Parameters: {routed_model.get_num_params() / 1e6:.2f}M")
    print(f"  Forward pass time: {forward_time*1000:.2f} ms")
    
    # Check routing statistics
    if router_outputs:
        avg_ratio = sum(out['routing_ratio'] for out in router_outputs) / len(router_outputs)
        print(f"  Average routing ratio: {avg_ratio:.2%}")
        print(f"  Tokens through attention: {avg_ratio * batch_size * seqlen:.0f}/{batch_size * seqlen}")
    
    # Memory usage
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        with torch.no_grad():
            _ = routed_model(input_ids, deterministic=True)
        
        memory_used = torch.cuda.max_memory_allocated() / 1024**2
        print(f"  Peak memory: {memory_used:.2f} MB")
    
    print("\n✓ Benchmark complete!")


def test_position_invariance():
    """Test that router is position-invariant (or has minimal position bias)"""
    print("\n" + "="*60)
    print("TEST: Position Invariance")
    print("="*60)
    
    d_model = 256
    batch_size = 2
    seqlen = 128
    
    router = EfficientTokenRouter(
        d_model=d_model,
        hidden_dim=64,
        target_ratio=0.15,
        use_position_invariance=True,
        layer_idx=0,
        total_layers=6
    )
    
    # Create identical tokens at different positions
    x = torch.randn(1, 1, d_model).expand(batch_size, seqlen, d_model).clone()
    
    # Get routing scores
    _, aux = router(x, deterministic=True)
    probs = aux['router_probs']
    
    # Check variance in probabilities
    # If position-invariant, all tokens should get similar scores
    prob_std = probs.std().item()
    prob_range = (probs.max() - probs.min()).item()
    
    print(f"✓ Probability std: {prob_std:.4f}")
    print(f"✓ Probability range: {prob_range:.4f}")
    
    # With position invariance, std should be small
    if router.use_position_invariance:
        assert prob_std < 0.1, f"Position-invariant router should have low std, got {prob_std}"
        print("✓ Position invariance verified!")
    else:
        print("✓ Position bias allowed (use_position_invariance=False)")


def test_routing_collapse_detection():
    """Test that routing collapse is detected"""
    print("\n" + "="*60)
    print("TEST: Routing Collapse Detection")
    print("="*60)
    
    monitor = RouterMonitor(num_layers=1)
    
    # Test 1: All tokens routed (collapsed to all-attention)
    all_routed = torch.ones(4, 64) * 0.9
    collapse_info = monitor.detect_collapse(all_routed, target_ratio=0.15)
    print(f"✓ All routed (probs=0.9):")
    print(f"  Collapsed: {collapse_info['collapsed']}")
    print(f"  Deviation: {collapse_info['deviation']:.4f}")
    assert collapse_info['collapsed'], "Should detect all-routed collapse"
    
    # Test 2: No tokens routed (collapsed to all-SSM)
    none_routed = torch.ones(4, 64) * 0.1
    collapse_info = monitor.detect_collapse(none_routed, target_ratio=0.15)
    print(f"\n✓ None routed (probs=0.1):")
    print(f"  Collapsed: {collapse_info['collapsed']}")
    print(f"  Deviation: {collapse_info['deviation']:.4f}")
    assert collapse_info['collapsed'], "Should detect none-routed collapse"
    
    # Test 3: Healthy routing
    healthy = torch.rand(4, 64) * 0.3 + 0.05  # Range [0.05, 0.35]
    collapse_info = monitor.detect_collapse(healthy, target_ratio=0.15)
    print(f"\n✓ Healthy routing:")
    print(f"  Collapsed: {collapse_info['collapsed']}")
    print(f"  Entropy: {collapse_info['entropy']:.4f}")
    print(f"  Variance: {collapse_info['variance']:.4f}")
    
    print("\n✓ Collapse detection test passed!")


def test_generation():
    """Test autoregressive generation with routing"""
    print("\n" + "="*60)
    print("TEST: Generation")
    print("="*60)
    
    vocab_size = 1000
    model = RoutedHybridModel(
        vocab_size=vocab_size,
        d_model=256,
        n_layers=4,
        n_heads=8,
        target_ratio=0.15
    )
    model.eval()
    
    # Start with a sequence
    start_ids = torch.randint(0, vocab_size, (1, 10))
    
    print(f"✓ Starting sequence length: {start_ids.shape[1]}")
    
    # Generate
    generated = model.generate(
        start_ids,
        max_new_tokens=20,
        temperature=1.0,
        deterministic_routing=True
    )
    
    print(f"✓ Generated sequence length: {generated.shape[1]}")
    print(f"✓ Tokens generated: {generated.shape[1] - start_ids.shape[1]}")
    
    assert generated.shape[1] == start_ids.shape[1] + 20
    assert generated[:, :start_ids.shape[1]].equal(start_ids)
    
    print("✓ Generation test passed!")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("RUNNING ALL TESTS FOR ROUTED HYDRA")
    print("="*60)
    
    try:
        test_router_basic()
        test_routed_layer()
        test_routed_model()
        test_gradient_flow()
        test_router_loss()
        test_router_monitoring()
        test_position_invariance()
        test_routing_collapse_detection()
        test_generation()
        benchmark_efficiency()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    run_all_tests()