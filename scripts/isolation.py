#!/usr/bin/env python3
"""
Progressive Feature Testing for NarrativeQA

Tests each feature incrementally to find what works:
1. Baseline (minimal features)
2. + Gradient balancing
3. + Position invariance
4. + Curriculum learning
5. + Router loss components
6. + Mixed precision (AMP)
7. Full config

This identifies exactly which feature causes CUDA errors.
"""

import subprocess
import sys
import json
from pathlib import Path
import yaml

# Test configurations in order of complexity
TESTS = [
    {
        'name': 'baseline',
        'description': 'Minimal features (known working)',
        'config': {
            'use_gradient_balancing': False,
            'use_position_invariance': False,
            'router_loss_variance': 0.0,
            'router_loss_position_inv': 0.0,
            'router_loss_diversity': 0.0,
            'curriculum_warmup': 0,
            'use_amp': False,
        }
    },
    {
        'name': 'gradient_balancing',
        'description': '+ Gradient balancing',
        'config': {
            'use_gradient_balancing': True,
            'use_position_invariance': False,
            'router_loss_variance': 0.0,
            'router_loss_position_inv': 0.0,
            'router_loss_diversity': 0.0,
            'curriculum_warmup': 0,
            'use_amp': False,
        }
    },
    {
        'name': 'position_invariance',
        'description': '+ Position invariance',
        'config': {
            'use_gradient_balancing': True,
            'use_position_invariance': True,
            'router_loss_variance': 0.0,
            'router_loss_position_inv': 0.005,
            'router_loss_diversity': 0.0,
            'curriculum_warmup': 0,
            'use_amp': False,
        }
    },
    {
        'name': 'curriculum',
        'description': '+ Curriculum learning',
        'config': {
            'use_gradient_balancing': True,
            'use_position_invariance': True,
            'router_loss_variance': 0.0,
            'router_loss_position_inv': 0.005,
            'router_loss_diversity': 0.0,
            'curriculum_warmup': 2000,
            'use_amp': False,
        }
    },
    {
        'name': 'full_router_loss',
        'description': '+ All router loss components',
        'config': {
            'use_gradient_balancing': True,
            'use_position_invariance': True,
            'router_loss_variance': 0.01,
            'router_loss_position_inv': 0.005,
            'router_loss_diversity': 0.005,
            'curriculum_warmup': 2000,
            'use_amp': False,
        }
    },
    {
        'name': 'with_amp',
        'description': '+ Mixed precision (AMP)',
        'config': {
            'use_gradient_balancing': True,
            'use_position_invariance': True,
            'router_loss_variance': 0.01,
            'router_loss_position_inv': 0.005,
            'router_loss_diversity': 0.005,
            'curriculum_warmup': 2000,
            'use_amp': True,
        }
    },
]


def create_test_config(test_name, test_config):
    """Create a config file for testing"""
    
    base_config = {
        'model': {
            'd_model': 256,
            'n_layers': 2,
            'n_heads': 4,
            'd_state': 16,
            'd_conv': 4,
            'expand': 2,
            'dropout': 0.0,
            'tie_weights': True,
        },
        'router': {
            'hidden_dim': 32,
            'target_ratio': 0.15,
            'use_gradient_balancing': test_config['use_gradient_balancing'],
            'use_position_invariance': test_config['use_position_invariance'],
            'use_checkpoint': False,
            'use_query_aware_routing': False,
        },
        'curriculum': {
            'warmup_steps': test_config['curriculum_warmup'],
            'heuristic_type': 'uniform',
        },
        'router_loss': {
            'total_weight': 0.01,
            'load_balance': 0.01,
            'entropy': 0.01,
            'variance': test_config['router_loss_variance'],
            'position_invariance': test_config['router_loss_position_inv'],
            'layer_diversity': test_config['router_loss_diversity'],
        },
        'data': {
            'dataset_name': 'squad',
            'tokenizer_name': 'gpt2',
            'max_length': 128,
            'batch_size': 2,
            'num_workers': 0,
            'max_train_samples': 50,
            'max_val_samples': 10,
        },
        'training': {
            'num_epochs': 1,
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'use_amp': test_config['use_amp'],
            'gradient_accumulation_steps': 1,
            'use_fused_optimizer': False,
            'compile_model': False,
        },
        'logging': {
            'use_wandb': False,
            'project_name': 'hydra-progressive-test',
            'run_name': f'test-{test_name}',
            'output_dir': f'outputs/progressive_test/{test_name}',
            'log_interval': 10,
            'monitor_interval': 50,
            'save_every': 1,
        }
    }
    
    # Save config
    config_path = Path(f'configs/progressive_test_{test_name}.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False)
    
    return config_path


def run_test(test_name, test_config, test_description):
    """Run a single test"""
    print("\n" + "="*70)
    print(f"TEST: {test_name}")
    print(f"DESC: {test_description}")
    print("="*70)
    
    # Create config
    config_path = create_test_config(test_name, test_config)
    print(f"✓ Config created: {config_path}")
    
    # Run training
    cmd = [
        'python', 'scripts/train_qa.py',
        '--config', str(config_path),
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print("-"*70)
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True,
            timeout=600  # 10 minute timeout per test
        )
        
        print("-"*70)
        print(f"✓ TEST PASSED: {test_name}")
        return True
        
    except subprocess.CalledProcessError as e:
        print("-"*70)
        print(f"✗ TEST FAILED: {test_name}")
        print(f"Error: {e}")
        return False
        
    except subprocess.TimeoutExpired:
        print("-"*70)
        print(f"✗ TEST TIMEOUT: {test_name}")
        return False


def main():
    print("\n" + "="*70)
    print("PROGRESSIVE FEATURE TESTING")
    print("="*70)
    print("\nThis will test each feature incrementally to find what works.")
    print("Each test runs 1 epoch with 50 samples (~2-3 minutes).")
    print(f"\nTotal tests: {len(TESTS)}")
    print("="*70)
    
    input("\nPress Enter to start testing...")
    
    results = {}
    
    for i, test in enumerate(TESTS, 1):
        print(f"\n\n{'='*70}")
        print(f"PROGRESS: Test {i}/{len(TESTS)}")
        print(f"{'='*70}")
        
        success = run_test(
            test['name'],
            test['config'],
            test['description']
        )
        
        results[test['name']] = {
            'success': success,
            'description': test['description'],
            'config': test['config']
        }
        
        if not success:
            print(f"\n{'='*70}")
            print("STOPPING: Test failed")
            print(f"{'='*70}")
            print(f"\nFailed at: {test['name']}")
            print(f"Description: {test['description']}")
            print(f"\nThe previous test passed, so the issue is with:")
            
            if i > 1:
                prev_test = TESTS[i-2]
                print(f"\n Working config: {prev_test['name']}")
                print(f" Failing config: {test['name']}")
                print(f"\nDifference:")
                for key, value in test['config'].items():
                    prev_value = prev_test['config'][key]
                    if value != prev_value:
                        print(f"  {key}: {prev_value} → {value}")
            
            break
    
    # Summary
    print("\n\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result['success'] else "✗ FAIL"
        print(f"{status}: {test_name} - {result['description']}")
    
    # Save results
    results_path = Path('/projectnb/cs523aw/students/waqar/outputs/progressive_test/results.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_path}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Find last passing test
    last_pass = None
    for test_name, result in results.items():
        if result['success']:
            last_pass = test_name
        else:
            break
    
    if last_pass:
        print(f"\n Safe config: Use features from '{last_pass}' test")
        print(f"\nTo use this for NarrativeQA:")
        print(f"1. Copy configs/progressive_test_{last_pass}.yaml")
        print(f"2. Update:")
        print(f"   - dataset_name: 'narrativeqa'")
        print(f"   - max_length: 2048")
        print(f"   - d_model: 768")
        print(f"   - n_layers: 12")
        print(f"   - max_train_samples: null (use all data)")
        print(f"   - num_epochs: 20")
    else:
        print("\n✗ All tests failed - check base model")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    main()