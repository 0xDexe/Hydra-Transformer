"""
Robust Evaluation Script for HYDRA Models

Handles checkpoints with NaN validation loss by checking weight health.
Can evaluate model even if training metrics are corrupted.

"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.routed_model import RoutedHybridModel
from src.data.qa_datasets import get_qa_dataloaders


def check_weights_health(state_dict: dict, verbose: bool = True) -> tuple:
    """
    Check if model weights are valid (no NaN/Inf).
    
    Returns:
        (is_healthy, stats_dict)
    """
    nan_params = []
    inf_params = []
    zero_params = []
    total_params = 0
    
    for name, param in state_dict.items():
        total_params += 1
        
        if torch.isnan(param).any():
            nan_params.append(name)
        if torch.isinf(param).any():
            inf_params.append(name)
        if (param == 0).all():
            zero_params.append(name)
    
    stats = {
        'total_params': total_params,
        'nan_params': len(nan_params),
        'inf_params': len(inf_params),
        'zero_params': len(zero_params),
        'nan_param_names': nan_params[:5],  # First 5
        'inf_param_names': inf_params[:5],
    }
    
    is_healthy = (len(nan_params) == 0 and len(inf_params) == 0)
    
    if verbose:
        print(f"\n--- Weight Health Check ---")
        print(f"Total parameters: {total_params}")
        print(f"Parameters with NaN: {len(nan_params)}")
        print(f"Parameters with Inf: {len(inf_params)}")
        print(f"Parameters all zeros: {len(zero_params)}")
        
        if nan_params:
            print(f"\n⚠ WARNING: {len(nan_params)} parameters contain NaN!")
            print("First 5 affected:")
            for name in nan_params[:5]:
                print(f"  - {name}")
        
        if inf_params:
            print(f"\n⚠ WARNING: {len(inf_params)} parameters contain Inf!")
            print("First 5 affected:")
            for name in inf_params[:5]:
                print(f"  - {name}")
        
        if is_healthy:
            print("\n✓ All weights are valid (no NaN/Inf)")
        else:
            print("\n✗ Weights are CORRUPTED")
    
    return is_healthy, stats


class RobustEvaluator:
    """Evaluator that handles corrupted checkpoints"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', force: bool = False):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        self.force = force
        
        print(f"\n{'='*70}")
        print("ROBUST MODEL EVALUATION")
        print(f"{'='*70}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Force mode: {force}")
        
        # Load checkpoint
        print("\nLoading checkpoint...")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            print("✓ Checkpoint file loaded")
        except Exception as e:
            print(f"✗ ERROR loading checkpoint: {e}")
            raise
        
        # Check checkpoint contents
        print(f"\n--- Checkpoint Info ---")
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('val_loss', 'N/A')
        global_step = checkpoint.get('global_step', 'N/A')
        
        print(f"Epoch: {epoch}")
        print(f"Val loss: {val_loss}")
        print(f"Global step: {global_step}")
        
        # Check if val_loss is NaN
        val_loss_is_nan = False
        if isinstance(val_loss, float):
            import math
            if math.isnan(val_loss) or math.isinf(val_loss):
                val_loss_is_nan = True
                print(f"\n⚠ WARNING: Validation loss is {'NaN' if math.isnan(val_loss) else 'Inf'}")
                print("This checkpoint may have been saved during unstable training.")
                print("Checking if model weights are still usable...")
        
        # Check weight health
        state_dict = checkpoint.get('model_state_dict', {})
        if not state_dict:
            print("\n✗ ERROR: No model_state_dict in checkpoint!")
            raise ValueError("Checkpoint missing model weights")
        
        is_healthy, weight_stats = check_weights_health(state_dict, verbose=True)
        
        # Decide whether to proceed
        if not is_healthy and not force:
            print("\n✗ CANNOT EVALUATE: Model weights are corrupted")
            print("\nTo evaluate anyway (may produce garbage), use --force flag")
            raise ValueError("Corrupted model weights")
        elif not is_healthy and force:
            print("\n⚠ FORCING EVALUATION with corrupted weights!")
            print("Results will likely be meaningless, but proceeding as requested...")
        
        self.config = checkpoint.get('config', {})
        
        # Create model
        print("\n--- Creating Model ---")
        try:
            self.model = RoutedHybridModel(
                vocab_size=self.config.get('vocab_size', 50257),
                d_model=self.config.get('d_model', 768),
                n_layers=self.config.get('n_layers', 12),
                n_heads=self.config.get('n_heads', 12),
                d_state=self.config.get('d_state', 16),
                d_conv=self.config.get('d_conv', 4),
                expand=self.config.get('expand', 2),
                d_ff=self.config.get('d_ff', 3072),
                dropout=self.config.get('dropout', 0.1),
                router_hidden_dim=self.config.get('router_hidden_dim', 64),
                target_ratio=self.config.get('target_ratio', 0.15),
                use_gradient_balancing=self.config.get('use_gradient_balancing', True),
                use_position_invariance=self.config.get('use_position_invariance', True),
                use_checkpoint=False,
                tie_weights=self.config.get('tie_weights', True),
                use_query_aware_routing=self.config.get('use_query_aware_routing', False),
            ).to(self.device)
            
            print(f"✓ Model architecture created")
        except Exception as e:
            print(f"✗ ERROR creating model: {e}")
            print("\nTrying with default parameters...")
            # Fallback to defaults
            self.model = RoutedHybridModel(
                vocab_size=50257,
                d_model=768,
                n_layers=12,
                n_heads=12,
            ).to(self.device)
            print("✓ Model created with defaults")
        
        # Load weights
        print("\n--- Loading Weights ---")
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("✓ Weights loaded successfully (strict mode)")
        except Exception as e:
            print(f"⚠ Strict loading failed: {e}")
            print("Trying non-strict loading...")
            try:
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                print(f"✓ Weights loaded (non-strict)")
                if missing:
                    print(f"  Missing keys: {len(missing)}")
                if unexpected:
                    print(f"  Unexpected keys: {len(unexpected)}")
            except Exception as e2:
                print(f"✗ ERROR loading weights: {e2}")
                raise
        
        self.model.eval()
        
        num_params = self.model.get_num_params()
        print(f"✓ Model ready: {num_params / 1e6:.1f}M parameters")
        
        # Store weight health info
        self.weight_health = {
            'is_healthy': is_healthy,
            'stats': weight_stats,
            'val_loss_was_nan': val_loss_is_nan,
        }
        
        print(f"{'='*70}\n")
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        dataset_name: str,
        max_samples: int = None
    ) -> Dict:
        """Run evaluation"""
        
        print(f"\n{'='*70}")
        print(f"EVALUATING ON {dataset_name.upper()}")
        print(f"{'='*70}")
        
        if not self.weight_health['is_healthy']:
            print("⚠ WARNING: Evaluating with potentially corrupted weights!")
            print("Results may be unreliable.\n")
        
        # Metrics
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        num_nan_batches = 0
        
        # Router statistics
        router_stats = {
            'routing_ratios': [],
            'layer_ratios': [[] for _ in range(self.config.get('n_layers', 12))]
        }
        
        # Timing
        start_time = time.time()
        total_tokens_processed = 0
        
        # Iterate
        pbar = tqdm(dataloader, desc="Evaluating")
        
        for batch_idx, batch in enumerate(pbar):
            # Check max_samples
            if max_samples and batch_idx * dataloader.batch_size >= max_samples:
                break
            
            try:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                loss, logits, router_outputs = self.model(
                    input_ids,
                    labels=labels,
                    deterministic=True,
                    return_router_outputs=True
                )
                
                # Check for NaN in outputs
                if torch.isnan(loss) or torch.isinf(loss):
                    num_nan_batches += 1
                    pbar.set_postfix({
                        'loss': 'NaN',
                        'nan_batches': num_nan_batches
                    })
                    continue
                
                # Loss
                total_loss += loss.item()
                
                # Accuracy
                predictions = logits.argmax(dim=-1)
                mask = labels != -100
                correct = (predictions == labels) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                
                # Router statistics
                if router_outputs:
                    for layer_idx, router_out in enumerate(router_outputs):
                        if 'routing_mask' in router_out:
                            routing_mask = router_out['routing_mask']
                            ratio = routing_mask.float().mean().item()
                            router_stats['layer_ratios'][layer_idx].append(ratio)
                    
                    # Overall routing ratio
                    avg_ratio = sum(
                        r.get('routing_ratio', 0.0) for r in router_outputs
                    ) / len(router_outputs)
                    router_stats['routing_ratios'].append(avg_ratio)
                
                # Throughput
                total_tokens_processed += input_ids.numel()
                
                num_batches += 1
                
                # Update progress
                if batch_idx % 10 == 0:
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'acc': f"{total_correct / max(total_tokens, 1) * 100:.2f}%",
                        'nan_batches': num_nan_batches
                    })
            
            except Exception as e:
                print(f"\n⚠ Error on batch {batch_idx}: {e}")
                num_nan_batches += 1
                continue
        
        # Compute final metrics
        elapsed_time = time.time() - start_time
        
        if num_batches == 0:
            print("\n✗ ERROR: All batches failed!")
            print("Model weights are likely completely corrupted.")
            return None
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = total_correct / total_tokens * 100 if total_tokens > 0 else 0.0
        throughput = total_tokens_processed / elapsed_time
        
        # Router metrics
        avg_routing_ratio = np.mean(router_stats['routing_ratios']) if router_stats['routing_ratios'] else 0.0
        layer_routing_ratios = [
            np.mean(ratios) if ratios else 0.0 
            for ratios in router_stats['layer_ratios']
        ]
        
        results = {
            'dataset': dataset_name,
            'num_samples': num_batches * dataloader.batch_size,
            'num_batches': num_batches,
            'num_nan_batches': num_nan_batches,
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'total_tokens': total_tokens,
            'correct_tokens': total_correct,
            'time_seconds': elapsed_time,
            'throughput_tokens_per_sec': throughput,
            'router': {
                'avg_routing_ratio': avg_routing_ratio,
                'target_ratio': self.config.get('target_ratio', 0.15),
                'layer_routing_ratios': layer_routing_ratios,
            },
            'model': {
                'num_parameters': self.model.get_num_params(),
                'num_layers': self.config.get('n_layers', 12),
                'd_model': self.config.get('d_model', 768),
            },
            'checkpoint_health': self.weight_health,
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Pretty print results"""
        if results is None:
            print("\n✗ Evaluation failed - no results to display")
            return
        
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        
        # Health warning
        if not results['checkpoint_health']['is_healthy']:
            print("\n⚠ ⚠ ⚠  WARNING: CHECKPOINT WAS CORRUPTED  ⚠ ⚠ ⚠")
            print("Results below may not be meaningful!")
            print(f"{'='*70}\n")
        
        print(f"\nDataset: {results['dataset']}")
        print(f"Samples: {results['num_samples']}")
        print(f"Successful batches: {results['num_batches']}")
        print(f"Failed batches (NaN): {results['num_nan_batches']}")
        
        print(f"\n--- Language Modeling Metrics ---")
        print(f"Loss:       {results['loss']:.4f}")
        print(f"Perplexity: {results['perplexity']:.2f}")
        print(f"Accuracy:   {results['accuracy']:.2f}%")
        
        print(f"\n--- Router Statistics ---")
        print(f"Avg routing ratio: {results['router']['avg_routing_ratio']:.2%}")
        print(f"Target ratio:      {results['router']['target_ratio']:.2%}")
        
        print(f"\nPer-layer routing ratios:")
        for i, ratio in enumerate(results['router']['layer_routing_ratios']):
            print(f"  Layer {i:2d}: {ratio:.2%}")
        
        print(f"\n--- Performance ---")
        print(f"Time:       {results['time_seconds']:.1f} seconds")
        print(f"Throughput: {results['throughput_tokens_per_sec']:.0f} tokens/sec")
        
        print(f"\n--- Model Info ---")
        print(f"Parameters: {results['model']['num_parameters'] / 1e6:.1f}M")
        print(f"Layers:     {results['model']['num_layers']}")
        print(f"d_model:    {results['model']['d_model']}")
        
        print(f"\n{'='*70}\n")
    
    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON"""
        if results is None:
            print("✗ Cannot save results - evaluation failed")
            return
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Robust evaluation for HYDRA models")
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to checkpoint file')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='naturalqa',
                       choices=['narrativeqa', 'naturalqa', 'squad'],
                       help='Dataset to evaluate on')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples')
    
    # Data loading
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--max-length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    
    # Output
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results JSON')
    
    # Force evaluation
    parser.add_argument('--force', action='store_true',
                       help='Force evaluation even if weights are corrupted')
    
    args = parser.parse_args()
    
    # Auto-generate output path
    if args.output is None:
        checkpoint_dir = Path(args.checkpoint).parent
        args.output = checkpoint_dir / f'eval_results_{args.dataset}_robust.json'
    
    # Load model
    try:
        evaluator = RobustEvaluator(args.checkpoint, force=args.force)
    except Exception as e:
        print(f"\n✗ FATAL ERROR: Cannot create evaluator")
        print(f"Error: {e}")
        print("\nIf you want to try anyway, use --force flag")
        sys.exit(1)
    
    # Load dataset
    print(f"\n{'='*70}")
    print("LOADING DATASET")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples or 'all'}")
    
    try:
        _, val_loader, tokenizer = get_qa_dataloaders(
            dataset_name=args.dataset,
            tokenizer_name=evaluator.config.get('tokenizer_name', 'gpt2'),
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_train_samples=0,
            max_val_samples=args.max_samples,
        )
        
        print(f"✓ Loaded {len(val_loader)} batches")
        print(f"{'='*70}\n")
    except Exception as e:
        print(f"\n✗ ERROR loading dataset: {e}")
        sys.exit(1)
    
    # Run evaluation
    try:
        results = evaluator.evaluate(
            val_loader,
            args.dataset,
            max_samples=args.max_samples
        )
    except Exception as e:
        print(f"\n✗ ERROR during evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    if results:
        evaluator.save_results(results, args.output)
        print("\n✓ Evaluation complete!")
    else:
        print("\n✗ Evaluation failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()