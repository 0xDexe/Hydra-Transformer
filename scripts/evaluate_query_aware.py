"""
Evaluation Script for Query-Aware HYDRA Model

Evaluates trained query-aware model on test set with proper question extraction.

Usage:
    python scripts/evaluate_query_aware.py \
        --checkpoint outputs/query_aware_naturalqa/checkpoint_best.pt \
        --dataset naturalqa
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

from src.model.query_aware_model import QueryAwareHybridModel
from src.model.question_extractor import QuestionExtractor
from src.data.qa_datasets import get_qa_dataloaders


class QueryAwareEvaluator:
    """Evaluator for query-aware HYDRA model"""
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = Path(checkpoint_path)
        
        print(f"\n{'='*70}")
        print("LOADING QUERY-AWARE MODEL")
        print(f"{'='*70}")
        print(f"Checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint['config']
        
        print(f"✓ Checkpoint loaded")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
        
        # Create model
        print("\nCreating model...")
        self.model = QueryAwareHybridModel(
            vocab_size=self.config['vocab_size'],
            d_model=self.config['d_model'],
            n_layers=self.config['n_layers'],
            n_heads=self.config['n_heads'],
            d_state=self.config.get('d_state', 16),
            d_conv=self.config.get('d_conv', 4),
            expand=self.config.get('expand', 2),
            d_ff=self.config.get('d_ff', 3072),
            dropout=self.config.get('dropout', 0.1),
            target_ratio=self.config.get('target_ratio', 0.15),
            router_hidden_dim=self.config.get('hidden_dim', 64),
            use_gradient_balancing=self.config.get('use_gradient_balancing', True),
            use_position_invariance=self.config.get('use_position_invariance', True),
            tie_weights=self.config.get('tie_weights', True),
        ).to(self.device)
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✓ Model loaded: {self.model.get_num_params() / 1e6:.1f}M parameters")
        print(f"{'='*70}\n")
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        question_extractor: QuestionExtractor,
        dataset_name: str,
        max_samples: int = None
    ) -> Dict:
        """Run full evaluation"""
        
        print(f"\n{'='*70}")
        print(f"EVALUATING ON {dataset_name.upper()}")
        print(f"{'='*70}")
        
        # Metrics
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        num_batches = 0
        
        # Router statistics
        router_stats = {
            'routing_ratios': [],
            'query_weights': [],
            'layer_ratios': [[] for _ in range(self.config['n_layers'])]
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
            
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Extract question mask
            if 'text' in batch:
                texts = batch['text']
                question_mask = question_extractor.extract_question_mask_from_ids(
                    input_ids.cpu(),
                    texts
                )
                question_mask = question_mask.to(self.device)
            else:
                question_mask = None
            
            # Forward pass
            loss, logits, router_outputs = self.model(
                input_ids,
                labels=labels,
                question_mask=question_mask,
                deterministic=True,
                return_router_outputs=True
            )
            
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
                
                # Query weights
                query_weight = router_outputs[0].get('query_weight', 0.0)
                router_stats['query_weights'].append(query_weight)
                
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
                    'acc': f"{total_correct / max(total_tokens, 1) * 100:.2f}%"
                })
        
        # Compute final metrics
        elapsed_time = time.time() - start_time
        
        avg_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        accuracy = total_correct / total_tokens * 100 if total_tokens > 0 else 0.0
        throughput = total_tokens_processed / elapsed_time
        
        # Router metrics
        avg_routing_ratio = np.mean(router_stats['routing_ratios']) if router_stats['routing_ratios'] else 0.0
        avg_query_weight = np.mean(router_stats['query_weights']) if router_stats['query_weights'] else 0.0
        layer_routing_ratios = [
            np.mean(ratios) if ratios else 0.0 
            for ratios in router_stats['layer_ratios']
        ]
        
        results = {
            'dataset': dataset_name,
            'num_samples': num_batches * dataloader.batch_size,
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'total_tokens': total_tokens,
            'correct_tokens': total_correct,
            'time_seconds': elapsed_time,
            'throughput_tokens_per_sec': throughput,
            'query_aware': {
                'avg_query_weight': avg_query_weight,
                'avg_routing_ratio': avg_routing_ratio,
                'target_ratio': self.config.get('target_ratio', 0.15),
                'layer_routing_ratios': layer_routing_ratios,
            },
            'model': {
                'num_parameters': self.model.get_num_params(),
                'num_layers': self.config['n_layers'],
                'd_model': self.config['d_model'],
            }
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Pretty print results"""
        print(f"\n{'='*70}")
        print("EVALUATION RESULTS")
        print(f"{'='*70}")
        
        print(f"\nDataset: {results['dataset']}")
        print(f"Samples: {results['num_samples']}")
        
        print(f"\n--- Language Modeling Metrics ---")
        print(f"Loss:       {results['loss']:.4f}")
        print(f"Perplexity: {results['perplexity']:.2f}")
        print(f"Accuracy:   {results['accuracy']:.2f}%")
        
        print(f"\n--- Query-Aware Routing ---")
        print(f"Avg query weight:  {results['query_aware']['avg_query_weight']:.3f}")
        print(f"Routing ratio:     {results['query_aware']['avg_routing_ratio']:.2%}")
        print(f"Target ratio:      {results['query_aware']['target_ratio']:.2%}")
        
        print(f"\nPer-layer routing ratios:")
        for i, ratio in enumerate(results['query_aware']['layer_routing_ratios']):
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
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Query-Aware HYDRA")
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True)
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='naturalqa',
                       choices=['narrativeqa', 'naturalqa', 'squad'])
    parser.add_argument('--max-samples', type=int, default=None)
    
    # Data loading
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Output
    parser.add_argument('--output', type=str, default=None)
    
    args = parser.parse_args()
    
    # Auto-generate output path
    if args.output is None:
        checkpoint_dir = Path(args.checkpoint).parent
        args.output = checkpoint_dir / f'eval_results_{args.dataset}.json'
    
    # Load model
    evaluator = QueryAwareEvaluator(args.checkpoint)
    
    # Load dataset
    print(f"\n{'='*70}")
    print("LOADING DATASET")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Max samples: {args.max_samples or 'all'}")
    
    _, val_loader, tokenizer = get_qa_dataloaders(
        dataset_name=args.dataset,
        tokenizer_name=evaluator.config.get('tokenizer_name', 'gpt2'),
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_train_samples=0,
        max_val_samples=args.max_samples,
    )
    
    # Create question extractor
    question_extractor = QuestionExtractor(tokenizer)
    
    print(f"✓ Loaded {len(val_loader)} batches")
    print(f"✓ Question extractor ready")
    print(f"{'='*70}\n")
    
    # Run evaluation
    results = evaluator.evaluate(
        val_loader,
        question_extractor,
        args.dataset,
        max_samples=args.max_samples
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save results
    evaluator.save_results(results, args.output)
    
    print("\n✓ Evaluation complete!")


if __name__ == '__main__':
    main()