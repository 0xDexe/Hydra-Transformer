"""
Train routed HYDRA model from YAML configuration

Usage:
    python tests/train_from_config.py --config config/routed_hybrid.yaml
    python tests/train_from_config.py --config config/routed_hybrid.yaml --resume outputs/routed-v1/checkpoint_latest.pt
"""

import yaml
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train_routed import RoutedTrainer, RoutedTrainConfig


def load_config_from_yaml(yaml_path):
    """Load configuration from YAML file"""
    with open(yaml_path, 'r') as f:
        cfg_dict = yaml.safe_load(f)
    
    # Create config object
    config = RoutedTrainConfig()
    
    # Update from YAML sections
    if 'model' in cfg_dict:
        for k, v in cfg_dict['model'].items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    if 'router' in cfg_dict:
        for k, v in cfg_dict['router'].items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    if 'curriculum' in cfg_dict:
        for k, v in cfg_dict['curriculum'].items():
            # Map YAML keys to config keys
            if k == 'warmup_steps':
                config.router_warmup_steps = v
            elif k == 'heuristic_type':
                config.curriculum_heuristic = v
    
    if 'router_loss' in cfg_dict:
        loss_cfg = cfg_dict['router_loss']
        config.router_loss_weight = loss_cfg.get('total_weight', 0.01)
        config.load_loss_weight = loss_cfg.get('load_balance', 0.01)
        config.entropy_loss_weight = loss_cfg.get('entropy', 0.01)
        config.variance_loss_weight = loss_cfg.get('variance', 0.01)
        config.position_inv_weight = loss_cfg.get('position_invariance', 0.005)
        config.diversity_loss_weight = loss_cfg.get('layer_diversity', 0.005)
    
    if 'data' in cfg_dict:
        for k, v in cfg_dict['data'].items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    if 'training' in cfg_dict:
        for k, v in cfg_dict['training'].items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    if 'logging' in cfg_dict:
        for k, v in cfg_dict['logging'].items():
            if hasattr(config, k):
                setattr(config, k, v)
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Train routed HYDRA model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/routed_hybrid.yaml',
        help='Path to YAML config file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Override run name'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable wandb logging'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config_from_yaml(args.config)
    
    # Apply overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.run_name:
        config.run_name = args.run_name
    if args.no_wandb:
        config.use_wandb = False
    
    # Create trainer
    trainer = RoutedTrainer(config)
    
    # Resume if checkpoint provided
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()