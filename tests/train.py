"""
Train from Config Script

Train HYDRA models with pre-configured or custom settings.

Usage:
    # Use default config for a model
    python train_from_config.py --model v1
    python train_from_config.py --model baseline
    
    # Use custom config
    python train_from_config.py --model v2 --config my_custom_config.yaml
    
    # List available models
    python train_from_config.py --list
"""

import argparse
import yaml
from pathlib import Path

from src.models import ModelType, get_model_type, get_default_config, list_models, create_model
from train_model import Trainer, TrainConfig


def load_config_from_yaml(yaml_path):
    """
    Load training configuration from YAML file
    
    Args:
        yaml_path: Path to YAML config file
        
    Returns:
        TrainConfig object
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")
    
    print(f"Loading config from: {yaml_path}")
    
    with open(yaml_path) as f:
        config_dict = yaml.safe_load(f)
    
    # Create base config
    config = TrainConfig()
    
    # Update with values from YAML
    # Model parameters
    if 'model' in config_dict:
        for key, value in config_dict['model'].items():
            setattr(config, key, value)
    
    # Data parameters
    if 'data' in config_dict:
        for key, value in config_dict['data'].items():
            setattr(config, key, value)
    
    # Training parameters
    if 'training' in config_dict:
        for key, value in config_dict['training'].items():
            setattr(config, key, value)
    
    # Logging parameters
    if 'logging' in config_dict:
        for key, value in config_dict['logging'].items():
            setattr(config, key, value)
    
    return config


def train_model(model_type, config_path=None):
    """
    Train a model with specified configuration
    
    Args:
        model_type: ModelType enum
        config_path: Optional path to custom config file
        
    Returns:
        Trained trainer object
    """
    # Determine which config to use
    if config_path is None:
        config_path = get_default_config(model_type)
        print(f"\nUsing default config for {model_type.value}")
    else:
        print(f"\nUsing custom config: {config_path}")
    
    # Load config
    config = load_config_from_yaml(config_path)
    
    # Print configuration summary
    print(f"\n{'='*60}")
    print("CONFIGURATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model variant: {model_type.value}")
    print(f"Config file: {config_path}")
    print(f"\nModel Architecture:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    if hasattr(config, 'layer_pattern'):
        print(f"  pattern: {config.layer_pattern}")
    print(f"  n_heads: {config.n_heads}")
    print(f"\nTraining:")
    print(f"  epochs: {config.num_epochs}")
    print(f"  batch_size: {config.batch_size}")
    print(f"  learning_rate: {config.learning_rate}")
    print(f"  mixed_precision: {config.use_mixed_precision}")
    print(f"\nData:")
    print(f"  dataset: {config.dataset_name}/{config.dataset_config}")
    print(f"  max_length: {config.max_length}")
    print(f"\nOutput:")
    print(f"  output_dir: {config.output_dir}")
    print(f"  wandb: {config.use_wandb}")
    print(f"{'='*60}\n")
    
    # Create model using models.py
    print(f"Creating model: {model_type.value}")
    model = create_model(model_type, config)
    print(f"Model created: {type(model).__name__}\n")
    
    # Create trainer with the model
    trainer = Trainer(config, model=model)
    
    # Train
    trainer.train()
    
    return trainer


def main():
    parser = argparse.ArgumentParser(
        description='Train HYDRA models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train baseline model with default config
  python train_from_config.py --model v1
  
  # Train routed model with custom config
  python train_from_config.py --model v2 --config configs/my_config.yaml
  
  # List available models
  python train_from_config.py --list
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Model variant (v1/baseline, v2/routed, v3/hierarchical)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom config YAML file (optional, uses default if not specified)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available models and exit'
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        list_models()
        return
    
    # Validate model argument
    if not args.model:
        parser.error("--model is required (or use --list to see available models)")
    
    # Get model type
    try:
        model_type = get_model_type(args.model)
    except ValueError as e:
        parser.error(str(e))
    
    # Train the model
    train_model(model_type, args.config)


if __name__ == '__main__':
    main()