"""
Model Registry for HYDRA

This file defines all available model variants and their default configurations.
Each model variant has its own instantiation function.
"""

from enum import Enum
from pathlib import Path

from src.model.hybrid_model import HybridSSMTransformer
# Import other model variants when implemented
# from src.model.routed_model import RoutedHybridTransformer
# from src.model.hierarchical_model import HierarchicalHybridTransformer


class ModelType(Enum):
    """Available model variants"""
    BASELINE = "v1"
    ROUTED = "v2"
    HIERARCHICAL = "v3"


# Default config files for each model type
DEFAULT_CONFIGS = {
    ModelType.BASELINE: "configs/baseline.yaml",
    ModelType.ROUTED: "configs/routed.yaml",
    ModelType.HIERARCHICAL: "configs/hierarchical.yaml",
}


# Model descriptions
MODEL_DESCRIPTIONS = {
    ModelType.BASELINE: "Baseline hybrid SSM-Transformer with alternating layers",
    ModelType.ROUTED: "Dynamic token routing between SSM and attention blocks",
    ModelType.HIERARCHICAL: "Hierarchical state organization across layers",
}


# Model instantiation functions
def create_baseline_model(config):
    """Create v1 baseline model"""
    return HybridSSMTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        pattern=config.layer_pattern,
        n_heads=config.n_heads,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        dropout=config.dropout,
    )


def create_routed_model(config):
    """Create v2 routed model"""
    # TODO: Implement RoutedHybridTransformer
    # For now, use baseline with a note
    print("WARNING: Routed model not yet implemented, using baseline")
    return HybridSSMTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        pattern=config.layer_pattern,
        n_heads=config.n_heads,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        dropout=config.dropout,
    )
    # return RoutedHybridTransformer(
    #     vocab_size=config.vocab_size,
    #     d_model=config.d_model,
    #     n_layers=config.n_layers,
    #     n_heads=config.n_heads,
    #     d_state=config.d_state,
    #     routing_threshold=config.routing_threshold,
    #     router_temperature=config.router_temperature,
    #     dropout=config.dropout,
    # )


def create_hierarchical_model(config):
    """Create v3 hierarchical model"""
    # TODO: Implement HierarchicalHybridTransformer
    # For now, use baseline with a note
    print("WARNING: Hierarchical model not yet implemented, using baseline")
    return HybridSSMTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        pattern=config.layer_pattern,
        n_heads=config.n_heads,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        dropout=config.dropout,
    )
    # return HierarchicalHybridTransformer(
    #     vocab_size=config.vocab_size,
    #     d_model=config.d_model,
    #     n_layers=config.n_layers,
    #     n_heads=config.n_heads,
    #     d_state=config.d_state,
    #     hierarchy_levels=config.hierarchy_levels,
    #     state_compression_ratio=config.state_compression_ratio,
    #     dropout=config.dropout,
    # )


# Map model types to their creation functions
MODEL_CREATORS = {
    ModelType.BASELINE: create_baseline_model,
    ModelType.ROUTED: create_routed_model,
    ModelType.HIERARCHICAL: create_hierarchical_model,
}


def get_model_creator(model_type):
    """
    Get model creation function for a model type
    
    Args:
        model_type: ModelType enum
        
    Returns:
        Model creation function
    """
    return MODEL_CREATORS[model_type]


def create_model(model_type, config):
    """
    Create a model instance
    
    Args:
        model_type: ModelType enum
        config: Training configuration
        
    Returns:
        Model instance
    """
    creator_fn = get_model_creator(model_type)
    return creator_fn(config)


def get_model_type(model_name):
    """
    Get ModelType from string
    
    Args:
        model_name: String like "v1", "v2", "v3", "baseline", "routed", or "hierarchical"
        
    Returns:
        ModelType enum
    """
    model_name = model_name.lower()
    
    # Map various names to ModelType
    name_mapping = {
        'v1': ModelType.BASELINE,
        'baseline': ModelType.BASELINE,
        'v2': ModelType.ROUTED,
        'routed': ModelType.ROUTED,
        'v3': ModelType.HIERARCHICAL,
        'hierarchical': ModelType.HIERARCHICAL,
    }
    
    if model_name not in name_mapping:
        available = list(name_mapping.keys())
        raise ValueError(f"Unknown model type: {model_name}. Available: {available}")
    
    return name_mapping[model_name]


def get_default_config(model_type):
    """
    Get default config file path for a model type
    
    Args:
        model_type: ModelType enum
        
    Returns:
        Path to default config file
    """
    return DEFAULT_CONFIGS[model_type]


def list_models():
    """Print all available models"""
    print("\nAvailable Models:")
    print("=" * 60)
    for model_type in ModelType:
        config_file = DEFAULT_CONFIGS[model_type]
        description = MODEL_DESCRIPTIONS[model_type]
        print(f"{model_type.value:15s} | {description}")
        print(f"{'':15s} | Default config: {config_file}")
        print("-" * 60)