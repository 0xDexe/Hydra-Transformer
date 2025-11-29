import yaml
from pathlib import Path
from src.train import Trainer, TrainConfig

def load_config(config_path):
    """Load config from YAML"""
    with open(config_path) as f:
        cfg_dict = yaml.safe_load(f)
    
    config = TrainConfig()
    
    # Update from yaml
    for k, v in cfg_dict['model'].items():
        setattr(config, k, v)
    for k, v in cfg_dict['data'].items():
        setattr(config, k, v)
    for k, v in cfg_dict['training'].items():
        setattr(config, k, v)
    for k, v in cfg_dict['logging'].items():
        setattr(config, k, v)
    
    return config

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/baseline.yaml')
    #TOKEN_ROUTING_CHANGE
    # parser.add_argument('--config', type=str, default='configs/tr_config.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    trainer = Trainer(config)
    trainer.train()