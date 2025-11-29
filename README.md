# HYDRA: Hybrid Yielding Dynamic Routing Architecture

A modular PyTorch implementation of hybrid SSM-Transformer architectures with support for multiple model variants.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/0xDexe/Hydra-Transformer.git
cd Hydra-Transformer

# Install dependencies
pip install torch transformers datasets mamba-ssm flash-attn wandb pyyaml tqdm
```

### List Available Models

```bash
python train.py --list
```

Output:
```
Available Models:
============================================================
v1              | Baseline hybrid SSM-Transformer with alternating layers
                | Default config: configs/baseline.yaml
------------------------------------------------------------
v2              | Dynamic token routing between SSM and attention blocks
                | Default config: configs/routed.yaml
------------------------------------------------------------
v3              | Hierarchical state organization across layers
                | Default config: configs/hierarchical.yaml
------------------------------------------------------------
```

### Train a Model

#### Using Default Configuration
```bash
# Train baseline model (v1)
python train.py --model v1

# Train routed model (v2)
python train.py --model v2

# Train hierarchical model (v3)
python train.py --model v3
```

You can also use descriptive names:
```bash
python train.py --model baseline
python train.py --model routed
python train.py --model hierarchical
```

#### Using Custom Configuration
```bash
python train.py --model v2 --config configs/my_custom_config.yaml
```

### Resume Training

```bash
# Resume from specific checkpoint
python resume_training.py \
    --checkpoint outputs/baseline/checkpoint_epoch_5.pt \
    --num_epochs 10

# Auto-detect latest checkpoint
python resume_training.py \
    --checkpoint_dir outputs/baseline \
    --num_epochs 5

# Resume with different output directory
python resume_training.py \
    --checkpoint outputs/baseline/checkpoint_epoch_5.pt \
    --num_epochs 10 \
    --output_dir outputs/baseline_continued
```

## Project Structure

```
Hydra-Transformer/
├── models.py                       # Model registry
├── train.py            # Main training script
├── resume_training.py              # Resume training script
├── configs/
│   ├── baseline.yaml              # v1 default config
│   ├── routed.yaml                # v2 default config
│   └── hierarchical.yaml          # v3 default config
├── src/
│   ├── train_model.py                   # Model-agnostic trainer
│   ├── model/
│   │   ├── hybrid_model.py        # v1 implementation
│   │   ├── routed_model.py        # v2 implementation (TODO)
│   │   └── hierarchical_model.py  # v3 implementation (TODO)
│   └── data/
│       └── dataset.py             # Data loading utilities
└── outputs/                       # Training outputs
    └── {experiment_name}/
        ├── checkpoint_epoch_0.pt
        ├── checkpoint_epoch_1.pt
        └── training_log.jsonl
```

## Model Variants

### v1 - Baseline
- **Architecture:** Standard hybrid SSM-Transformer
- **Features:** Alternating SSM and attention layers
- **Best for:** General-purpose language modeling tasks
- **Config:** `configs/baseline.yaml`

### v2 - Routed (Coming Soon)
- **Architecture:** Dynamic token routing
- **Features:** Routes ~15% critical tokens to attention, rest to SSM
- **Best for:** Long contexts, computational efficiency
- **Config:** `configs/routed.yaml`

### v3 - Hierarchical (Coming Soon)
- **Architecture:** Hierarchical state organization
- **Features:** Multi-level state compression across layers
- **Best for:** Very long sequences, hierarchical processing
- **Config:** `configs/hierarchical.yaml`

## Configuration

### Config File Format

```yaml
# Model architecture parameters
model:
  d_model: 768                    # Model dimension
  n_layers: 12                    # Number of layers
  layer_pattern: 'alternating'    # Layer arrangement pattern
  n_heads: 12                     # Number of attention heads
  d_state: 16                     # SSM state dimension
  d_conv: 4                       # SSM convolution size
  expand: 2                       # SSM expansion factor
  dropout: 0.1                    # Dropout rate
  vocab_size: 50257              # Vocabulary size

# Data configuration
data:
  dataset_name: 'wikitext'
  dataset_config: 'wikitext-103-v1'
  tokenizer_name: 'gpt2'
  max_length: 1024
  batch_size: 8
  num_workers: 4

# Training parameters
training:
  num_epochs: 20
  learning_rate: 0.0003
  weight_decay: 0.01
  grad_clip: 1.0
  use_mixed_precision: true       # Enable mixed precision training

# Logging configuration
logging:
  use_wandb: true                 # Enable Weights & Biases logging
  project_name: 'hydra-transformer'
  run_name: 'v1-baseline'
  output_dir: 'outputs/v1-baseline'
```

### Creating Custom Configurations

```bash
# Copy a default config
cp configs/baseline.yaml configs/my_experiment.yaml

# Edit the config file
nano configs/my_experiment.yaml

# Train with custom config
python train.py --model v1 --config configs/my_experiment.yaml
```

## 📊 Training Outputs

### Checkpoints
Saved to `{output_dir}/checkpoint_epoch_{n}.pt`

Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Scheduler state dict
- Gradient scaler state (if using mixed precision)
- Training configuration
- Training time
- Validation loss

### Logs
Saved to `{output_dir}/training_log.jsonl`

Each line contains:
```json
{
  "timestamp": "2025-11-29T20:00:00",
  "epoch": 0,
  "train_loss": 2.5432,
  "val_loss": 2.4123,
  "val_perplexity": 11.16,
  "epoch_time_seconds": 45.2,
  "learning_rate": 0.0003
}
```

### Console Output
```
============================================================
EPOCH 0 SUMMARY
============================================================
Train Loss:       2.5432
Val Loss:         2.4123
Val Perplexity:   11.16
Epoch Time:       45.2s
Learning Rate:    3.00e-04
============================================================
```

## Advanced Usage

### Mixed Precision Training

Mixed precision is enabled by default. To disable:

```yaml
# In your config file
training:
  use_mixed_precision: false
```

### Weights & Biases Integration

Enable W&B logging in your config:

```yaml
logging:
  use_wandb: true
  project_name: 'my-project'
  run_name: 'my-experiment'
```

Or disable it:

```yaml
logging:
  use_wandb: false
```

### Custom Datasets

Modify the data section in your config:

```yaml
data:
  dataset_name: 'your_dataset'
  dataset_config: 'your_config'
  tokenizer_name: 'your_tokenizer'
  max_length: 2048
  batch_size: 4
```

## 🐛 Debugging

### Test Model Forward Pass

```python
from models import create_model, ModelType
from src.train import TrainConfig

config = TrainConfig()
model = create_model(ModelType.BASELINE, config)

import torch
input_ids = torch.randint(0, config.vocab_size, (2, 128))
labels = torch.randint(0, config.vocab_size, (2, 128))

loss, logits = model(input_ids, labels=labels)
print(f"Loss: {loss.item():.4f}")
print(f"Logits shape: {logits.shape}")
```

### Check Model Parameters

```bash
python scripts/debug_model.py
```

## Performance Tips

1. **Batch Size:** Increase for better GPU utilization (if memory allows)
2. **Mixed Precision:** Keep enabled for faster training and lower memory
3. **Gradient Accumulation:** For effective larger batch sizes
4. **Number of Workers:** Set to number of CPU cores for data loading

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Adding new model variants
- Creating configuration files
- Code style and testing

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Mamba SSM implementation: [state-spaces/mamba](https://github.com/state-spaces/mamba)
- FlashAttention: [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- Inspired by hybrid architectures like Jamba and Zamba

## Contact

- GitHub Issues: [Report bugs or request features](https://github.com/0xDexe/Hydra-Transformer/issues)
- Discussions: [Ask questions](https://github.com/0xDexe/Hydra-Transformer/discussions)

---

**Happy Training! 🚀**
