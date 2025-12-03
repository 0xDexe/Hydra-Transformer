# Contributing to HYDRA

Thank you for your interest in contributing to HYDRA! This guide will help you add new model variants, configurations, and features.

## 🎯 Table of Contents

- [Adding a New Model Variant](#adding-a-new-model-variant)
- [Creating Configuration Files](#creating-configuration-files)
- [Mandatory Configuration Attributes](#mandatory-configuration-attributes)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Your Changes](#testing-your-changes)

---

## Adding a New Model Variant

Follow these steps to add a new model variant (e.g., v4, v5):

### Step 1: Implement the Model Class

Create a new file in `src/model/`:

```python
# src/model/my_new_model.py

import torch
import torch.nn as nn

class MyNewModel(nn.Module):
    """
    Description of your model architecture
    """
    
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        dropout=0.1,
        # Add your custom parameters here
        custom_param1=None,
        custom_param2=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Build your architecture here
        self.embedding = nn.Embedding(vocab_size, d_model)
        # ... your layers
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids, labels=None):
        """
        REQUIRED: Must return (loss, logits)
        
        Args:
            input_ids: (batch, seq_len) token IDs
            labels: (batch, seq_len) labels for loss calculation
            
        Returns:
            loss: scalar tensor (if labels provided)
            logits: (batch, seq_len, vocab_size) predictions
        """
        # Forward pass
        x = self.embedding(input_ids)
        # ... your model logic
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )
        
        return loss, logits
    
    # OPTIONAL: Add these methods for better logging
    def get_num_params(self, non_embedding=False):
        """Return number of parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embedding.weight.numel()
        return n_params
    
    def get_layer_info(self):
        """Return list of layer descriptions"""
        return [f"Layer {i}: CustomLayer" for i in range(self.n_layers)]
    
    # OPTIONAL: Custom optimizer configuration
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """Return configured optimizer"""
        # Separate parameters with/without weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        from torch.optim import AdamW
        optimizer = AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
```

### Step 2: Register in `models.py`

Add your model to the registry:

```python
# At the top of models.py
from src.model.my_new_model import MyNewModel  # Add this import

# 1. Add to ModelType enum
class ModelType(Enum):
    BASELINE = "v1"
    ROUTED = "v2"
    HIERARCHICAL = "v3"
    MY_NEW_MODEL = "v4"  # ← Add this

# 2. Add default config path
DEFAULT_CONFIGS = {
    ModelType.BASELINE: "configs/baseline.yaml",
    ModelType.ROUTED: "configs/routed.yaml",
    ModelType.HIERARCHICAL: "configs/hierarchical.yaml",
    ModelType.MY_NEW_MODEL: "configs/my_new_model.yaml",  # ← Add this
}

# 3. Add description
MODEL_DESCRIPTIONS = {
    ModelType.BASELINE: "Baseline hybrid SSM-Transformer with alternating layers",
    ModelType.ROUTED: "Dynamic token routing between SSM and attention blocks",
    ModelType.HIERARCHICAL: "Hierarchical state organization across layers",
    ModelType.MY_NEW_MODEL: "Brief description of your model",  # ← Add this
}

# 4. Create the model creation function
def create_my_new_model(config):
    """Create v4 model"""
    return MyNewModel(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        dropout=config.dropout,
        # Add your custom parameters
        custom_param1=getattr(config, 'custom_param1', None),
        custom_param2=getattr(config, 'custom_param2', None),
    )

# 5. Register the creation function
MODEL_CREATORS = {
    ModelType.BASELINE: create_baseline_model,
    ModelType.ROUTED: create_routed_model,
    ModelType.HIERARCHICAL: create_hierarchical_model,
    ModelType.MY_NEW_MODEL: create_my_new_model,  # ← Add this
}
```

### Step 3: Create Configuration File

See [Creating Configuration Files](#creating-configuration-files) section below.

### Step 4: Test Your Model

```bash
# List models (should show your new model)
python train_from_config.py --list

# Test training
python train_from_config.py --model v4 --config configs/my_new_model.yaml
```

---

## Creating Configuration Files

Create a new YAML file in `configs/`:

### Template: `configs/my_new_model.yaml`

```yaml
# Model architecture parameters
model:
  # MANDATORY - Core model parameters
  d_model: 768                    # Model dimension
  n_layers: 12                    # Number of layers
  n_heads: 12                     # Number of attention heads
  dropout: 0.1                    # Dropout rate
  vocab_size: 50257              # Vocabulary size (will be auto-updated)
  
  # OPTIONAL - Standard parameters (if your model uses them)
  layer_pattern: 'alternating'    # Layer arrangement
  d_state: 16                     # SSM state dimension
  d_conv: 4                       # Convolution size
  expand: 2                       # Expansion factor
  
  # CUSTOM - Your model-specific parameters
  custom_param1: 0.5              # Your custom parameter
  custom_param2: true             # Another custom parameter

# Data configuration
data:
  # MANDATORY - Dataset parameters
  dataset_name: 'wikitext'
  dataset_config: 'wikitext-103-v1'
  tokenizer_name: 'gpt2'
  max_length: 1024
  batch_size: 8
  num_workers: 4

# Training parameters
training:
  # MANDATORY - Training hyperparameters
  num_epochs: 20
  learning_rate: 0.0003
  weight_decay: 0.01
  grad_clip: 1.0
  
  # OPTIONAL - Training features
  use_mixed_precision: true       # Enable mixed precision

# Logging configuration
logging:
  # MANDATORY - Logging parameters
  use_wandb: true
  project_name: 'hydra-transformer'
  run_name: 'v4-my-new-model'
  output_dir: 'outputs/v4-my-new-model'
```

---

## ✅ Mandatory Configuration Attributes

### Model Section (Required)
```yaml
model:
  d_model: 768          # int - Model hidden dimension
  n_layers: 12          # int - Number of layers
  n_heads: 12           # int - Number of attention heads
  dropout: 0.1          # float - Dropout probability
  vocab_size: 50257     # int - Vocabulary size
```

### Data Section (Required)
```yaml
data:
  dataset_name: 'wikitext'           # str - HuggingFace dataset name
  dataset_config: 'wikitext-103-v1'  # str - Dataset configuration
  tokenizer_name: 'gpt2'             # str - HuggingFace tokenizer name
  max_length: 1024                   # int - Maximum sequence length
  batch_size: 8                      # int - Batch size
  num_workers: 4                     # int - DataLoader workers
```

### Training Section (Required)
```yaml
training:
  num_epochs: 20         # int - Number of training epochs
  learning_rate: 0.0003  # float - Learning rate
  weight_decay: 0.01     # float - Weight decay
  grad_clip: 1.0         # float - Gradient clipping threshold
```

### Logging Section (Required)
```yaml
logging:
  use_wandb: true                        # bool - Enable W&B
  project_name: 'hydra-transformer'      # str - W&B project name
  run_name: 'my-experiment'              # str - W&B run name
  output_dir: 'outputs/my-experiment'    # str - Output directory
```

### Optional Attributes

You can add custom attributes to any section:

```yaml
model:
  # Standard attributes
  d_model: 768
  
  # Your custom attributes
  my_custom_param: 42
  another_param: true
```

Access them in your model creation function:

```python
def create_my_model(config):
    return MyModel(
        d_model=config.d_model,
        my_custom_param=getattr(config, 'my_custom_param', 42),  # with default
    )
```

---

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints where helpful

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1, param2):
    """
    Brief description of function.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    pass
```

### Model Implementation

**Required methods:**

```python
def forward(self, input_ids, labels=None):
    """Must return (loss, logits)"""
    pass
```

**Recommended methods:**

```python
def get_num_params(self, non_embedding=False):
    """Return parameter count"""
    pass

def get_layer_info(self):
    """Return layer descriptions"""
    pass

def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
    """Return optimizer"""
    pass
```

### Configuration Files

- Use lowercase with underscores: `my_config.yaml`
- Group related parameters
- Add comments for clarity
- Include default values where appropriate

---

## 🧪 Testing Your Changes

### 1. Quick Test

```bash
# Test model creation
python -c "
from models import create_model, ModelType
from src.train import TrainConfig

config = TrainConfig()
model = create_model(ModelType.MY_NEW_MODEL, config)
print('Model created successfully!')
print(f'Parameters: {model.get_num_params() / 1e6:.2f}M')
"
```

### 2. Forward Pass Test

```python
# test_my_model.py
import torch
from models import create_model, ModelType
from src.train import TrainConfig

config = TrainConfig()
config.vocab_size = 1000
model = create_model(ModelType.MY_NEW_MODEL, config)

# Test forward pass
batch_size = 2
seq_len = 128
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

loss, logits = model(input_ids, labels=labels)

assert loss is not None, "Loss should not be None"
assert logits.shape == (batch_size, seq_len, config.vocab_size)
print("✓ Forward pass test passed!")
```

### 3. Training Test

```bash
# Quick training test (1 epoch, small dataset)
python train_from_config.py \
    --model v4 \
    --config configs/test_config.yaml
```

Create `configs/test_config.yaml`:

```yaml
model:
  d_model: 256
  n_layers: 4
  n_heads: 8
  dropout: 0.1
  vocab_size: 50257

data:
  dataset_name: 'wikitext'
  dataset_config: 'wikitext-2-raw-v1'  # Smaller dataset
  tokenizer_name: 'gpt2'
  max_length: 128
  batch_size: 4
  num_workers: 2

training:
  num_epochs: 1
  learning_rate: 0.0003
  weight_decay: 0.01
  grad_clip: 1.0
  use_mixed_precision: false

logging:
  use_wandb: false
  project_name: 'test'
  run_name: 'test-run'
  output_dir: 'outputs/test'
```

---

## Checklist for New Models

- [ ] Model implementation in `src/model/my_new_model.py`
- [ ] `forward()` method returns `(loss, logits)`
- [ ] Import added to `models.py`
- [ ] Enum variant added to `ModelType`
- [ ] Default config path added to `DEFAULT_CONFIGS`
- [ ] Description added to `MODEL_DESCRIPTIONS`
- [ ] Creation function implemented
- [ ] Creation function added to `MODEL_CREATORS`
- [ ] Configuration file created in `configs/`
- [ ] All mandatory config attributes present
- [ ] Forward pass test passes
- [ ] Training test completes successfully
- [ ] Code follows style guidelines
- [ ] Docstrings added

---

## Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/my-new-model`
3. **Make your changes** following this guide
4. **Test thoroughly** using the tests above
5. **Commit with clear messages**: `git commit -m "Add v4 model with custom routing"`
6. **Push to your fork**: `git push origin feature/my-new-model`
7. **Open a Pull Request** with:
   - Clear description of changes
   - Results of tests
   - Example usage

---

## Tips

1. **Start Simple**: Begin with a minimal implementation, then add features
2. **Copy Existing Models**: Use `hybrid_model.py` as a template
3. **Test Incrementally**: Test each component before moving to the next
4. **Document Everything**: Clear docs help others understand your work
5. **Ask Questions**: Open a discussion if you're unsure about anything

---

## Examples

### Example: Adding a Simple Variant

Let's say you want to add a model that's just the baseline with different parameters:

```python
# models.py

class ModelType(Enum):
    # ... existing
    LARGE_BASELINE = "v4"

DEFAULT_CONFIGS = {
    # ... existing
    ModelType.LARGE_BASELINE: "configs/large_baseline.yaml",
}

def create_large_baseline_model(config):
    """Create v4 large baseline model"""
    from src.model.hybrid_model import HybridSSMTransformer
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

MODEL_CREATORS = {
    # ... existing
    ModelType.LARGE_BASELINE: create_large_baseline_model,
}
```

Then create `configs/large_baseline.yaml` with larger dimensions.

---

## Questions?

- **Issues**: [GitHub Issues](https://github.com/0xDexe/Hydra-Transformer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/0xDexe/Hydra-Transformer/discussions)

Thank you for contributing to HYDRA! 🚀
