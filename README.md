# Quick Start Guide

---

## Step 1: Install Dependencies (2 minutes)

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install all other dependencies
pip install -r requirements.txt
```

---

## Step 2: Test Installation (1 minute)

```bash
# Test the model
python tests/test_model.py
```

Expected output:
```
MODEL FORWARD PASS TESTING
Model created with 21.87M parameters
Forward pass successful
All tests passed!
```

---

## Step 3: Quick Training Test (2 minutes)

```bash
# Run a quick training test
python tests/test_training.py
```

This will:
- Load a small dataset (WikiText-2)
- Train for 1 epoch
- Save checkpoints
- Create training logs

---

## Step 4: Full Training (Optional)

```bash
# Train with the baseline configuration
python tests/train_from_config.py --config config/baseline.yaml
```

This will:
- Train on WikiText-103
- Save checkpoints every 2 hours
- Log all losses
- Track significant changes

---

## Step 5: Monitor Training (Optional)

While training, in another terminal:

```bash
# Monitor training progress
python scripts/monitor.py outputs/baseline-v1
```

This generates plots and statistics

---

## File Locations

After testing/training:

```
outputs/
└── baseline-v1/  (or test/)
    ├── training_log.jsonl        # Complete training log
    ├── checkpoint_epoch_{epoch}.pt      # Latet epoch model

```

---

## Training Log Format

Check your logs:
```bash
tail -f outputs/baseline-v1/training_log.jsonl
```

Example entry:
```json
{"timestamp": "2025-01-15T10:30:45", "step": 100, "epoch": 1, 
 "split": "train", "loss": 4.567, "lr": 0.0003}
```

---
## Additional Features

### Resume Training Automatically 

1. Resume from a specific checkpoint:

```
python  scripts/resume_training.py \
    --checkpoint outputs/baseline/checkpoint_epoch_5.pt \
    --num_epochs 10
```

2. Auto-detect latest checkpoint:

```
python  scripts/resume_training.py \
    --checkpoint_dir outputs/baseline \
    --num_epochs 5
```

3. Resume with different output directory:

```
python scripts/resume_training.py \
    --checkpoint outputs/baseline/checkpoint_epoch_5.pt \
    --num_epochs 10 \
    --output_dir outputs/baseline_continued
```


4. Just continue with original config:
```
python  scripts/resume_training.py \
    --checkpoint outputs/baseline/checkpoint_epoch_5.pt

```



---
## Common Issues

### 1. Import Error
```
ModuleNotFoundError: No module named 'src'
```
**Fix**: Run from project root, not subdirectories

### 2. CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Fix**: Edit `config/baseline.yaml`:
```yaml
data:
  batch_size: 4  # Reduce from 8
  max_length: 512  # Reduce from 1024
```

### 3. FlashAttention Not Found
```
ImportError: cannot import name 'flash_attn_func'
```
**Fix**: 
```bash
pip install flash-attn --no-build-isolation
```

---


## Understanding the Output

### Console Output
```
==================================================
Epoch 1/20
==================================================
Epoch 1: 100%|████████| 1000/1000 [10:00<00:00]
Train loss: 4.567
Val loss: 4.123, Perplexity: 61.23
Saved best model with val_loss: 4.123
```

### Log File
```bash
# View recent logs
tail -20 outputs/baseline-v1/training_log.jsonl

# Count significant changes
grep "significant_change" outputs/baseline-v1/training_log.jsonl | wc -l

# Get validation losses
grep "validation" outputs/baseline-v1/training_log.jsonl
```

---

## Customization

### Change Model Size

In `config/baseline.yaml`:
```yaml
model:
  d_model: 512    # Smaller model (from 768)
  n_layers: 6     # Fewer layers (from 12)
```

### Change Checkpoint Interval

In `src/train.py`, line ~150:
```python
self.checkpoint_interval_seconds = 1 * 60 * 60  # 1 hour (from 2)
```

### Change Layer Pattern

In `config/baseline.yaml`:
```yaml
model:
  layer_pattern: 'ssm_first'  # Try different patterns
```

---

## Debug Mode

Run debug script to inspect model:
```bash
python scripts/debug_model.py
```

Output:
- Layer structure
- Parameter breakdown
- Memory footprint
- Shape tests

---

## WandB Integration

To enable WandB logging:

1. Install: `pip install wandb`
2. Login: `wandb login`
3. In `config/baseline.yaml`:
```yaml
logging:
  use_wandb: true
  project_name: 'my-project'
  run_name: 'my-experiment'
```

---

**Happy Training!**
