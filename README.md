# 1. Test the model
python -m tests.test_model

# 2. Quick training test (5 min)
python scripts/test_training.py

# 3. Check model size
python scripts/debug_model.py

# 4. Start full training
python scripts/train_from_config.py --config configs/baseline.yaml

# 5. Monitor in another terminal
watch -n 60 python scripts/monitor.py outputs/baseline-v1

# 6. Check wandb
# Go to wandb.ai and look at your run