from train_model import Trainer, TrainConfig

# Override config for quick test
config = TrainConfig()
config.d_model = 256
config.n_layers = 4
config.batch_size = 4
config.max_length = 128
config.num_epochs = 1
config.use_wandb = False
config.dataset_config = 'wikitext-2-raw-v1'  # Smaller dataset
config.output_dir = 'outputs/test'

print("Running quick training test...")
trainer = Trainer(config)
trainer.train()
print(" ==== Training test complete! ==== ")