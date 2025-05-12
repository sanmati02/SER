# fine_tune_meld.py
from mser.trainer import MSERTrainer

# Path to config and pretrained model
config_path = "configs/bi_lstm.yml"  # updated for MELD paths
pretrained_model_path = "trained_models/vanilla/model.pth"  # path to RAVDESS pretrained checkpoint
save_model_path = "trained_models/meld/"
log_dir = "log/"

# Initialize trainer
trainer = MSERTrainer(
    configs=config_path,
    use_gpu=True
)

# Optional: Freeze LSTM layers if only fine-tuning the attention and classification head
# for param in trainer.model.lstm.parameters():
#     param.requires_grad = False

# Start training
trainer.train(
    save_model_path=save_model_path,
    log_dir=log_dir,
    pretrained_model=pretrained_model_path,
)

# Optionally evaluate after training
trainer.evaluate(
    resume_model=f"{save_model_path}/best_model/",
    save_dir="meld_eval/"
)
