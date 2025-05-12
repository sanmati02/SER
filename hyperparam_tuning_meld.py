
import optuna
from mser.trainer import MSERTrainer
import torch
import os

# Define objective function to optimize
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    batch_size = trial.suggest_int('batch_size', 16, 64, step=16)
    weight_decay = trial.suggest_loguniform(1e-6, 1e-5, 1e-4)
    epochs = trial.suggest_int('epochs', 10, 50)
    
    # Paths for config and pretrained model
    config_path = "configs/bi_lstm.yml"
    pretrained_model_path = "trained_models/vanilla/model.pth"
    save_model_path = "trained_models/meld/"
    log_dir = "log/"
    
    # Initialize trainer
    trainer = MSERTrainer(
        configs=config_path,
        use_gpu=True
    )

    # Update the trainer config with the trial hyperparameters
    trainer.model.config['learning_rate'] = learning_rate
    trainer.model.config['batch_size'] = batch_size
    trainer.model.config['weight_decay'] = weight_decay
    trainer.model.config['epochs'] = epochs

    # Start training
    trainer.train(
        save_model_path=save_model_path,
        log_dir=log_dir,
        pretrained_model=pretrained_model_path,
    )

    # Evaluate the model after training
    results = trainer.evaluate(
        resume_model=f"{save_model_path}/best_model/",
        save_dir="meld_eval/"
    )
    
    # Return the validation accuracy or another performance metric
    return results['accuracy']

# Create Optuna study
study = optuna.create_study(direction='maximize')  # Maximizing accuracy
study.optimize(objective, n_trials=20)  # Number of trials to run

# Print the best hyperparameters and their result
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_trial.value}")
print("Best hyperparameters:")
for key, value in study.best_trial.params.items():
    print(f"{key}: {value}")
