import argparse
import functools
import time
import os
from mser.trainer import MSERTrainer
from mser.trainer import MSERTrainer
from mser.utils.utils import add_arguments, print_arguments
# Directory where your trained models are saved
trained_model_dir = 'trained_models/meld'
model_names = os.listdir(trained_model_dir)

# Loop through each trained model
for model_name in model_names:
    # Construct the path to the best model of each experiment
    model_path = os.path.join(trained_model_dir, model_name, 'StackedLSTMAdditiveAttention_CustomFeature', 'best_model')
    
    if os.path.exists(model_path):
        print(f"Evaluating model: {model_name}")

        # Initialize the trainer for evaluation
        trainer = MSERTrainer(
            configs='configs/bi_lstm.yml',  # Path to your config
            use_gpu=True
        )
        
        # Evaluate the model using the evaluate_confidence method
        save_dir = os.path.join(trained_model_dir, model_name, 'eval')
        trainer.evaluate_confidence(
            resume_model=model_path,
            save_dir=save_dir,
            emotion_labels = ["Neutral", "Joy", "Sadness", "Anger", "Fear", "Disgust", "Surprise"]
        )
        print(f"Evaluation for {model_name} saved to {save_dir}")
    else:
        print(f"Model path not found for {model_name}. Skipping.")
