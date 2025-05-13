# Speech Emotion Recognition (SER)

---
## Project Description

This project investigates how depth and attention mechanisms in LSTM-based architectures influence both the performance and interpretability of Speech Emotion Recognition (SER) systems. We explore four model variants: single- and dual-layer BiLSTMs, each with and without additive attention. For evaluation, we benchmark across studio-quality (RAVDESS) and noisy, conversational datasets (MELD). We additionally analyze the effect of speech duration, speaker attributes, and emotional intensity on model robustness.

---

## Model Architectures

* **BiLSTM**: Bidirectional Long Short-Term Memory network
* **Stacked BiLSTM**: Multiple layers of BiLSTM for deeper representations
* **LSTM with Additive Attention**: Incorporates attention mechanism to focus on important time steps
* **Stacked LSTM with Additive Attention**: Combines deep LSTM layers with attention for enhanced performance

---

## Project Structure

```bash
SER/
├── analysis/                 # Scripts or notebooks for analyzing results (e.g., accuracy trends, errors)
├── configs/                 # YAML configuration files for training and evaluation setups
├── dataset/                # Dataset storage and preprocessing logic (e.g., manifest generation, slicing)
├── mser/                   # Core module for the SER pipeline
│   ├── models/             # Model definitions: BiLSTM, Attention-LSTM, etc.
│   ├── optimizer/          # Optimizer setups and learning rate schedulers
│   ├── metrics/            # Evaluation metrics (accuracy, F1-score, confusion matrix logic)
│   ├── utils/              # Utility functions: argument parsing, logging, plotting
│   ├── data_utils/         # Dataset loading logic and collate functions for DataLoader
│   └── trainer.py          # Main training loop handling epochs, logging, checkpointing
├── duration_tuning/        # Scripts or configs related to tuning dataset duration thresholds
├── tuning/                 # Hyperparameter tuning scripts (e.g., with Optuna or grid search)
├── create_features.py      # Script to extract and save features (MFCC) from raw audio
├── extract_features.py     # Script for feature generation
├── train.py                # Main script to launch training using config files
├── eval.py                 # Script to evaluate a trained model on validation/test data
├── requirements.txt        # Python dependencies for the project (install with pip)
└── README.md               # Project documentation you're currently reading
```


---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/sanmati02/SER.git
   cd SER
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Training a Model on RAVDESS

- Ensure that `config/bi_lstm.yaml` is set to `train_list_features.txt` and `test_list_features.txt`. 
- Adjust the model name in `config/bi_lstm.yaml` to `BiLSTM`, `StackedLSTM`, `LSTMAdditiveAttention`, or `StackedLSTMAdditiveAttention`, 

Run: 

```bash
python CUDA_VISIBLE_DEVICES=0 python train.py --configs=configs/bi_lstm.yml
```

### Evaluating a Model (RAVDESS)

* Ensure that the `eval.py` file points to the correct model checkpoint. 
* Example path: models/StackedLSTMAdditiveAttention_CustomFeature/best_model/

Run:

```bash
python eval.py --configs=configs/bi_lstm.yml
```

### Fine-tuning a Model on MELD

- Download dataset from https://affective-meld.github.io/ 
- Extract .wav files using:
```bash
python extract_audios.py
```
- Ensure that `config/bi_lstm.yaml` is set to `MELD_data/train_reformatted_features.txt` and `MELD_data/test_reformatted_features.txt`. 
- Adjust the model name in `config/bi_lstm.yaml` to `BiLSTM` or `StackedLSTMAdditiveAttention`
- Set the path to the trained model

Run:

```bash
python fine_tune_meld.py
```

### Evaluating Model Confidence (any model)

* Ensure that the `eval_confidence.py` file points to the correct model checkpoint. 
* Example path: models/StackedLSTMAdditiveAttention_CustomFeature/best_model/

Run:

```bash
python eval_confidence.py --configs=configs/bi_lstm.yml

## Features

* **MFCC**: Mel-Frequency Cepstral Coefficients
* **Chroma**: Pitch class profiles
* **Spectral Contrast**: Difference between peaks and valleys in a spectrum
* **Zero-Crossing Rate**: Rate at which the signal changes sign
* **RMS Energy**: Root Mean Square Energy

---

## Evaluation

* Generates confusion matrices to visualize model performance across different emotion classes
* Calculates metrics such as accuracy, precision, recall, and F1-score
* For attention models, generates heatmaps for overall and class attention weights

 ---
 
 ## Acknowledgements & Inspiration

This project was inspired by the SpeechEmotionRecognition-Pytorch repository by yeyupiaoling. We referenced parts of their codebase — particularly in the data preprocessing pipeline and feature extraction setup — to help structure our initial implementation.

However, several key modifications were made to adapt the pipeline to our specific requirements:

- Integrated the MELD dataset for multi-turn dialogue emotion recognition.
- Rewrote the training loop and logging logic to support more flexible experimentation (duration tuning and hyperparameter tuning)
- Added new model architectures, including stacked BiLSTMs and attention-based LSTMs.
- Reworked feature extraction to preserve temporal structure (avoiding average pooling over time).
- Implemented additional evaluation metrics and visualizations for model interpretability.
- Added modules to support external testing and fine-tuning on MELD dataset 
