import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import re

def plot_comparison(dataframes, labels, metric, title, ylabel, save_dir='plots'):
    plt.figure(figsize=(10, 6))
    for df, label in zip(dataframes, labels):
        max_epochs = len(df)
        plt.plot(range(1, max_epochs + 1), df[metric], label=f'max_duration={label} (Epochs: {max_epochs})')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Clean title to create filename
    filename = title.lower().replace(' ', '_') + '.png'
    filepath = os.path.join(save_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {filepath}")
    
    # Show the plot
    plt.show()
    plt.close()

def extract_duration(filename):
    """Extract duration number from filename like training_results_dur_X.csv"""
    match = re.search(r'training_results_dur_(\d+)', filename)
    if match:
        return match.group(1)
    return "unknown"

def main():
    # Get all CSV files in the current directory matching the pattern
    csv_files = sorted(glob.glob('training_results_dur_*.csv'))
    
    if len(csv_files) < 1:
        print("No matching CSV files found in the current directory.")
        print("Expected files named like: training_results_dur_1.csv, training_results_dur_2.csv, etc.")
        return
    
    # Read all CSV files into a list of DataFrames and extract durations
    dataframes = []
    durations = []
    for file in csv_files[:5]:  # Process up to 5 files
        df = pd.read_csv(file)
        dataframes.append(df)
        durations.append(extract_duration(file))
    
    # Create comparison plots for each metric
    metrics = [
        ('Train Loss', 'Training Loss Comparison', 'Loss'),
        ('Train Accuracy', 'Training Accuracy Comparison', 'Accuracy'),
        ('Eval Loss', 'Evaluation Loss Comparison', 'Loss'),
        ('Eval Accuracy', 'Evaluation Accuracy Comparison', 'Accuracy')
    ]
    
    for metric, title, ylabel in metrics:
        plot_comparison(dataframes, durations, metric, title, ylabel)

if __name__ == "__main__":
    main()