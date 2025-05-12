import numpy as np
import torch

def collate_fn(batch):
    # Sort the batch by the length of the input tensor in descending order
    # This helps with padding and potential use in RNNs (e.g., with pack_padded_sequence)
    batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)

    # Determine if the input is a sequence (2D: [T, F]) or a single feature vector (1D: [F])
    is_sequence = batch[0][0].ndim == 2

    batch_size = len(batch)
    input_lens_ratio = []  # Stores the ratio of each input's length to the max length
    labels = []            # Ground truth labels for each input
    paths = []             # File paths or identifiers for each sample

    if is_sequence:
        # If input is a sequence, extract max sequence length and feature dimension
        max_seq_len = batch[0][0].shape[0]
        feat_dim = batch[0][0].shape[1]

        # Initialize input tensor with zeros (shape: [batch_size, max_seq_len, feat_dim])
        inputs = np.zeros((batch_size, max_seq_len, feat_dim), dtype='float32')

        for i, (tensor, label, path) in enumerate(batch):
            seq_len = tensor.shape[0]
            # Copy actual sequence data into the input tensor
            inputs[i, :seq_len, :] = tensor
            # Store normalized sequence length
            input_lens_ratio.append(seq_len / max_seq_len)
            labels.append(int(label))
            paths.append(path)
    else:
        # For non-sequential (1D) features
        max_feat_len = batch[0][0].shape[0]
        # Initialize input tensor with zeros (shape: [batch_size, max_feat_len])
        inputs = np.zeros((batch_size, max_feat_len), dtype='float32')

        for i, (tensor, label, path) in enumerate(batch):
            feat_len = tensor.shape[0]
            # Copy actual feature data into the input tensor
            inputs[i, :feat_len] = tensor
            # Store normalized feature length
            input_lens_ratio.append(feat_len / max_feat_len)
            labels.append(int(label))
            paths.append(path)

    # Convert inputs, labels, and length ratios to PyTorch tensors and return
    return (
        torch.tensor(inputs),               # Padded input tensor
        torch.tensor(labels),               # Corresponding labels
        torch.tensor(input_lens_ratio),     # Length ratios (useful for masking or attention)
        paths                               # Original file paths (for reference or debugging)
    )

