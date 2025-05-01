import numpy as np
import torch

# Process a batch of data
def collate_fn(batch):
    # Sort the batch by audio length (descending order)
    batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
    max_audio_length = batch[0][0].shape[0]
    batch_size = len(batch)

    # Create a zero tensor with the maximum audio length
    inputs = np.zeros((batch_size, max_audio_length), dtype='float32')
    input_lens_ratio = []
    labels = []

    paths = []
    for x in range(batch_size):
        tensor, label, path = batch[x]
        labels.append(label)
        paths.append(path)
        seq_length = tensor.shape[0]

        # Insert the data into the zero tensor (padding shorter sequences)
        inputs[x, :seq_length] = tensor[:]
        input_lens_ratio.append(seq_length / max_audio_length)

    input_lens_ratio = np.array(input_lens_ratio, dtype='float32')
    labels = np.array(labels, dtype='int64')

    return (torch.tensor(inputs), torch.tensor(labels), torch.tensor(input_lens_ratio), paths)
