import numpy as np
import torch


# Compute Accuracy
def accuracy(output, label):
    if isinstance(output, tuple):
        output = output[0]
    output = torch.nn.functional.softmax(output, dim=-1)
    output = output.data.cpu().numpy()
    output = np.argmax(output, axis=1)
    label = label.data.cpu().numpy()
    acc = np.mean((output == label).astype(int))
    return acc
