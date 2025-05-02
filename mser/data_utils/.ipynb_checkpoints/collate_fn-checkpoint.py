import numpy as np
import torch


#FOR NORMAL MODEL
# Process a batch of data
# def collate_fn(batch):
#     # Sort the batch by audio length (descending order)
#     batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
#     max_audio_length = batch[0][0].shape[0]
#     batch_size = len(batch)

#     # Create a zero tensor with the maximum audio length
#     inputs = np.zeros((batch_size, max_audio_length), dtype='float32')
#     input_lens_ratio = []
#     labels = []

#     paths = []
#     for x in range(batch_size):
#         tensor, label, path = batch[x]
#         labels.append(label)
#         paths.append(path)
#         seq_length = tensor.shape[0]

#         # Insert the data into the zero tensor (padding shorter sequences)
#         inputs[x, :seq_length] = tensor[:]
#         input_lens_ratio.append(seq_length / max_audio_length)

#     input_lens_ratio = np.array(input_lens_ratio, dtype='float32')
#     labels = np.array(labels, dtype='int64')

#     return (torch.tensor(inputs), torch.tensor(labels), torch.tensor(input_lens_ratio), paths)

def collate_fn(batch):
    batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)
    is_sequence = batch[0][0].ndim == 2

    batch_size = len(batch)
    input_lens_ratio = []
    labels = []
    paths = []

    if is_sequence:
        max_seq_len = batch[0][0].shape[0]
        feat_dim = batch[0][0].shape[1]
        inputs = np.zeros((batch_size, max_seq_len, feat_dim), dtype='float32')

        for i, (tensor, label, path) in enumerate(batch):
            seq_len = tensor.shape[0]
            inputs[i, :seq_len, :] = tensor
            input_lens_ratio.append(seq_len / max_seq_len)
            labels.append(int(label))
            paths.append(path)

    else:
        max_feat_len = batch[0][0].shape[0]
        inputs = np.zeros((batch_size, max_feat_len), dtype='float32')

        for i, (tensor, label, path) in enumerate(batch):
            feat_len = tensor.shape[0]
            inputs[i, :feat_len] = tensor
            input_lens_ratio.append(feat_len / max_feat_len)
            labels.append(int(label))
            paths.append(path)

    return (
        torch.tensor(inputs),
        torch.tensor(labels),
        torch.tensor(input_lens_ratio),
        paths
    )


##FOR NORMALMODEL (pt2)
# def collate_fn(batch):
#     # Infer from first sample whether features are 1D or 2D
#     is_sequence = batch[0][0].ndim == 2  # [T, F] vs [F]

#     batch = sorted(batch, key=lambda sample: sample[0].shape[0], reverse=True)

#     batch_size = len(batch)
#     input_lens_ratio = []
#     labels = []
#     paths = []

#     if is_sequence:
#         max_seq_len = batch[0][0].shape[0]
#         feat_dim = batch[0][0].shape[1]
#         inputs = np.zeros((batch_size, max_seq_len, feat_dim), dtype='float32')

#         for i, (tensor, label, path) in enumerate(batch):
#             seq_len = tensor.shape[0]
#             inputs[i, :seq_len, :] = tensor[:]
#             input_lens_ratio.append(seq_len / max_seq_len)
#             labels.append(label)
#             paths.append(path)

#     else:
#         max_feat_len = batch[0][0].shape[0]
#         inputs = np.zeros((batch_size, max_feat_len), dtype='float32')

#         for i, (tensor, label, path) in enumerate(batch):
#             feat_len = tensor.shape[0]
#             inputs[i, :feat_len] = tensor[:]
#             input_lens_ratio.append(feat_len / max_feat_len)
#             labels.append(label)
#             paths.append(path)

#     input_lens_ratio = np.array(input_lens_ratio, dtype='float32')
#     labels = np.array(labels, dtype='int64')



#     return (
#         torch.tensor(inputs),
#         torch.tensor(labels),
#         torch.tensor(input_lens_ratio),
#         paths
#     )
