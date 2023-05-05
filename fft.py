import torch
import numpy as np
import pywt

def wavelet_transform(input_tensor, wavelet='db1'):
    # Create an empty list to store the transformed sequences
    transformed_sequences = []

    # Iterate through the batch dimension
    for batch_idx in range(input_tensor.size(0)):
        transformed_batch = []

        # Iterate through the one_hot_dim dimension
        for dim_idx in range(input_tensor.size(2)):
            # Extract the one-hot vector along the num_sequence dimension
            one_hot_vector = input_tensor[batch_idx, :, dim_idx].numpy()

            # Compute the DWT
            coeffs = pywt.wavedec(one_hot_vector, wavelet)

            # Concatenate the approximation and detail coefficients
            transformed_coeffs = np.hstack(coeffs)

            # Append the transformed coefficients to the transformed_batch list
            transformed_batch.append(transformed_coeffs)

        # Stack the transformed coefficients along the one_hot_dim dimension
        transformed_batch_tensor = torch.tensor(transformed_batch, dtype=torch.float32).T

        # Append the transformed batch tensor to the transformed_sequences list
        transformed_sequences.append(transformed_batch_tensor)

    # Stack the transformed tensors along the batch dimension
    transformed_tensor = torch.stack(transformed_sequences)

    return transformed_tensor

# Example one-hot encoded sequence tensor
batch_size = 2
num_sequence = 10
one_hot_dim = 5

one_hot_sequence = torch.tensor([
    [
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
    ],
    [
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
],
], dtype=torch.float32)

transformed_sequence = wavelet_transform(one_hot_sequence, wavelet='db1')

print("Wavelet transformed sequence:")
print(transformed_sequence)
print("Shape of the one-hot encoded sequence:", one_hot_sequence.shape)
print("Shape of the wavelet transformed sequence:", transformed_sequence.shape)