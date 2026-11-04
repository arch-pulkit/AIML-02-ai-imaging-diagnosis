import numpy as np
import torch


def compute_pos_weight(labels: np.ndarray) -> torch.Tensor:

    n_samples = len(labels)
    pos = labels.sum(axis=0)

    #checking for 0 positives
    zero_pos_indices = np.where(pos == 0)[0]
    if len(zero_pos_indices) > 0:
        raise ValueError(f"Column indices {zero_pos_indices.tolist()} has 0 positive examples")

    neg = n_samples - pos
    weights = neg / pos

    return torch.tensor(weights, dtype=torch.float32)