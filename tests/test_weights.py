import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import SPLITS_CSV, NIHDataset
from src.data.nih import LABELS
from src.train.weights import compute_pos_weight


def test_compute_pos_weight_hand_computed():
    # Column 0: 2 positives, 2 negatives -> weight = 2/2 = 1.0
    # Column 1: 1 positive, 3 negatives -> weight = 3/1 = 3.0
    labels = np.array([[1, 0], [0, 0], [1, 1], [0, 0]], dtype="float32")
    pw = compute_pos_weight(labels)
    
    assert pw.dtype == torch.float32
    assert torch.equal(pw, torch.tensor([1.0, 3.0]))


def test_compute_pos_weight_zero_positives_raises():
    zero_labels = np.zeros((10, 14), dtype="float32")
    with pytest.raises(ValueError):
        compute_pos_weight(zero_labels)


def test_compute_pos_weight_real_data():
    dataset = NIHDataset(split="train")
    pw = compute_pos_weight(dataset.labels)
    
    assert pw.shape == (14,)
    assert pw.dtype == torch.float32
    
    # Cross-check against independent pandas computation on the train split CSV
    df = pd.read_csv(SPLITS_CSV)
    tr = df[df["split"] == "train"]
    pos = tr[LABELS].sum()
    ref = torch.tensor(((len(tr) - pos) / pos).to_numpy(), dtype=torch.float32)
    
    assert torch.allclose(pw, ref, rtol=1e-4, atol=1e-4)