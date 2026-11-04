import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.nih import LABELS
from src.train.train import evaluate


def make_loader(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = torch.tensor((rng.random((n, len(LABELS))) < 0.3).astype("float32"))
    x = y * 10 - 5  # logits +5 for positives, -5 for negatives
    return DataLoader(TensorDataset(x, y), batch_size=32)

class _Negate(torch.nn.Module):
    def forward(self, x):
        return -x

@pytest.mark.parametrize("sign, expected", [(1, 1.0), (-1, 0.0)])
def test_evaluate_known_auroc(sign, expected):
    model = torch.nn.Identity() if sign == 1 else _Negate()
    out = evaluate(model, make_loader(), torch.nn.BCEWithLogitsLoss(), torch.device("cpu"))
    assert list(out["auroc"]) == LABELS
    assert out["mean_auroc"] == pytest.approx(expected)
    assert all(v == pytest.approx(expected) for v in out["auroc"].values())

def test_evaluate_perfect_loss():
    out = evaluate(torch.nn.Identity(), make_loader(), torch.nn.BCEWithLogitsLoss(), torch.device("cpu"))
    assert out["loss"] == pytest.approx(np.log1p(np.exp(-5)), rel=1e-3)

def test_evaluate_single_class_raises():
    y = torch.zeros(50, len(LABELS))
    loader = DataLoader(TensorDataset(y, y), batch_size=16)
    with pytest.raises(ValueError):
        evaluate(torch.nn.Identity(), loader, torch.nn.BCEWithLogitsLoss(), torch.device("cpu"))