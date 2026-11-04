import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.data.nih import LABELS


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_one_epoch(model, loader, criterion, optimizer, device) -> float:


    """
    Train the model for one epoch over the given data loader.
    
    Args:
        model: The neural network model.
        loader: PyTorch DataLoader providing training batches.
        criterion: Loss function.
        optimizer: Optimizer for parameter updates.
        device: Computing device (mps or cpu).

    Returns:
        float: The mean loss over all batches in the epoch.
    """

    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_log_grad = None
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    assert n_batches > 0, "train_one_epoch got empty loader"
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:

    """
    Evaluate the model on a validation or test data loader.
    
    Args:
        model: The neural network model.
        loader: PyTorch DataLoader providing evaluation batches.
        criterion: Loss function.
        device: Computing device (mps or cpu).
        
    Returns:
        dict: Dictionary containing mean loss, mean AUROC, and per-class AUROC dictionary.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_logits = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        n_batches += 1

        all_logits.append(logits.cpu())
        all_targets.append(y.cpu())

    assert n_batches > 0, "evaluate got an empty loader"
    mean_loss = total_loss / n_batches

    logits_mat = torch.cat(all_logits, dim=0).numpy()
    targets_mat = torch.cat(all_targets, dim=0).numpy()

    auroc_dict = {}

    for i, label in enumerate(LABELS):
        y_true = targets_mat[:, i]
        y_score = logits_mat[:, i]

        if len(np.unique(y_true)) < 2:
            raise ValueError(f"Class {label!r} has only 1 unique value in targets")

        try:
            score = roc_auc_score(y_true, y_score)
        except ValueError as e:
            raise ValueError(f"Error computing AUROC for class {label!r}: {e}")

        auroc_dict[label] = float(score)

    mean_auroc = float(np.mean(list(auroc_dict.values())))

    return {
        "loss": float(mean_loss),
        "mean_auroc": float(mean_auroc),
        "auroc": auroc_dict
    }




