import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from src.data.dataset import NIHDataset
from src.data.nih import LABELS, PROJECT_ROOT
from src.models.classifier import build_model
from src.train.weights import compute_pos_weight

CHECKPOINT = PROJECT_ROOT / "models" / "densenet121_best.pt"
METRICS = PROJECT_ROOT / "reports" / "train_metrics.json"
HISTORY = PROJECT_ROOT / "reports" / "train_history.csv"
SMOKE_DIR = PROJECT_ROOT /"data" / "interim" / "smoke"
NUM_WORKERS = 4

def get_device() -> torch.device:
    #mps is availabe, else gpu
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

        
        score = roc_auc_score(y_true, y_score)
        auroc_dict[label] = float(score)

    mean_auroc = float(np.mean(list(auroc_dict.values())))

    return {
        "loss": float(mean_loss),
        "mean_auroc": float(mean_auroc),
        "auroc": auroc_dict
    }

def main(smoke: bool = False) -> None:
    """Train DenseNet121 on NIH train, select by val mean AUROC.
    
    Args:
        smoke (bool): If True, trains on 640 images for 2 epochs (full val) to test end-to-end plumbing.
    """

    params = yaml.safe_load((PROJECT_ROOT / "params.yaml").read_text())["train"]
    seed = params["seed"]
    epochs = 2 if smoke else params["epochs"]
    batch_size = params["batch_size"]
    lr = params["lr"]
    weight_decay = params["weight_decay"]
    lr_factor = params["lr_factor"]
    lr_patience = params["lr_patience"]

    assert isinstance(lr, float), f"Expected lr to be float, got {type(lr)} {(lr)}"
    assert isinstance(weight_decay, float), f"Expected weight_decay to be float, got {type(weight_decay)}"

    torch.manual_seed(seed)
    dev = get_device()
    print(f"Using device: {dev} | Smoke mode: {smoke}")

    #building datasets and loaders
    train_ds = NIHDataset(split="train")
    pw = compute_pos_weight(train_ds.labels)

    if smoke:
        train_ds = Subset(train_ds, range(640))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0
    )


    val_ds = NIHDataset(split="val")
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0
    )

    model = build_model(pretrained=True).to(dev)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw.to(dev))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=lr_factor, patience=lr_patience
        )

    out_checkpoint = SMOKE_DIR / "densenet121_best.pt" if smoke else CHECKPOINT
    out_metrics = SMOKE_DIR / "train_metrics.json" if smoke else METRICS
    out_history = SMOKE_DIR / "train_history.csv" if smoke else HISTORY

    out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_history.parent.mkdir(parents=True, exist_ok=True)


    best_val_auroc = -1.0
    logs = []

    for epoch in range(epochs):
        t_start = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, dev)
        val_metrics = evaluate(model, val_loader, criterion, dev)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["mean_auroc"])

        elapsed = time.perf_counter() - t_start

        print(
            f"Epoch {epoch+1:2d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Mean AUROC: {val_metrics['mean_auroc']:.4f} | "
            f"LR: {current_lr:.1e} | "
            f"Time: {elapsed:.0f}s"
        )

        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mean_auroc": val_metrics["mean_auroc"],
            "lr": current_lr,
            "time_sec": elapsed
        })

        pd.DataFrame(logs).to_csv(out_history, index=False)

        if val_metrics["mean_auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["mean_auroc"]
            checkpoint_payload = {
                "state_dict":model.state_dict(),
                "labels": LABELS,
                "epoch": epoch + 1,
                "val_mean_auroc": best_val_auroc
            }

            torch.save(checkpoint_payload, out_checkpoint)
            out_metrics.write_text(json.dumps({"epoch": epoch + 1, **val_metrics}, indent=2))            
            print(f"Saved new best checkpoint with Val Mean AUROC: {best_val_auroc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet121 classifier on NIH ChestX-ray14")
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test with subset of data and 2 epochs")
    args = parser.parse_args()
    main(smoke=args.smoke)


