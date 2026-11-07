"""
Runs the trained DenseNet121 model on val and test splits and saves raw logits to CSV.

Inference is slow and analysis is fast, so imference runs once here and every later analysis reads these CSVs.
CSVs hold image IDs + logits.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import NIHDataset
from src.data.nih import LABELS, PROJECT_ROOT
from src.models.classifier import build_model
from src.train.train import CHECKPOINT, NUM_WORKERS, get_device

PRED_DIR = PROJECT_ROOT / "data" / "processed" / "predictions"
IMAGE_COL = "image_index"
BATCH_SIZE = 64


@torch.no_grad()
def predict(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    """
    Run inference and returning raw digits.

    Args:
        model: PyTorch Classification Model
        loader: DataLoader yielding (images, labels)
        device: Device to run inference

    Return:
        NumPy array of shape (N, len(LABELS) with float32 raw logits)
    """

    model.eval()
    batches = []

    for images, _ in loader:
        logits = model(images.to(device))
        batches.append(logits.cpu())

    return torch.cat(batches).numpy()


def main() -> None:
    """
    Load cheeckpoints, generate preds for val and test splits, save to CSV.
    """
    device = get_device()
    print(f"Using device: {device}")

    ckpt = torch.load(CHECKPOINT, map_location="cpu")

    assert CHECKPOINT.exists(), f"Checkpoint not found at {CHECKPOINT}"
    assert ckpt["labels"] == LABELS, (
        f"Checkpoint labels {ckpt['labels']} != LABELS {LABELS}"
    )

    # Build models and load weights
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    print(
        f"checkpoint: epoch {ckpt['epoch']}, val mean AUROC {ckpt['val_mean_auroc']:.5f}"
    )

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        dataset = NIHDataset(split=split)

        image_ids = [p.name for p in dataset.image_paths]
        assert len(image_ids) == len(dataset), (
            f"{split}: {len(image_ids)} IDs vs {len(dataset)} items"
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
        )
        logits = predict(model, loader, device)

        df = pd.DataFrame(logits, columns=[f"logit_{label}" for label in LABELS])
        df.insert(0, IMAGE_COL, image_ids)

        out_path = PRED_DIR / f"preds_{split}.csv"
        df.to_csv(out_path, index=False)
        print(split, df.shape)


if __name__ == "__main__":
    main()
