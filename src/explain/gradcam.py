"""
Grad-CAM for the DenseNet121 chest X-RAY classifier, implemented with hooks.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image

from src.data.dataset import NIHDataset
from src.data.make_splits import OUT_PATH as SPLITS_CSV
from src.data.nih import LABELS, NIH_DIR, PROJECT_ROOT
from src.data.resize_images import OUT_DIR as IMAGE_DIR
from src.evaluation.predict import IMAGE_COL
from src.models.classifier import build_model
from src.train.train import CHECKPOINT

BBOX_CSV = NIH_DIR / "BBox_List_2017.csv"
BBOX_COLS = [IMAGE_COL, "label", "x", "y", "w", "h"]
# bbox files uses a different name than LABELS
BBOX_RENAME = {"Infiltrate": "Infiltration"}
ORIG_SIZE = 1024
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "gradcam"


def load_model(device: str = "cpu") -> torch.nn.Module:
    """
    Load the trained checkpoint into a fresh DenseNet121 in eval mode
    """
    ckpt = torch.load(CHECKPOINT, map_location=device)
    assert list(ckpt["labels"]) == list(LABELS), "checkpoint labels differ from LABELS"
    model = build_model(pretrained=False)
    model.load_state_dict(ckpt["state_dict"])

    return model.to(device).eval()


def capture(
    model: torch.nn.Module, x: torch.Tensor, label_idx: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Runs one image through the model and capture model.features' output and its gradient.
    """
    assert not model.training, (
        "call model.eval() first (BatchNorm on 1 image would be garbage)"
    )
    assert x.ndim == 3, f"expected one image (C, H, W), got shape {tuple(x.shape)}"
    assert 0 <= label_idx < len(LABELS), f"label_idx {label_idx} out of range"
    x = x.to(next(model.parameters()).device)
    store = {}

    def save_activation(module, inputs, output):
        # clone: DenseNet.forward applies relu to this tensor next, which would silently overwrite a stored view of it.
        store["act"] = output.detach().clone()

        def save_grad(grad):
            store["grad"] = grad.detach().clone()

        output.register_hook(save_grad)  # fires during backward()

    handle = model.features.register_forward_hook(save_activation)
    try:
        with torch.enable_grad():
            logits = model(x.unsqueeze(0))
            target = logits[0, label_idx]
            target.backward()
    finally:
        handle.remove()  # a leftover hook would fire on every later forward pass
        model.zero_grad(set_to_none=True)  # we don't need parameter grads; free them

    assert "act" in store and "grad" in store, "hook did not fire: wrong target layer?"
    assert store["act"].shape == store["grad"].shape, (
        "activation/gradient shape mismatch"
    )

    return store["act"], store["grad"], float(target.item())

def cam_from_maps(act: torch.Tensor, grad: torch.Tensor, size: tuple[int, int]) -> np.ndarray:
    """
    Grad-CAM math on captured maps: weight = mean gradient per map, ReLU(weighted sum),
    bilinear upsample to `size`, min-max to [0, 1]. A flat map is returned as zeros.
    """
    weights = grad.mean(dim=(2, 3), keepdim=True)  # (1, K, 1, 1): one weight per map
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))  # (1, 1, h, w)
    cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
    cam = cam[0, 0].cpu().numpy()
    lo, hi = cam.min(), cam.max()
    if hi == lo:
        return np.zeros_like(cam, dtype=np.float32)
    return ((cam - lo) / (hi - lo)).astype(np.float32)


def gradcam(
    model: torch.nn.Module, x: torch.Tensor, label_idx: int
) -> tuple[np.ndarray, float]:
    """
    Grad-CAM heatmap for one image and one label.
    """
    act, grad, logit = capture(model, x, label_idx)
    return cam_from_maps(act, grad, x.shape[-2:]), logit


def load_bboxes() -> pd.DataFrame:
    """
    NIH bounding boxes as columns image_index, label (LABELS names), x, y, w, h (1024 px).
    """

    raw = pd.read_csv(BBOX_CSV)
    assert list(raw.columns[:2]) == ["Image Index", "Finding Label"], (
        f"unexpected header: {list(raw.columns)}"
    )
    assert raw.iloc[:, 6:].isna().all().all(), "columns after the 6th are not empty"
    df = raw.iloc[:, :6].copy()
    df.columns = BBOX_COLS
    df["label"] = df["label"].replace(BBOX_RENAME)
    unknown = set(df["label"]) - set(LABELS)
    assert not unknown, f"bbox labels not in LABELS: {unknown}"
    assert df[["x", "y", "w", "h"]].notna().all().all(), "missing box coordinates"

    return df


def save_overlay(image_id: str, size: int, rows: list, boxes: pd.DataFrame) -> Path:
    """
    Save [image | image+CAM per label] with all boxes of this image drawn on every panel.
    """
    gray = np.asarray(
        Image.open(IMAGE_DIR / image_id)
        .convert("L")
        .resize((size, size), Image.Resampling.BILINEAR)
    )
    scale = size / ORIG_SIZE
    fig, axes = plt.subplots(1, len(rows) + 1, figsize=(4 * (len(rows) + 1), 4.4))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title(image_id)
    for ax, (label, cam, logit) in zip(axes[1:], rows, strict=True):
        ax.imshow(gray, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        ax.set_title(f"{label} (logit {logit:+.2f})")
    for ax in axes:
        for b in boxes.itertuples():
            ax.add_patch(
                Rectangle(
                    (b.x * scale, b.y * scale),
                    b.w * scale,
                    b.h * scale,
                    fill=False,
                    edgecolor="lime",
                    linewidth=1.5,
                )
            )
            ax.text(b.x * scale, b.y * scale - 2, b.label, color="lime", fontsize=8)
        ax.axis("off")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{Path(image_id).stem}_{'_'.join(r[0] for r in rows)}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM overlay for one test image.")
    parser.add_argument(
        "--image", help="file name; default: first test bbox image of labels[0]"
    )
    parser.add_argument("--labels", nargs="+", default=["Effusion", "Cardiomegaly"])
    args = parser.parse_args()

    # fail fast: cheap lookups before loading the model
    unknown = [lab for lab in args.labels if lab not in LABELS]
    assert not unknown, f"unknown labels {unknown}; valid: {LABELS}"
    boxes = load_bboxes()
    splits = pd.read_csv(SPLITS_CSV, usecols=[IMAGE_COL, "split"])
    test_ids = set(splits.loc[splits["split"] == "test", IMAGE_COL])
    in_test = boxes[IMAGE_COL].isin(test_ids)

    print(
        f"bbox: {len(boxes)} boxes on {boxes[IMAGE_COL].nunique()} images; "
        f"{boxes.loc[in_test, IMAGE_COL].nunique()} of those images in test"
    )

    if args.image is None:
        candidates = boxes.loc[in_test & (boxes["label"] == args.labels[0]), IMAGE_COL]
        assert len(candidates) > 0, (
            f"no test bbox images for {args.labels[0]}; pass --image"
        )
        image_id = candidates.iloc[0]
    else:
        image_id = args.image
    assert image_id in test_ids, f"{image_id} is not in the test split"

    ds = NIHDataset(split="test")
    names = [p.name for p in ds.image_paths]
    x, y_true = ds[names.index(image_id)]
    positives = [lab for lab, v in zip(LABELS, y_true.tolist(), strict=True) if v == 1]
    print(f"{image_id} NIH labels: {positives or ['No Finding']}")

    model = load_model()
    rows = []

    for lab in args.labels:
        cam, logit = gradcam(model, x, LABELS.index(lab))
        rows.append((lab, cam, logit))
        print(
            f"{lab:<20} logit {logit:+.3f}  cam {cam.shape} [{cam.min():.2f}, {cam.max():.2f}]"
        )
    out = save_overlay(image_id, x.shape[-1], rows, boxes[boxes[IMAGE_COL] == image_id])
    print(f"saved {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
