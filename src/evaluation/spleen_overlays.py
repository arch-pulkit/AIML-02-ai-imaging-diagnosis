"""
CT + true / predicted spleen contours for selected validation volumes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.nih import PROJECT_ROOT
from src.evaluation.spleen_infer import CKPTS, PRED_DIR, TOL, check_split, val_tf

OVERLAY_PNG = PROJECT_ROOT / "reports" / "spleen_overlays.png"
CASES = [
    ("spleen_9", "best"),
    ("spleen_26", "best"),
    ("spleen_26", "last"),
    ("spleen_49", "last"),
]


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """
    Hard Dice of two boolean masks (gt never empty for val volumes)
    ."""
    return float(2 * (pred & gt).sum() / (pred.sum() + gt.sum()))


def show(
    ax, img: np.ndarray, gt: np.ndarray, pred: np.ndarray, z: int, title: str
) -> None:
    """
    Axial slice z, displayed as in 10a: CT (already windowed to 0..1) + true/predicted contours.
    """
    ax.imshow(np.rot90(img[:, :, z]), cmap="gray", vmin=0, vmax=1)
    for mask, color in ((gt, "lime"), (pred, "red")):
        m = np.rot90(mask[:, :, z]).astype(float)
        if (
            m.any()
        ):  # an empty mask on this slice is a valid result (e.g. spleen missed)
            ax.contour(m, levels=[0.5], colors=color, linewidths=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main() -> None:
    by_name = {Path(d["image"]).name.removesuffix(".nii.gz"): d for d in check_split()}
    per_vol = {
        tag: torch.load(path, map_location="cpu")["per_vol"]
        for tag, path in CKPTS.items()
    }

    fig, axes = plt.subplots(len(CASES), 2, figsize=(8, 4 * len(CASES)))
    for row, (vol, tag) in zip(axes, CASES, strict=True):
        d = val_tf(by_name[vol])
        img = d["image"][0].cpu().numpy()
        gt = d["label"][0].cpu().numpy() > 0.5
        pred = np.load(PRED_DIR / f"{tag}_{vol}.npz")["pred"]
        assert pred.shape == gt.shape, (
            f"{vol} {tag}: pred {pred.shape} vs label {gt.shape}"
        )
        dc, ref = dice(pred, gt), per_vol[tag][f"{vol}.nii.gz"]
        assert abs(dc - ref) < TOL, (
            f"{vol} {tag}: saved mask Dice {dc:.4f} vs checkpoint {ref:.4f}"
        )

        z_spleen = int(gt.sum(axis=(0, 1)).argmax())
        z_err = int((pred ^ gt).sum(axis=(0, 1)).argmax())
        for ax, z, what in zip(
            row, (z_spleen, z_err), ("most spleen", "most error"), strict=True
        ):
            fn = int((gt & ~pred)[:, :, z].sum())
            fp = int((pred & ~gt)[:, :, z].sum())
            show(
                ax,
                img,
                gt,
                pred,
                z,
                f"{vol} {tag} (Dice {dc:.3f})\n{what}: z={z}  FN {fn}  FP {fp} vox",
            )
        print(
            f"{vol:10} {tag}  dice {dc:.4f}  shape {gt.shape}  z_spleen {z_spleen}  z_err {z_err}"
        )

    fig.suptitle(
        "Spleen U-Net (validation): true = lime, predicted = red\n"
        "axial, anterior up, patient's LEFT on image LEFT (mirror of radiological convention)",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OVERLAY_PNG, dpi=120)
    print("saved", OVERLAY_PNG)


if __name__ == "__main__":
    main()
