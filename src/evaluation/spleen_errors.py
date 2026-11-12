"""
TP / FN / FP error maps (axial + coronal) for selected spleen validation volumes.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Patch

from src.data.nih import PROJECT_ROOT
from src.evaluation.spleen_infer import (
    CKPTS,
    PIXDIM,
    PRED_DIR,
    TOL,
    check_split,
    val_tf,
)
from src.evaluation.spleen_overlays import CASES, dice

ERRORS_PNG = PROJECT_ROOT / "reports" / "spleen_errors.png"
COLORS = {
    "TP": (0.0, 0.8, 0.0, 0.45),
    "FN (missed)": (0.1, 0.4, 1.0, 0.7),
    "FP (extra)": (1.0, 0.1, 0.1, 0.7),
}


def error_rgba(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    RGBA overlay of a 2D slice: TP green, FN blue, FP red, transparent elsewhere.
    """
    out = np.zeros((*gt.shape, 4))
    masks = (gt & pred, gt & ~pred, pred & ~gt)
    for mask, color in zip(masks, COLORS.values(), strict=True):
        out[mask] = color
    return out


def panel(
    ax, img: np.ndarray, gt: np.ndarray, pred: np.ndarray, aspect: float, title: str
) -> None:
    """
    2D slice displayed as in 10a (np.rot90 of RAS): patient's left on image left.
    """
    ax.imshow(np.rot90(img), cmap="gray", vmin=0, vmax=1, aspect=aspect)
    ax.imshow(np.rot90(error_rgba(gt, pred)), aspect=aspect)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main() -> None:
    by_name = {Path(d["image"]).name.removesuffix(".nii.gz"): d for d in check_split()}
    per_vol = {
        tag: torch.load(path, map_location="cpu")["per_vol"]
        for tag, path in CKPTS.items()
    }
    coronal_aspect = PIXDIM[2] / PIXDIM[0]  # rows are 2.0 mm slices, columns 1.5 mm

    fig, axes = plt.subplots(len(CASES), 2, figsize=(9, 4.5 * len(CASES)))
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

        nz = gt.shape[2]
        zs = np.nonzero(gt.any(axis=(0, 1)))[0]
        z = int(gt.sum(axis=(0, 1)).argmax())  # axial slice with most spleen
        y = int(gt.sum(axis=(0, 2)).argmax())  # coronal slice with most spleen
        edge = (int(gt[:, :, 0].sum()), int(gt[:, :, -1].sum()))
        print(
            f"{vol:10} {tag}  dice {dc:.4f}  slices 0-{nz - 1}  spleen z {zs.min()}-{zs.max()}  "
            f"spleen voxels on first/last slice {edge}"
        )

        panel(
            ax=row[0],
            img=img[:, :, z],
            gt=gt[:, :, z],
            pred=pred[:, :, z],
            aspect=1.0,
            title=f"{vol} {tag} (Dice {dc:.3f})\naxial z={z}",
        )
        panel(
            ax=row[1],
            img=img[:, y, :],
            gt=gt[:, y, :],
            pred=pred[:, y, :],
            aspect=coronal_aspect,
            title=f"coronal y={y}\nspleen z {zs.min()}-{zs.max()} of 0-{nz - 1}",
        )

    fig.legend(
        handles=[Patch(color=c[:3], label=k) for k, c in COLORS.items()],
        loc="lower center",
        ncol=3,
        fontsize=9,
    )
    fig.suptitle(
        "Spleen U-Net (validation) error maps\n"
        "axial: anterior up | coronal: superior up | patient's LEFT on image LEFT",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(ERRORS_PNG, dpi=120)
    print("saved", ERRORS_PNG)


if __name__ == "__main__":
    main()
