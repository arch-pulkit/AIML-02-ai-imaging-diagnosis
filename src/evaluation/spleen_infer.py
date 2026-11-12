"""
reload the Kaggle spleen U-Net on the Mac and re-run validation inference.
"""

import argparse
import time
from pathlib import Path

import monai
import numpy as np
import pandas as pd
import torch
from monai.apps import DecathlonDataset
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)
from scipy import ndimage

from src.data.nih import PROJECT_ROOT
from src.evaluation.spleen import BEST_CKPT, LAST_CKPT, N_VAL, SW_ROI

MSD_ROOT = PROJECT_ROOT / "data" / "interim" / "msd"  # git-ignored; ~3 GB incl. the tar
TASK = "Task09_Spleen"
SEED = 42  # must match the Kaggle notebook (Cell 3)
VAL_FRAC = 0.2
N_LABELLED = 41
CKPTS = {"best": BEST_CKPT, "last": LAST_CKPT}
PRED_DIR = PROJECT_ROOT / "data" / "interim" / "spleen" / "preds"
TOL = 1e-3  # Mac CPU vs Kaggle GPU: argmax ties at borderline voxels
FAILURES_CSV = PROJECT_ROOT / "reports" / "spleen_failures.csv"  # per-volume Dice + TP/FP/FN, git-tracked

# --- copied from notebooks/04_spleen_unet.ipynb, cell index 6 (transforms) ---
KEYS = ["image", "label"]
PIXDIM = (1.5, 1.5, 2.0)
HU_MIN, HU_MAX = -57, 164

base = [
    LoadImaged(keys=KEYS),
    EnsureChannelFirstd(keys=KEYS),
    Orientationd(keys=KEYS, axcodes="RAS"),
    Spacingd(keys=KEYS, pixdim=PIXDIM, mode=("bilinear", "nearest")),
    ScaleIntensityRanged(
        keys="image", a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True
    ),
    CropForegroundd(keys=KEYS, source_key="image", allow_smaller=True),
]
val_tf = Compose(base)
# --- end copy ---

VOXEL_ML = float(np.prod(PIXDIM)) / 1000  # 1.5 * 1.5 * 2.0 = 4.5 mm^3 = 0.0045 mL


# --- copied from notebooks/04_spleen_unet.ipynb, cell index 12 ---
def build_unet() -> UNet:
    """
    3D U-Net from the MONAI spleen tutorial: 1 input channel, 2 classes.
    """
    return UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=2,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm=Norm.BATCH,
    )


# --- end copy ---


# --- validation post-processing, as in notebook cell index 13 ---
post_pred = AsDiscrete(argmax=True, to_onehot=2)
post_label = AsDiscrete(to_onehot=2)
dice_metric = DiceMetric(include_background=False, reduction="mean")


def datalist(section: str) -> list[dict]:
    """
    Same DecathlonDataset call as the Kaggle notebook; returns [{"image", "label"}] paths.
    """
    ds = DecathlonDataset(
        root_dir=MSD_ROOT,
        task=TASK,
        section=section,
        transform=(),
        download=True,
        seed=SEED,
        val_frac=VAL_FRAC,
        cache_rate=0.0,
    )
    return ds.data


def names(files: list[dict]) -> set[str]:
    """
    Image file names (e.g. spleen_26.nii.gz) of a datalist
    """
    return {Path(d["image"]).name for d in files}


def check_split() -> list[dict]:
    """
    Rebuild the Kaggle split and assert it matches the checkpoint's validation volumes.
    """
    MSD_ROOT.mkdir(
        parents=True, exist_ok=True
    )  # DecathlonDataset requires an existing root_dir
    train, val = datalist("training"), datalist("validation")
    assert not names(train) & names(val), "train and val overlap"
    assert len(names(train) | names(val)) == N_LABELLED, (
        "union is not all labelled volumes"
    )
    assert len(val) == N_VAL, f"{len(val)} val volumes, expected {N_VAL}"
    for d in train + val:
        assert Path(d["image"]).name == Path(d["label"]).name, (
            f"image/label mismatch {d}"
        )
    ckpt_names = set(torch.load(BEST_CKPT, map_location="cpu")["per_vol"])
    assert names(val) == ckpt_names, (
        f"val {sorted(names(val))} vs ckpt {sorted(ckpt_names)}"
    )
    print(
        f"train {len(train)}  val {len(val)}  union {N_LABELLED}  == checkpoint per_vol keys"
    )
    print("val:", sorted(names(val)))
    return val


def load_unet(path: Path, device: torch.device) -> tuple[UNet, dict]:
    """
    Strict-load a Kaggle checkpoint into build_unet(); assert its settings match this file.
    """
    ckpt = torch.load(path, map_location="cpu")
    assert tuple(ckpt["pixdim"]) == PIXDIM, f"pixdim {ckpt['pixdim']} vs {PIXDIM}"
    assert tuple(ckpt["hu_window"]) == (HU_MIN, HU_MAX), (
        f"hu_window {ckpt['hu_window']}"
    )
    model = build_unet()
    model.load_state_dict(ckpt["state_dict"])  # strict: any key mismatch raises
    return model.to(device).eval(), ckpt


@torch.no_grad()
def segment(
    model: UNet, item: dict, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Whole-volume sliding-window prediction → (Dice as in the notebook, pred mask, true mask).
    """
    d = val_tf(item)
    img, lab = d["image"], d["label"]
    spacing = np.linalg.norm(np.asarray(img.affine)[:3, :3], axis=0)
    assert np.allclose(spacing, PIXDIM), f"spacing {spacing} vs {PIXDIM}"
    logits = sliding_window_inference(
        img[None].to(device), SW_ROI, sw_batch_size=4, predictor=model
    )
    pred_oh = post_pred(decollate_batch(logits)[0])
    lab_oh = post_label(lab.to(device))
    dice = dice_metric(y_pred=[pred_oh], y=[lab_oh]).item()
    dice_metric.reset()
    return dice, pred_oh[1].cpu().numpy() > 0.5, lab[0].cpu().numpy() > 0.5


def failure_stats(pred: np.ndarray, gt: np.ndarray) -> dict:
    """
    TP/FP/FN in mL, own Dice, connected components; FP 'far' = blobs not touching the spleen.
    """
    tp, fp, fn = (pred & gt).sum(), (pred & ~gt).sum(), (~pred & gt).sum()
    comps, n_comp = ndimage.label(pred, structure=np.ones((3, 3, 3)))  # 26-connectivity
    touching = np.unique(comps[pred & gt])  # components that overlap the true spleen
    far = pred & ~np.isin(comps, touching)
    sizes = np.bincount(comps.ravel())[1:]
    return {
        "dice_own": 2
        * tp
        / (2 * tp + fp + fn),  # gt is never empty in val → no zero division
        "true_ml": gt.sum() * VOXEL_ML,
        "pred_ml": pred.sum() * VOXEL_ML,
        "tp_ml": tp * VOXEL_ML,
        "fp_ml": fp * VOXEL_ML,
        "fp_far_ml": far.sum() * VOXEL_ML,
        "fn_ml": fn * VOXEL_ML,
        "n_comp": n_comp,
        # empty prediction is a valid output → NaN, not an error
        "largest_frac": sizes.max() / sizes.sum() if n_comp else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--vols", nargs="+", help="e.g. spleen_26 (default: all 8 val volumes)"
    )
    p.add_argument("--ckpts", nargs="+", default=list(CKPTS), choices=list(CKPTS))
    p.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    args = p.parse_args()

    print(
        "monai",
        monai.__version__,
        "| torch",
        torch.__version__,
        "| device",
        args.device,
    )
    by_name = {Path(d["image"]).name: d for d in check_split()}
    vols = list(by_name) if args.vols is None else [f"{v}.nii.gz" for v in args.vols]
    assert set(vols) <= set(by_name), (
        f"not val volumes: {sorted(set(vols) - set(by_name))}"
    )
    device = torch.device(args.device)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag in args.ckpts:
        model, ckpt = load_unet(CKPTS[tag], device)
        for name in vols:
            t0 = time.time()
            dice, pred, gt = segment(model, by_name[name], device)
            s = failure_stats(pred, gt)
            assert abs(s["dice_own"] - dice) < 1e-5, (
                f"own Dice {s['dice_own']} vs MONAI {dice}"
            )
            vol = name.removesuffix(".nii.gz")
            np.savez_compressed(PRED_DIR / f"{tag}_{vol}.npz", pred=pred)
            ref = ckpt["per_vol"][name]
            rows.append(
                {
                    "ckpt": tag,
                    "epoch": ckpt["epoch"],
                    "volume": vol,
                    "dice": dice,
                    "kaggle": ref,
                    "diff": dice - ref,
                    **s,
                    "sec": time.time() - t0,
                }
            )
            print(
                f"{tag} ep{ckpt['epoch']} {vol:10} dice {dice:.4f} (kaggle {ref:.4f})  "
                f"{rows[-1]['sec']:.0f}s",
                flush=True,
            )

    df = pd.DataFrame(rows).drop(columns=["dice_own", "tp_ml"])
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    bad = df[df["diff"].abs() > TOL]
    assert bad.empty, (
        f"not reproduced within {TOL}:\n{bad[['ckpt', 'volume', 'dice', 'kaggle']]}"
    )
    print(f"reload check OK: {len(df)} Dice values within {TOL} of Kaggle")

    if args.vols is None and args.ckpts == list(CKPTS):  # only a complete run goes into the report
        df.drop(columns="sec").round(4).to_csv(FAILURES_CSV, index=False)
        print("saved", FAILURES_CSV)


if __name__ == "__main__":
    main()
