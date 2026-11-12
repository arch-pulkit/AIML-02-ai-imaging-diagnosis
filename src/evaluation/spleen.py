"""Step 10e: evaluation of the MSD Task09 spleen 3D U-Net trained on Kaggle (curve + metrics)."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.nih import PROJECT_ROOT

SPLEEN_DIR = PROJECT_ROOT / "data" / "interim" / "spleen"
HISTORY = SPLEEN_DIR / "spleen_history.csv"
BEST_CKPT = SPLEEN_DIR / "spleen_unet_best.pt"
LAST_CKPT = SPLEEN_DIR / "spleen_unet_last.pt"
CURVE_PNG = PROJECT_ROOT / "reports" / "spleen_curve.png"
SPLEEN_METRICS = PROJECT_ROOT / "reports" / "spleen_metrics.json"
VAL_INTERVAL = 5  # must match the Kaggle notebook (Cell 13)
N_VAL = 8
SW_ROI = (
    160,
    160,
    160,
)  # sliding-window ROI of validation (Cell 13); not stored in the checkpoint
PLATEAU_FROM = 400  # validations after this epoch; chosen from the 10e-1 block table
HIST_COLS = ["epoch", "train_loss", "train_sec", "val_dice", "val_dice_min", "val_sec"]
CKPT_KEYS = ["epoch", "hu_window", "per_vol", "pixdim", "roi", "state_dict", "val_dice"]
KAGGLE_ENV = {  # recorded from the Kaggle run (Cell 2), not read from the checkpoint
    "monai": "1.6.1",
    "torch": "2.11.0+cu130",
    "nibabel": "5.4.2",
    "gpu": "Tesla T4",
}


def load_history() -> pd.DataFrame:
    """Read the Kaggle history CSV; assert one row per epoch and validation every VAL_INTERVAL."""
    h = pd.read_csv(HISTORY)
    assert list(h.columns) == HIST_COLS, f"unexpected columns {list(h.columns)}"
    assert (h["epoch"].to_numpy() == np.arange(1, len(h) + 1)).all(), (
        "epochs are not 1..N"
    )
    is_val = h["val_dice"].notna().to_numpy()
    expected = (h["epoch"] % VAL_INTERVAL == 0).to_numpy()
    assert (is_val == expected).all(), (
        "validation rows are not every VAL_INTERVAL epochs"
    )
    v = h[is_val]
    assert (v["val_dice_min"] <= v["val_dice"]).all(), (
        "worst-volume Dice above the mean"
    )
    return h


def load_ckpt(path, h: pd.DataFrame) -> dict:
    """Load a spleen checkpoint on CPU and cross-check it against its epoch's history row."""
    ck = torch.load(path, map_location="cpu")
    assert sorted(ck) == CKPT_KEYS, f"{path.name}: keys {sorted(ck)}"
    row = h.loc[h["epoch"] == ck["epoch"]]
    assert len(row) == 1, f"{path.name}: epoch {ck['epoch']} not in history"
    dices = np.array([float(d) for d in ck["per_vol"].values()])
    assert len(dices) == N_VAL, f"{path.name}: {len(dices)} volumes, expected {N_VAL}"
    assert np.isclose(float(ck["val_dice"]), row["val_dice"].item(), atol=1e-6), (
        "val_dice vs history"
    )
    assert np.isclose(dices.min(), row["val_dice_min"].item(), atol=1e-6), (
        "min per_vol vs history"
    )
    assert np.isclose(dices.mean(), float(ck["val_dice"]), atol=1e-4), (
        "mean per_vol vs val_dice"
    )
    return ck


def summary(per_vol: dict) -> dict:
    """Mean, sample SD, min and the per-volume dict (plain floats) for one checkpoint."""
    d = np.array([float(x) for x in per_vol.values()])
    return {
        "mean_dice": float(d.mean()),
        "sd": float(d.std(ddof=1)),
        "min": float(d.min()),
        "per_vol": {k: float(x) for k, x in per_vol.items()},
    }


def plateau(h: pd.DataFrame) -> dict:
    """Distribution of val mean Dice over all validations after PLATEAU_FROM."""
    v = h.loc[h["epoch"] > PLATEAU_FROM, "val_dice"].dropna()
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return {
        "from_epoch": PLATEAU_FROM,
        "n": len(v),
        "mean": float(v.mean()),
        "median": float(q50),
        "q25": float(q25),
        "q75": float(q75),
        "min": float(v.min()),
        "max": float(v.max()),
    }


def plot_curve(h: pd.DataFrame, best_epoch: int) -> None:
    """Save train loss (left) and val mean / worst-volume Dice (right) to CURVE_PNG."""
    v = h.dropna(subset=["val_dice"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(h["epoch"], h["train_loss"], lw=0.8)
    ax1.set(
        xlabel="epoch", ylabel="soft Dice loss (train crops)", title="Training loss"
    )
    ax2.plot(v["epoch"], v["val_dice"], marker=".", lw=1, label="mean (8 volumes)")
    ax2.plot(v["epoch"], v["val_dice_min"], marker=".", lw=1, label="worst volume")
    ax2.axvline(
        best_epoch, color="grey", ls="--", lw=0.8, label=f"best epoch {best_epoch}"
    )
    ax2.set(
        xlabel="epoch",
        ylabel="Dice, spleen (1.5x1.5x2 mm space)",
        title="Validation Dice",
        ylim=(0, 1),
    )
    ax2.legend(loc="lower right")
    fig.tight_layout()
    CURVE_PNG.parent.mkdir(exist_ok=True)
    fig.savefig(CURVE_PNG, dpi=150)
    plt.close(fig)
    print(f"saved {CURVE_PNG}")


def main() -> None:
    h = load_history()
    best = load_ckpt(BEST_CKPT, h)
    last = load_ckpt(LAST_CKPT, h)
    v = h.dropna(subset=["val_dice"])
    assert best["epoch"] == int(v.loc[v["val_dice"].idxmax(), "epoch"]), (
        "best ckpt is not the max"
    )
    assert last["epoch"] == len(h), (
        f"last ckpt epoch {last['epoch']}, history has {len(h)}"
    )
    assert set(best["per_vol"]) == set(last["per_vol"]), (
        "different val volumes in best and last"
    )
    for key in ("pixdim", "hu_window", "roi"):
        assert best[key] == last[key], f"{key} differs between best and last"

    sb, sl, pl = summary(best["per_vol"]), summary(last["per_vol"]), plateau(h)
    table = pd.DataFrame({"best": sb["per_vol"], "last": sl["per_vol"]})
    table["diff"] = table["last"] - table["best"]
    table.index = [n.removesuffix(".nii.gz") for n in table.index]
    print(f"per volume (best epoch {best['epoch']}, last epoch {last['epoch']})")
    print(table.sort_values("best").to_string(float_format="%.4f"))
    for name, s in (("best", sb), ("last", sl)):
        print(
            f"{name:<5} mean {s['mean_dice']:.4f}  sd {s['sd']:.4f}  min {s['min']:.4f}"
        )
    print(
        f"plateau (epochs > {PLATEAU_FROM}, n={pl['n']})  mean {pl['mean']:.4f}  "
        f"median {pl['median']:.4f}  IQR {pl['q25']:.4f}-{pl['q75']:.4f}  "
        f"range {pl['min']:.4f}-{pl['max']:.4f}"
    )

    out = {
        "dataset": "MSD Task09_Spleen",
        "split": {
            "train": 33,
            "val": N_VAL,
            "seed": 42,
            "test": "none (MSD test labels not public)",
        },
        "dice_space": "resampled 1.5x1.5x2.0 mm, spleen only, hard (argmax)",
        "settings": {
            "pixdim": best["pixdim"],
            "hu_window": best["hu_window"],
            "train_crop": best["roi"],
            "sw_roi": SW_ROI,
            "max_epochs": len(h),
            "val_interval": VAL_INTERVAL,
        },
        "env": KAGGLE_ENV,
        "best": {"epoch": best["epoch"], **sb},
        "last": {"epoch": last["epoch"], **sl},
        "plateau": pl,
    }
    SPLEEN_METRICS.write_text(json.dumps(out, indent=2) + "\n")
    print(f"saved {SPLEEN_METRICS}")
    plot_curve(h, best["epoch"])


if __name__ == "__main__":
    main()
