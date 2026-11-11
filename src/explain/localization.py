"""
Grad-CAM localization vs NIH bounding boxes (test split): pointing game + mass lift,
each with a shuffled-box control (same label, box taken from another image).

Run: python -m src.explain.localization
"""

import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.dataset import NIHDataset
from src.data.nih import LABELS, PROJECT_ROOT
from src.evaluation.predict import IMAGE_COL
from src.evaluation.thresholds import THRESHOLDS
from src.explain.gradcam import ORIG_SIZE, gradcam, load_bboxes, load_model
from src.train.train import get_device

LOC_JSON = PROJECT_ROOT / "reports" / "gradcam_localization.json"
LOC_BOXES = PROJECT_ROOT / "reports" / "gradcam_localization_boxes.csv"
TOL_PX = 15  # pointing-game tolerance in CAM pixels; fixed BEFORE looking at results


def scale_box(row, size: int) -> tuple[float, float, float, float]:
    """
    Box (x, y, w, h) in original 1024-px coordinates -> (x0, y0, x1, y1) in CAM pixels.
    """
    s = size / ORIG_SIZE

    return row.x * s, row.y * s, (row.x + row.w) * s, (row.y + row.h) * s


def box_mask(box: tuple[float, float, float, float], size: int) -> np.ndarray:
    """
    (size, size) bool mask: True where the pixel CENTER lies inside the box.
    """
    x0, y0, x1, y1 = box
    centers = np.arange(size) + 0.5
    cols = (centers >= x0) & (centers < x1)
    rows = (centers >= y0) & (centers < y1)

    return rows[:, None] & cols[None, :]


def box_metrics(
    cam: np.ndarray, box: tuple[float, float, float, float], tol: float = TOL_PX
) -> dict:
    """
    Pointing hit and heat inside one box.

    hit: hottest pixel inside the box grown by `tol` px on each side.
    lift = mass_frac / area_frac (1.0 = no better than a uniform map).
    A flat map (valid Grad-CAM output) counts as a miss with NaN mass/lift.
    """
    assert cam.ndim == 2 and cam.shape[0] == cam.shape[1], (
        f"expected square map, got {cam.shape}"
    )
    mask = box_mask(box, cam.shape[0])
    assert mask.any(), f"box {box} covers no pixel centers"
    area_frac = float(mask.mean())

    if cam.max() == cam.min():
        return {
            "hit": False,
            "mass_frac": np.nan,
            "area_frac": area_frac,
            "lift": np.nan,
            "flat": True,
        }
    r, c = np.unravel_index(cam.argmax(), cam.shape)
    x0, y0, x1, y1 = box
    hit = (x0 - tol <= c + 0.5 < x1 + tol) and (y0 - tol <= r + 0.5 < y1 + tol)
    mass_frac = float(cam[mask].sum() / cam.sum())

    return {
        "hit": bool(hit),
        "mass_frac": mass_frac,
        "area_frac": area_frac,
        "lift": mass_frac / area_frac,
        "flat": False,
    }


def summarize(d: pd.DataFrame) -> dict:
    """
    Summary of a group of boxes; lifts use the median (ratios are skewed; NaN skipped).
    """

    return {
        "n": len(d),
        "flagged": float(d["flagged"].mean()),
        "hit": float(d["hit"].mean()),
        "hit_shuf": float(d["hit_shuf"].mean()),
        "lift": float(d["lift"].median()),
        "lift_shuf": float(d["lift_shuf"].median()),
    }


def main() -> None:
    # fail fast: cheap lookups before the model and the long loop
    boxes = load_bboxes()
    thresholds = json.loads(THRESHOLDS.read_text())
    ds = NIHDataset(split="test")
    index = {p.name: i for i, p in enumerate(ds.image_paths)}
    missing = set(boxes[IMAGE_COL]) - set(index)
    assert not missing, f"{len(missing)} bbox images are not in the test split"
    device = get_device()
    model = load_model(str(device))
    print(f"{len(boxes)} boxes, {boxes[IMAGE_COL].nunique()} images, device {device}")

    records = []
    for label, group in boxes.groupby("label", sort=True):
        idx = LABELS.index(label)
        cams, logits = {}, {}
        for image_id in tqdm(group[IMAGE_COL].unique(), desc=label):
            x, _ = ds[index[image_id]]
            cams[image_id], logits[image_id] = gradcam(model, x, idx)
        # control: pair each row with the box of the next row of the same label (another image)
        shuffled = group.iloc[np.roll(np.arange(len(group)), 1)]
        for row, other in zip(group.itertuples(), shuffled.itertuples(), strict=True):
            image_id = getattr(row, IMAGE_COL)
            cam = cams[image_id]
            real = box_metrics(cam, scale_box(row, cam.shape[0]))
            ctrl = box_metrics(cam, scale_box(other, cam.shape[0]))
            records.append(
                {
                    IMAGE_COL: image_id,
                    "label": label,
                    "logit": logits[image_id],
                    "flagged": logits[image_id] >= thresholds[label]["threshold_logit"],
                    **real,
                    "hit_shuf": ctrl["hit"],
                    "lift_shuf": ctrl["lift"],
                    "self_match": image_id == getattr(other, IMAGE_COL),
                }
            )

    df = pd.DataFrame(records)
    assert len(df) == len(boxes), f"lost boxes: {len(df)} of {len(boxes)}"
    per_label = {label: summarize(d) for label, d in df.groupby("label")}
    per_label["ALL"] = summarize(df)
    by_flagged = {
        ("flagged" if f else "not_flagged"): summarize(d)
        for f, d in df.groupby("flagged")
    }

    print(
        f"\ntolerance {TOL_PX} px; flat maps {int(df['flat'].sum())}; "
        f"shuffled self-matches {int(df['self_match'].sum())}"
    )
    print(pd.DataFrame.from_dict(per_label, orient="index").round(3).to_string())
    print()
    print(pd.DataFrame.from_dict(by_flagged, orient="index").round(3).to_string())

    out = {
        "split": "test",
        "tol_px": TOL_PX,
        "n_boxes": len(df),
        "n_flat": int(df["flat"].sum()),
        "n_self_match": int(df["self_match"].sum()),
        "by_flagged": by_flagged,
        "per_label": per_label,
    }
    LOC_JSON.write_text(json.dumps(out, indent=2) + "\n")
    df.to_csv(LOC_BOXES, index=False)
    print(
        f"\nsaved {LOC_JSON.relative_to(PROJECT_ROOT)}, {LOC_BOXES.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()
