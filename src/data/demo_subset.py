"""
Fixed demo subset of NIH test images: select, convert to DICOM, upload to Orthanc.

For education only, not clinical use.

    python -m src.data.demo_subset            # select + convert
    python -m src.data.demo_subset --upload   # ... and upload to Orthanc
"""

import argparse
import base64
import json
import os
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dicom_io import png_to_dicom
from src.data.make_splits import OUT_PATH
from src.data.nih import LABELS, NIH_DIR, PROJECT_ROOT, find_images, load_labels

BBOX_CSV = NIH_DIR / "BBox_List_2017.csv"
SUBSET_CSV = PROJECT_ROOT / "data" / "demo" / "demo_subset.csv"
DICOM_DIR = PROJECT_ROOT / "data" / "interim" / "demo_dicom"

SEED = 42
N_BOX_PER_LABEL = 2  # 8 box labels -> 16 images
N_BOX_EXTRA = 4  # -> 20 box images
N_NORMAL = 10
N_RANDOM = 10

ORTHANC_URL = os.environ.get("ORTHANC_URL", "http://localhost:8042")
ORTHANC_USER = os.environ.get("ORTHANC_USER", "demo")
ORTHANC_PASSWORD = os.environ.get("ORTHANC_PASSWORD", "demo")


def _pick(rng, candidates, n, exclude) -> list[str]:
    """
    Sample n ids from candidates minus exclude; sorted first so it's deterministic.
    """
    pool = sorted(set(candidates) - set(exclude))

    assert len(pool) >= n, f"need {n}, only {len(pool)} candidates"

    return [str(x) for x in rng.choice(pool, size=n, replace=False)]


def select_subset(splits: pd.DataFrame, boxes: pd.DataFrame) -> pd.DataFrame:
    test = splits[splits["split"] == "test"]
    test_ids = set(test["image_index"])
    box_ids = set(boxes["Image Index"])

    assert box_ids <= test_ids, f"{len(box_ids - test_ids)} box images not in test"

    rng = np.random.default_rng(SEED)
    picked: list[str] = []

    for _, grp in boxes.groupby("Finding Label"):  # groupby sorts the labels
        picked += _pick(rng, grp["Image Index"], N_BOX_PER_LABEL, picked)
    picked += _pick(rng, box_ids, N_BOX_EXTRA, picked)
    groups = dict.fromkeys(picked, "box")

    normal_mask = (test[LABELS].sum(axis=1) == 0).to_numpy()
    normal_ids = test.loc[normal_mask, "image_index"]
    for img_id in _pick(rng, normal_ids, N_NORMAL, groups):
        groups[img_id] = "no_finding"
    for img_id in _pick(rng, test_ids, N_RANDOM, groups):
        groups[img_id] = "random"

    box_labels = boxes.groupby("Image Index")["Finding Label"].agg(
        lambda s: "|".join(sorted(set(s)))
    )
    subset = pd.DataFrame({"image_index": list(groups), "group": list(groups.values())})
    subset["box_labels"] = subset["image_index"].map(box_labels).fillna("")
    cols = ["image_index", "patient_id", "view_position", "patient_sex", "patient_age"]
    subset = subset.merge(test[cols], on="image_index", how="left", validate="1:1")

    assert subset["patient_id"].notna().all(), "subset image missing from splits"

    return subset


def convert(subset: pd.DataFrame) -> list[Path]:
    hits = sorted(NIH_DIR.glob("Data_Entry*.csv"))
    assert len(hits) == 1, hits
    meta = load_labels(hits[0]).set_index("image_index", drop=False)
    assert meta.index.is_unique, "duplicate image_index in Data_Entry"
    images = find_images(NIH_DIR)

    paths = []
    for img_id in subset["image_index"]:
        out = DICOM_DIR / f"{Path(img_id).stem}.dcm"
        png_to_dicom(images[img_id], meta.loc[img_id], out)
        paths.append(out)

    return paths


def upload(paths: list[Path]) -> Counter:
    """
    POST each file to Orthanc; raises on any HTTP error.
    """
    token = base64.b64encode(f"{ORTHANC_USER}:{ORTHANC_PASSWORD}".encode()).decode()
    headers = {"Authorization": f"Basic {token}", "Content-Type": "application/dicom"}
    statuses: Counter = Counter()

    for path in paths:
        req = urllib.request.Request(
            f"{ORTHANC_URL}/instances",
            data=path.read_bytes(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            statuses[json.load(resp)["Status"]] += 1

    return statuses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upload", action="store_true", help="also upload to Orthanc")
    args = parser.parse_args()

    splits = pd.read_csv(OUT_PATH)
    boxes = pd.read_csv(BBOX_CSV)
    assert {"Image Index", "Finding Label"} <= set(boxes.columns), list(boxes.columns)
    print(f"box rows: {len(boxes)}, box images: {boxes['Image Index'].nunique()}")

    subset = select_subset(splits, boxes)
    SUBSET_CSV.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(SUBSET_CSV, index=False)
    print(f"subset: {len(subset)} images -> {SUBSET_CSV.relative_to(PROJECT_ROOT)}")

    for col in ["group", "view_position", "patient_sex"]:
        print(f"  {col}: {subset[col].value_counts().sort_index().to_dict()}")
    labels = subset["box_labels"].str.split("|", regex=False).explode()
    print(
        f"  box labels: {labels[labels.str.len() > 0].value_counts().sort_index().to_dict()}"
    )

    paths = convert(subset)
    print(
        f"converted: {len(paths)} DICOM files -> {DICOM_DIR.relative_to(PROJECT_ROOT)}"
    )

    if args.upload:
        print(f"upload to {ORTHANC_URL}: {dict(upload(paths))}")


if __name__ == "__main__":
    main()
