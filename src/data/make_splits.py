from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit
from src.data.nih import LABELS, NIH_DIR, PROJECT_ROOT, load_labels

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "splits.csv"

KEEP_COLS = [
    "image_index",
    "patient_id",
    "split",
    "patient_age",
    "patient_sex",
    "view_position",
] + LABELS


def main():
    # 1. Load parameters from params.yaml for DVC reproducibility
    params_path = PROJECT_ROOT / "params.yaml"
    print(f"Loading parameters from {params_path}...")
    params = yaml.safe_load(params_path.read_text())["splits"]
    val_frac = params["val_frac"]
    seed = params["seed"]
    print(f"-> Configuration loaded: val_frac={val_frac}, seed={seed}")

    # Load metadata and official test list
    data_dir = NIH_DIR
    metadata_path = data_dir / "Data_Entry_2017.csv"
    test_list_path = data_dir / "test_list.txt"

    print(f"Loading Metadata from {metadata_path}")
    df = load_labels(metadata_path)

    print(f"Loading test list from {test_list_path}")
    test_set = set(test_list_path.read_text().split())

    # Test split and Validate size
    df["split"] = "train"
    is_test = df["image_index"].isin(test_set)
    df.loc[is_test, "split"] = "test"

    # Fail-loud check: assert number of test rows equals len(test_set)
    actual_test_count = is_test.sum()
    expected_test_count = len(test_set)
    assert actual_test_count == expected_test_count, (
        f"Test row count mismatch! Expected {expected_test_count}, got {actual_test_count}."
    )
    print(f"Successfully verified test split: {actual_test_count} images.")

    # Processing multihot labels safely
    print("Building Multi-Hot Labels")
    label_list = df["finding_labels"].str.split("|")

    for label in LABELS:
        df[label] = label_list.apply(lambda lst: 1 if label in lst else 0)

    # Clean Age, Outlier detection
    if "patient_age" in df.columns:
        df["patient_age"] = pd.to_numeric(df["patient_age"], errors="coerce")
        invalid_age = (df["patient_age"] < 0) | (df["patient_age"] > 120)
        df.loc[invalid_age, "patient_age"] = np.nan

    # Group Shuffle Split using YAML parameters
    print("Performing patient-level train/val split using GroupShuffleSplit")
    rest_mask = df["split"] != "test"
    rest = df[rest_mask].copy()

    gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(gss.split(rest, groups=rest["patient_id"]))

    val_index_labels = rest.iloc[val_idx].index
    df.loc[val_index_labels, "split"] = "val"

    # Patient level overlap check across splits
    print("Running patient overlap checks")
    train_patients = set(df[df["split"] == "train"]["patient_id"])
    val_patients = set(df[df["split"] == "val"]["patient_id"])
    test_patients = set(df[df["split"] == "test"]["patient_id"])

    # testing
    assert not (train_patients & val_patients), "train/val patient overlap"
    assert not (train_patients & test_patients), "train/test patient overlap"
    assert not (val_patients & test_patients), "val/test patient overlap"
    print("Patient overlap check passed: all splits are strictly disjoint by patient ID.")

    print("\nDataset Split Summary")
    print(df["split"].value_counts())

    # Keep only specified columns for the processed output
    df_out = df[KEEP_COLS]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving splits to {OUT_PATH}...")
    df_out.to_csv(OUT_PATH, index=False)
    print("Done!")


if __name__ == "__main__":
    main()