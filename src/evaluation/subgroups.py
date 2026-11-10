"""Per-label AUROC inside patient subgroups of the test set (Step 9f).

Subgroups: sex, age band, view position (AP vs PA). Read-only analysis of test:
nothing is chosen here. Writes reports/subgroup_metrics.json.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.nih import LABELS, PROJECT_ROOT
from src.evaluation.metrics import load_predictions, per_class_auroc

SUBGROUP_METRICS = PROJECT_ROOT / "reports" / "subgroup_metrics.json"

# Column names in splits.csv after the nih.py renames (verified with head -1).
SEX_COL = "patient_sex"
AGE_COL = "patient_age"
VIEW_COL = "view_position"

AGE_BINS = [0, 40, 60, np.inf]  # right=False -> [0, 40), [40, 60), [60, inf)
AGE_LABELS = ["<40", "40-59", "60+"]
MIN_POS = 30  # fewer positives (or negatives) than this in a group -> AUROC too noisy

EXPECTED_GROUPS = {
    "sex": {"M", "F"},
    "age": set(AGE_LABELS),
    "view": {"AP", "PA"},
}


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with an "age_band" column (NaN where age is NaN)."""
    df = df.copy()
    df["age_band"] = pd.cut(df[AGE_COL], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return df


def subgroup_auroc(df: pd.DataFrame, group_col: str) -> dict:
    """Per-label AUROC inside each group of group_col.

    Args:
        df: Output of load_predictions (+ any extra grouping columns).
        group_col: column whose values define the groups.

    Returns:
        {group: {"n_images": int, "auroc": {label: float or None},
                 "n_pos": {label: int}}}
        AUROC is None when the group has fewer than MIN_POS positives or negatives.
    """
    # groupby silently DROPS rows whose group value is NaN -> count them explicitly.
    n_missing = int(df[group_col].isna().sum())
    print(f"{group_col}: {n_missing} images without a value (excluded)")

    out = {}
    for group, g in df.groupby(group_col, observed=True):
        auroc, n_pos = {}, {}
        for label in LABELS:
            pos = int(g[label].sum())
            neg = len(g) - pos
            n_pos[label] = pos
            if min(pos, neg) < MIN_POS:
                auroc[label] = None  # reported as "not enough data", never as a number
            else:
                auroc[label] = float(roc_auc_score(g[label], g[f"logit_{label}"]))
        out[str(group)] = {"n_images": len(g), "auroc": auroc, "n_pos": n_pos}

    n_grouped = sum(v["n_images"] for v in out.values())
    assert n_grouped + n_missing == len(df), f"{group_col}: rows lost in groupby"
    return out


def print_table(name: str, overall: dict, result: dict) -> None:
    """Print one table: label, overall AUROC, then AUROC per group ('-' = too few)."""
    groups = list(result)
    sizes = "  ".join(f"{g}: n={result[g]['n_images']}" for g in groups)
    print(f"\n== {name} ==  {sizes}")
    print(f"{'label':<20}{'all':>8}" + "".join(f"{g:>8}" for g in groups))

    for label in LABELS:
        cells = ""
        for g in groups:
            a = result[g]["auroc"][label]
            cells += f"{'-':>8}" if a is None else f"{a:>8.3f}"
        print(f"{label:<20}{overall[label]:>8.3f}{cells}")

    # Mean only over labels available in EVERY group, so the columns are comparable.
    common = [
        lab
        for lab in LABELS
        if all(result[g]["auroc"][lab] is not None for g in groups)
    ]
    all_mean = np.mean([overall[lab] for lab in common])
    group_means = "".join(
        f"{np.mean([result[g]['auroc'][lab] for lab in common]):>8.3f}" for g in groups
    )
    print(f"{f'MEAN ({len(common)} labels)':<20}{all_mean:>8.3f}{group_means}")


def main() -> None:
    """Subgroup AUROC on test for sex, age band and view; print tables, save JSON."""
    df = add_age_band(load_predictions("test"))
    overall = per_class_auroc(df)["auroc"]

    results = {}
    for name, col in (("sex", SEX_COL), ("age", "age_band"), ("view", VIEW_COL)):
        results[name] = subgroup_auroc(df, col)
        assert set(results[name]) == EXPECTED_GROUPS[name], (
            f"{name}: groups {set(results[name])} != expected {EXPECTED_GROUPS[name]}"
        )
        print_table(name, overall, results[name])

    with open(SUBGROUP_METRICS, "w") as f:
        json.dump(
            {
                "split": "test",
                "min_pos": MIN_POS,
                "overall": overall,
                "subgroups": results,
            },
            f,
            indent=2,
        )
    print(f"\nsaved {SUBGROUP_METRICS}")


if __name__ == "__main__":
    main()
