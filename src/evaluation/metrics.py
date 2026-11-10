"""
Per class AUROC on val and test from saved logits.

Reads data/processed/predictions/preds_{splits}.csv (IDs+logits), join labels and metadata from splits.csv
Writes reports/eval_metrics.json
"""

import json

import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data.make_splits import OUT_PATH as SPLITS_CSV
from src.data.nih import LABELS, PROJECT_ROOT
from src.evaluation.predict import IMAGE_COL, PRED_DIR

EVAL_METRICS = PROJECT_ROOT / "reports" / "eval_metrics.json"
SPLITS = ("val", "test")


def load_predictions(split: str) -> pd.DataFrame:
    preds = pd.read_csv(PRED_DIR / f"preds_{split}.csv")
    splits = pd.read_csv(SPLITS_CSV)

    df = preds.merge(splits, on=IMAGE_COL, how="left", validate="one_to_one")
    assert len(df) == len(preds), f"{split}: merge changed row count"

    n_missing = int(df[LABELS].isna().any(axis=1).sum())
    assert n_missing == 0, f"{split}: predicted images have no labels"

    assert (df["split"] == split).all(), f"{split}: file contains other splits"

    return df


def per_class_auroc(df: pd.DataFrame) -> dict:
    """
    Per-label AUROC computed on logits, plus the macro mean.

    Logits, not sigmoid: float32 sigmoid saturates above ~17 -> ties -> wrong ranks.

    Args:
        df: Output of load_predictions (labels + logit_<label> columns).

    Returns:
        {"mean_auroc": float, "auroc": {label: float},
         "n_pos": {label: int}, "n_images": int}
    """
    auroc, n_pos = {}, {}
    for label in LABELS:
        y = df[label]
        if y.nunique() != 2:
            raise ValueError(f"{label}: AUROC needs positives and negatives")
        auroc[label] = float(roc_auc_score(y, df[f"logit_{label}"]))
        n_pos[label] = int(y.sum())

    return {
        "mean_auroc": sum(auroc.values())
        / len(auroc),  # macro: every class counts equally
        "auroc": auroc,
        "n_pos": n_pos,
        "n_images": len(df),
    }


def main() -> None:
    """Compute per-class AUROC for val and test, print a comparison, save JSON."""
    results = {split: per_class_auroc(load_predictions(split)) for split in SPLITS}
    val, test = results["val"], results["test"]

    print(f"{'label':<20}{'val':>8}{'test':>8}{'diff':>8}{'test_pos':>10}")
    for label in sorted(LABELS, key=lambda lab: test["auroc"][lab], reverse=True):
        v, t = val["auroc"][label], test["auroc"][label]
        print(f"{label:<20}{v:>8.4f}{t:>8.4f}{t - v:>+8.4f}{test['n_pos'][label]:>10}")
    diff = test["mean_auroc"] - val["mean_auroc"]
    print(
        f"{'MEAN':<20}{val['mean_auroc']:>8.4f}{test['mean_auroc']:>8.4f}{diff:>+8.4f}"
    )

    with open(EVAL_METRICS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {EVAL_METRICS}")


if __name__ == "__main__":
    main()
