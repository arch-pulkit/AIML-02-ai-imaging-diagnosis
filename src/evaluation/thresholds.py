"""
Per-label decision thresholds chosen on val, applied to test.

Truns a logit into a yes/no flag: +ve if logit >= threshold.
Writes reports/thresolds.json
"""

import json

import numpy as np
from sklearn.metrics import roc_curve

from src.data.nih import LABELS, PROJECT_ROOT
from src.evaluation.metrics import load_predictions

THRESHOLDS = PROJECT_ROOT / "reports" / "thresholds.json"


def youden_threshold(y, logits) -> float:
    """
    Logit threshold that maximises Youden's J = TPR - FPR

    Args:
        y: 0/1 labels for one class
        logits: model logits for that class, same order

    Returns:
        The threshold of the ROC point farthest above the diagonal.

    """
    fpr, tpr, thr = roc_curve(y, logits)
    best = int(np.argmax(tpr - fpr))
    return float(thr[best])


def confusion_stats(y, logits, threshold: float) -> dict:

    y = np.asarray(y).astype(bool)
    pred = np.asarray(logits) >= threshold

    tp = int((pred & y).sum())  # sick, flagged
    fn = int((~pred & y).sum())  # sick, missed
    fp = int((pred & ~y).sum())  # healthy, flagged (false alarm)
    tn = int((~pred & ~y).sum())  # healthy, not flagged
    assert tp + fn + fp + tn == len(y), "confusion counts don't add up"

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": tp / (tp + fn),  # = TPR: share of sick images caught
        "specificity": tn / (tn + fp),  # = 1 - FPR: share of healthy left alone
        "ppv": tp / (tp + fp),  # of the flagged images, share truly sick
        "prevalence": float(y.mean()),  # share of images that are sick
    }


def main() -> None:
    """Pick a threshold per label on val, evaluate it on val and test, save JSON."""
    val, test = load_predictions("val"), load_predictions("test")

    results = {}
    print(
        f"{'label':<20}{'thr':>7}{'val_sen':>9}{'val_spe':>9}"
        f"{'tst_sen':>9}{'tst_spe':>9}{'tst_ppv':>9}{'tst_prev':>9}"
    )
    for label in LABELS:
        col = f"logit_{label}"
        thr = youden_threshold(val[label], val[col])  # CHOSEN on val only
        v = confusion_stats(val[label], val[col], thr)
        t = confusion_stats(test[label], test[col], thr)  # APPLIED to test, never tuned
        results[label] = {"threshold_logit": thr, "val": v, "test": t}
        print(
            f"{label:<20}{thr:>7.2f}{v['sensitivity']:>9.3f}{v['specificity']:>9.3f}"
            f"{t['sensitivity']:>9.3f}{t['specificity']:>9.3f}"
            f"{t['ppv']:>9.3f}{t['prevalence']:>9.4f}"
        )

    with open(THRESHOLDS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {THRESHOLDS}")


if __name__ == "__main__":
    main()
