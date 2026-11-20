"""Evaluation metrics (Step 4). PR-AUC is primary; the no-skill baseline is the fraud rate."""
import numpy as np
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

TOP_FRACTION = 0.01


def recall_top1pct(y_true, y_score, fraction=TOP_FRACTION):
    """Share of all fraud rows that fall in the top `fraction` of scores.

    The top slice holds ceil(fraction * n) rows. Rows tied with the cutoff score share
    the remaining slots evenly, so the result does not depend on row order.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = y_true.sum()
    if n_pos == 0:
        return np.nan
    k = int(np.ceil(fraction * len(y_score)))
    cutoff = np.sort(y_score)[::-1][k - 1]
    above = y_score > cutoff
    tied = y_score == cutoff
    slots = k - above.sum()
    caught = y_true[above].sum() + slots * y_true[tied].sum() / tied.sum()
    return float(caught / n_pos)


def top_rows(n, fraction=TOP_FRACTION):
    """Number of rows in the top `fraction` of a block of n rows: ceil(fraction * n)."""
    return int(np.ceil(fraction * n))


def precision_top1pct(y_true, y_score, fraction=TOP_FRACTION):
    """Share of the top `fraction` of scores that are fraud (ties shared as in `recall_top1pct`).

    For the same rows it equals recall_top1pct * n_fraud / top_rows(n), so both rank models the
    same way; precision reads more easily because recall here is capped at top_rows(n) / n_fraud.
    """
    y_true = np.asarray(y_true)
    return float(recall_top1pct(y_true, y_score, fraction) * y_true.sum() / top_rows(len(y_true), fraction))


def pick_threshold(y_true, y_score):
    """Threshold that maximises F1 on the given (validation) data; predict fraud when score >= it."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    p, r = precision[:-1], recall[:-1]
    denom = p + r
    f1 = np.divide(2 * p * r, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[np.argmax(f1)])


def evaluate(y_true, y_score, threshold=None):
    """pr_auc, auroc, precision, recall, recall_top1pct. Precision and recall are None without a threshold."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "precision": None,
        "recall": None,
        "recall_top1pct": recall_top1pct(y_true, y_score),
    }
    if threshold is not None:
        y_pred = (y_score >= threshold).astype(int)
        out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    return out
