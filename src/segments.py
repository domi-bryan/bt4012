"""Known vs unknown customers, and PR-AUC per segment with bootstrap intervals.

A row is **known** if its `uid` already appears in the reference rows (the rows the model was
trained on), otherwise **unknown**. The label only uses information available when the
transaction arrives, so it can also be used to weight or route models in an ensemble.

- Validation and test rows: the reference is the training block.
- Time-CV fold scores: the reference is that fold's training months (`fold_reference_uids`).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.split import time_cv_folds

SEGMENTS = ("known", "unknown")


def known_uid(uids, reference_uids):
    """Boolean array: True where the uid appears in `reference_uids`."""
    reference = reference_uids if isinstance(reference_uids, (set, frozenset)) else set(reference_uids)
    return pd.Series(uids).isin(reference).to_numpy()


def segment_labels(known):
    return np.where(np.asarray(known, dtype=bool), "known", "unknown")


def fold_reference_uids(df, train_idx):
    """{fold k: set of uids in fold k's training part}, using the same folds as `src.experiment`."""
    uid = df["uid"].to_numpy()
    return {k: set(uid[a]) for k, (a, _) in enumerate(time_cv_folds(df, train_idx), start=1)}


def bootstrap_pr_auc(y, scores, n_boot=200, seed=0):
    """Point PR-AUC and a 95% percentile interval, resampling rows with replacement.

    `scores` is one array or a list of arrays (e.g. one per seed) for the same rows; with several,
    the point and every resample use the mean PR-AUC over them, so the interval is for that mean.
    """
    y = np.asarray(y)
    runs = [np.asarray(s, dtype="float64") for s in (scores if isinstance(scores, (list, tuple)) else [scores])]
    point = float(np.mean([average_precision_score(y, s) for s in runs]))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        stats.append(np.mean([average_precision_score(y[i], s[i]) for s in runs]))
    lo, hi = np.percentile(stats, [2.5, 97.5]) if stats else (np.nan, np.nan)
    return point, float(lo), float(hi)


def segment_pr_auc(y, scores, known, n_boot=200, seed=0):
    """One row per segment: rows, fraud cases, PR-AUC (mean over score arrays) and its 95% interval."""
    y = np.asarray(y)
    runs = scores if isinstance(scores, (list, tuple)) else [scores]
    rows = []
    for name, mask in (("known", np.asarray(known, dtype=bool)), ("unknown", ~np.asarray(known, dtype=bool))):
        part = [np.asarray(s)[mask] for s in runs]
        if mask.sum() == 0 or y[mask].min() == y[mask].max():
            rows.append({"segment": name, "rows": int(mask.sum()), "fraud": int(y[mask].sum()),
                         "pr_auc": np.nan, "low": np.nan, "high": np.nan})
            continue
        point, lo, hi = bootstrap_pr_auc(y[mask], part, n_boot=n_boot, seed=seed)
        rows.append({"segment": name, "rows": int(mask.sum()), "fraud": int(y[mask].sum()),
                     "pr_auc": point, "low": lo, "high": hi})
    return pd.DataFrame(rows)
