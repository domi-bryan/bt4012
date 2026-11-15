"""Role C: holdout blocks and CV fold generators (Step 3).

Every function returns positional row indices into the time-sorted table from `src.data`,
so every model trains and scores on identical rows. Do not re-sort the table.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from src.data import CLEANED_PATH

BIN_DAYS = 30


def holdout_indices(df):
    """Chronological 80/10/10 split. Returns (train_idx, val_idx, test_idx).

    Validation and test each take len(df) // 10 rows; train takes the rest. On the full
    table that is rows 0-472,431, then 59,054 and 59,054.
    """
    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Table must be sorted by TransactionDT")
    n = len(df)
    n_hold = n // 10
    n_train = n - 2 * n_hold
    idx = np.arange(n)
    return idx[:n_train], idx[n_train:n_train + n_hold], idx[n_train + n_hold:]


def time_cv_folds(df, train_idx):
    """Expanding-window CV over 30-day bins of the training rows.

    Bin = (TransactionDay - first training day) // 30. Fold k trains on bins 0..k-1 and
    validates on bin k, for k = 1..last bin. Returns a list of (train_idx, val_idx).
    """
    train_idx = np.asarray(train_idx)
    day = df["TransactionDay"].to_numpy()[train_idx].astype(np.int64)
    bins = (day - day.min()) // BIN_DAYS
    return [(train_idx[bins < k], train_idx[bins == k]) for k in range(1, bins.max() + 1)]


def _check_aligned(train_idx, *arrays):
    for a in arrays:
        if len(a) != len(train_idx):
            raise ValueError("y and groups must be aligned with train_idx (one value per row)")


def random_cv_folds(train_idx, y, k=5, *, seed):
    """Shuffled StratifiedKFold over the train block. Leakage check only.

    y[i] is the label of row train_idx[i]. Returns a list of (train_idx, val_idx).
    """
    train_idx = np.asarray(train_idx)
    _check_aligned(train_idx, y)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(train_idx[a], train_idx[b]) for a, b in skf.split(train_idx, y)]


def group_cv_folds(train_idx, y, groups, k=5, *, seed):
    """Shuffled StratifiedGroupKFold over the train block. Leakage check only.

    y[i] and groups[i] belong to row train_idx[i]. Returns a list of (train_idx, val_idx).
    """
    train_idx = np.asarray(train_idx)
    _check_aligned(train_idx, y, groups)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    return [(train_idx[a], train_idx[b]) for a, b in sgkf.split(train_idx, y, groups)]


def _summary(name, idx, df):
    dt = df["TransactionDT"].to_numpy()[idx]
    day = df["TransactionDay"].to_numpy()[idx]
    return {"block": name, "rows": len(idx), "fraud": int(df["isFraud"].to_numpy()[idx].sum()),
            "fraud_rate_pct": 100 * df["isFraud"].to_numpy()[idx].mean(),
            "first_day": int(day.min()), "last_day": int(day.max()),
            "min_dt": int(dt.min()), "max_dt": int(dt.max())}


def split_summary(df):
    """Row count and fraud rate of each holdout block and each time-CV fold."""
    tr, va, te = holdout_indices(df)
    rows = [_summary(n, i, df) for n, i in (("train", tr), ("validation", va), ("test", te))]
    for k, (a, b) in enumerate(time_cv_folds(df, tr), start=1):
        rows.append(_summary(f"fold{k}_train", a, df))
        rows.append(_summary(f"fold{k}_val", b, df))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cols = ["TransactionDT", "TransactionDay", "isFraud"]
    table = split_summary(pd.read_parquet(CLEANED_PATH, columns=cols))
    print(table.to_string(index=False, float_format="%.4f"))
