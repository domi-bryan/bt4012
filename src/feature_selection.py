"""Choosing model inputs on the training block: V-column pruning and a time-stability check.

Both follow the EDA (notebooks/eda_features.ipynb sections 3, 11.1 and 13) and use training-block rows only.

Time-stability check: for one column at a time, learn the fraud rate per value (or per quantile bin) on the
first 30 days of the training block, use those rates to score the last 30 days, and compute the AUC there.

The early-window AUC is measured out of sample: the window is split into two halves by time, and each half is
scored with rates learned on the other half. (Scoring the same rows the rates were learned on makes pure noise
look like a signal of about 0.52.) Intervals use the Hanley-McNeil standard error. A column is only called
unstable on strong evidence:

- "holds later": late AUC above 0.5 (95% interval)
- "reversed later": late AUC below 0.5 (99.9% interval, because hundreds of columns are tested)
- "faded later": neither, but the early AUC is above 0.5 (99.9% interval)
- "weak in both windows": no clear signal in either window (typical for columns filled on very few rows)
- "constant early": only one value in the early window, so it can't be tested
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_profile import prune_group, v_groups

V_THRESHOLD = 0.75          # chosen with notebooks/logreg_v_threshold.ipynb
WINDOW_DAYS = 30
N_BINS = 20
SMOOTH = 50
Z_HOLDS = 1.96              # 95%, one side
Z_STRICT = 3.29             # 99.9%, used before calling a column reversed or faded
UNSTABLE = ("reversed later", "faded later")


def prune_v(train, threshold=V_THRESHOLD):
    """V columns kept and dropped by greedy correlation pruning within missing-pattern groups."""
    keep, drop = [], []
    for cols in v_groups(train).values():
        k, d, _ = prune_group(train, cols, threshold)
        keep += k
        drop += d
    order = {c: i for i, c in enumerate(train.columns)}
    return sorted(keep, key=order.get), sorted(drop, key=order.get)


def _bins(col_early, col_late):
    """Bins from the early window only. 20 or fewer distinct values: each value is a bin. Otherwise 20
    quantile bins. Missing is its own bin."""
    numeric = pd.api.types.is_numeric_dtype(col_early) and not pd.api.types.is_bool_dtype(col_early)
    if not numeric or col_early.nunique() <= N_BINS:
        return col_early.astype(object).fillna("(missing)"), col_late.astype(object).fillna("(missing)")
    x_e, x_l = col_early.astype("float64"), col_late.astype("float64")
    cuts = np.unique(np.nanquantile(x_e, np.linspace(0, 1, N_BINS + 1)))[1:-1]

    def b(x):
        out = pd.Series(np.searchsorted(cuts, x.to_numpy(), side="right"), index=x.index).astype(object)
        return out.where(x.notna(), "(missing)")
    return b(x_e), b(x_l)


def auc_se(auc, n_pos, n_neg):
    """Hanley-McNeil standard error of an AUC."""
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    return np.sqrt((auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2))
                   / (n_pos * n_neg))


def _rates(y, bins):
    g = y.groupby(bins).agg(["sum", "size"])
    return (g["sum"] + SMOOTH * y.mean()) / (g["size"] + SMOOTH)


def _auc(y, score):
    return roc_auc_score(y, score) if score.nunique() > 1 else 0.5


def time_stability(frame, cols, day_col="TransactionDay", target="isFraud", window_days=WINDOW_DAYS):
    """One row per column: out-of-sample AUC in the early window, AUC in the late window (rates learned on
    the whole early window), the late AUC's 95% interval, and the result. `frame` must be training rows."""
    day = frame[day_col]
    early = frame[day <= day.min() + window_days - 1]
    late = frame[day >= day.max() - window_days + 1]
    y_e, y_l = early[target], late[target]
    half = pd.Series(np.arange(len(early)) * 2 // len(early), index=early.index)   # two halves by time
    rows = {}
    for c in cols:
        b_e, b_l = _bins(early[c], late[c])
        if b_e.nunique() < 2:
            rows[c] = (np.nan, np.nan)
            continue
        oof = pd.Series(np.nan, index=early.index)
        for h in (0, 1):
            fit, score = half != h, half == h
            oof[score] = b_e[score].map(_rates(y_e[fit], b_e[fit])).astype("float64").fillna(y_e[fit].mean())
        s_l = b_l.map(_rates(y_e, b_e)).astype("float64").fillna(y_e.mean())
        rows[c] = (_auc(y_e, oof), _auc(y_l, s_l))
    out = pd.DataFrame(rows, index=["auc_early", "auc_late"]).T
    pos_e, pos_l = y_e.sum(), y_l.sum()
    se_e = auc_se(out["auc_early"], pos_e, len(y_e) - pos_e)
    se_l = auc_se(out["auc_late"], pos_l, len(y_l) - pos_l)
    out["late_low"], out["late_high"] = out["auc_late"] - Z_HOLDS * se_l, out["auc_late"] + Z_HOLDS * se_l
    out["result"] = np.select(
        [out["auc_late"] - Z_HOLDS * se_l > 0.5,
         out["auc_late"] + Z_STRICT * se_l < 0.5,
         out["auc_early"] - Z_STRICT * se_e > 0.5],
        ["holds later", "reversed later", "faded later"], "weak in both windows")
    out.loc[out["auc_late"].isna(), "result"] = "constant early"
    return out
