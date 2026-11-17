"""Blending the saved model scores: weights chosen on the time-CV fold scores, checked on validation.

Scores from different models live on different scales, so each model's scores are turned into
ranks in (0, 1] within each fold (fold scores) or within the block (validation scores) before
blending. A blend is a weighted sum of those ranks with non-negative weights that sum to 1.

A segment blend uses one set of weights for known customers and another for unknown ones
(`src.segments`); "known" needs no labels, so it is available when a transaction arrives.

Weights are always searched on the fold scores, never on the validation block, so the
validation PR-AUC of the chosen blend is an honest check. Nothing here writes to the log.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import segments


def weight_grid(models, step=0.1):
    """Every weighting of `models` in multiples of `step` that sums to 1, as a list of dicts."""
    n = round(1 / step)
    if not np.isclose(n * step, 1):
        raise ValueError("step must divide 1")
    out = []
    for c in itertools.product(range(n + 1), repeat=len(models)):
        if sum(c) == n:
            out.append({m: round(k * step, 10) for m, k in zip(models, c)})
    return out


def rank_within(df, models, by=None):
    """Copy of `df` with each model column replaced by its percentile rank, within groups of `by`."""
    out = df.copy()
    for m in models:
        out[m] = (out.groupby(by)[m] if by else out[m]).rank(pct=True)
    return out


def blend(df, weights):
    """Weighted sum of the (ranked) model columns."""
    return sum(df[m].to_numpy(dtype="float64") * w for m, w in weights.items() if w)


def segment_blend(df, known, weights_known, weights_unknown):
    """Blend known rows with `weights_known` and unknown rows with `weights_unknown`."""
    known = np.asarray(known, dtype=bool)
    return np.where(known, blend(df, weights_known), blend(df, weights_unknown))


def fold_pr_auc(y, score, fold):
    """Mean PR-AUC over folds (the same summary as the tuning runs)."""
    y, score, fold = np.asarray(y), np.asarray(score), np.asarray(fold)
    return float(np.mean([average_precision_score(y[fold == k], score[fold == k]) for k in np.unique(fold)]))


def search(df, models, step=0.1, mask=None, label="isFraud", fold="fold"):
    """Score every weighting on the fold scores (rows in `mask` only); best first.

    Returns one row per weighting: a column per model with its weight, and `oof_pr_auc`.
    """
    part = df if mask is None else df[np.asarray(mask, dtype=bool)]
    rows = []
    for w in weight_grid(models, step):
        rows.append({**w, "oof_pr_auc": fold_pr_auc(part[label], blend(part, w), part[fold])})
    return pd.DataFrame(rows).sort_values("oof_pr_auc", ascending=False, ignore_index=True)


def best_weights(table, models):
    """The weights in the first row of a `search` table, as a dict."""
    return {m: float(table.iloc[0][m]) for m in models}


def load_scores(preds_dir, models, feature_set, kind, seed=0):
    """One table of saved scores for `models` on the same rows: TransactionID, isFraud, (fold), one column per model.

    `kind` is "oof" (time-CV fold scores, seed 0 only) or "val" (validation scores for `seed`).
    Raises if the models' files cover different rows or disagree on the label.
    """
    name = "oof_seed0" if kind == "oof" else f"val_seed{seed}"
    keep = ["TransactionID", "isFraud"] + (["fold"] if kind == "oof" else [])
    table = None
    for m in models:
        p = pd.read_parquet(preds_dir / f"{m}_{feature_set}_{name}.parquet", columns=keep + ["score"])
        p = p.rename(columns={"score": m})
        if table is None:
            table = p
            continue
        if len(p) != len(table) or not p["TransactionID"].isin(table["TransactionID"]).all():
            raise ValueError(f"{m}: saved {kind} scores cover different rows from {models[0]}")
        merged = table.merge(p, on="TransactionID", suffixes=("", "_other"), validate="1:1")
        for c in keep[1:]:
            if not (merged[c] == merged[f"{c}_other"]).all():
                raise ValueError(f"{m}: column {c} disagrees with {models[0]}")
        table = merged[list(table.columns) + [m]]
    return table


def oof_table(preds_dir, models, feature_set, ids, train_idx):
    """Fold scores ranked within fold, with `known` per fold (uid seen in that fold's training months)."""
    t = rank_within(load_scores(preds_dir, models, feature_set, "oof"), models, by="fold")
    ref = segments.fold_reference_uids(ids, train_idx)
    uid = pd.Series(ids["uid"].to_numpy(), index=ids["TransactionID"].to_numpy()).reindex(t["TransactionID"])
    t["known"] = [u in ref[k] for u, k in zip(uid.to_numpy(), t["fold"].to_numpy())]
    return t


def val_table(preds_dir, models, feature_set, ids, train_idx, seed):
    """Validation scores for one seed, ranked within the block, with `known` (uid in the training block)."""
    t = rank_within(load_scores(preds_dir, models, feature_set, "val", seed), models)
    uid = pd.Series(ids["uid"].to_numpy(), index=ids["TransactionID"].to_numpy()).reindex(t["TransactionID"])
    t["known"] = segments.known_uid(uid, set(ids["uid"].iloc[train_idx]))
    return t
