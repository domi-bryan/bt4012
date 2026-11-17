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

    `kind` is "oof" (time-CV fold scores, seed 0 only), "val" or "test" (scores for `seed`).
    Raises if the models' files cover different rows or disagree on the label.
    """
    if kind not in ("oof", "val", "test"):
        raise ValueError(f"kind must be 'oof', 'val' or 'test', got {kind!r}")
    name = "oof_seed0" if kind == "oof" else f"{kind}_seed{seed}"
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


# ---------------------------------------------------------------------------
# The adopted segment blend (chosen in notebooks/13_ensemble_check.ipynb on the time-CV fold scores).
# Fixed before the experiment freeze; nothing below searches or changes the weights.

BLEND_NAME = "segment_blend"
BLEND_SET = "G3"
BLEND_MODELS = ("xgboost", "lgbm", "catboost", "rf")
BLEND_KNOWN = {"xgboost": 1.0}
BLEND_UNKNOWN = {"catboost": 0.6, "lgbm": 0.2, "xgboost": 0.1, "rf": 0.1}
BLEND_SPEC = {"models": list(BLEND_MODELS), "known": BLEND_KNOWN, "unknown": BLEND_UNKNOWN,
              "scores": "percentile rank within the scored block", "known_means": "uid in the train block"}
REPRO_TOL = 1e-6


def blend_preds_path(preds_dir, kind, seed):
    return preds_dir / f"{BLEND_NAME}_{BLEND_SET}_{kind}_seed{seed}.parquet"


def check_inputs(log, preds_dir, kind, seed):
    """Each input model's saved `kind` scores must reproduce its current G3 row in the log for `seed`."""
    from src import experiment
    log = experiment.current_runs(log)
    for m in BLEND_MODELS:
        r = log[(log.model == m) & (log.feature_set == BLEND_SET) & (log.split == kind) & (log.seed == seed)
                & (log.cv_scheme == "holdout")]
        if len(r) != 1:
            raise ValueError(f"{m}/{BLEND_SET} seed {seed}: expected 1 current {kind} row in the log, found {len(r)}")
        path = preds_dir / f"{m}_{BLEND_SET}_{kind}_seed{seed}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"{path.name} is missing (get it from the group drive)")
        p = pd.read_parquet(path, columns=["isFraud", "score"])
        if abs(average_precision_score(p["isFraud"], p["score"]) - float(r["pr_auc"].iloc[0])) > REPRO_TOL:
            raise ValueError(f"{path.name} does not reproduce the logged {kind} PR-AUC of {m}/{BLEND_SET} "
                             f"seed {seed}; it is from another run")


def blend_scores(preds_dir, kind, seed, uid_by_id, train_uids):
    """(table, score): the segment blend of the input models' saved `kind` scores for one seed."""
    t = rank_within(load_scores(preds_dir, list(BLEND_MODELS), BLEND_SET, kind, seed), list(BLEND_MODELS))
    t["known"] = segments.known_uid(uid_by_id.reindex(t["TransactionID"]).to_numpy(), train_uids)
    return t, segment_blend(t, t["known"], BLEND_KNOWN, BLEND_UNKNOWN)


def _logged(log, split, seed):
    r = log[(log.model == BLEND_NAME) & (log.feature_set == BLEND_SET) & (log.split == split) & (log.seed == seed)]
    if len(r) > 1:
        raise ValueError(f"{BLEND_NAME} {split} seed {seed}: {len(r)} rows in the log")
    return None if r.empty else r.iloc[0]


def _save(path, t, score, seed):
    out = pd.DataFrame({"TransactionID": t["TransactionID"].to_numpy(), "isFraud": t["isFraud"].to_numpy(),
                        "score": np.asarray(score, dtype="float64"), "known": t["known"].to_numpy(), "seed": seed})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)


def log_blend(split, uid_by_id, train_uids, seeds=(0, 1, 2), log_path=None, preds_dir=None):
    """Log the blend's rows for `split` ("val" or "test") from the input models' saved scores; skip seeds done.

    Validation: the F1 threshold is chosen on the blend's validation scores, as for every model.
    Test: the threshold stored in the blend's validation row is applied unchanged.
    """
    import json
    import time
    from src import experiment
    from src.metrics import evaluate, pick_threshold
    from src.results import LOG_PATH, log_run
    log_path = log_path or LOG_PATH
    preds_dir = preds_dir or experiment.PREDS_DIR
    if split not in ("val", "test"):
        raise ValueError("split must be 'val' or 'test'")
    out = []
    for seed in seeds:
        log = pd.read_csv(log_path)
        if _logged(log, split, seed) is not None:
            print(f"{BLEND_NAME} {split} seed {seed}: already logged", flush=True)
            continue
        check_inputs(log, preds_dir, split, seed)
        t0 = time.time()
        t, score = blend_scores(preds_dir, split, seed, uid_by_id, train_uids)
        if split == "val":
            threshold = pick_threshold(t["isFraud"], score)
        else:
            v = _logged(log, "val", seed)
            if v is None:
                raise ValueError(f"{BLEND_NAME} has no validation row for seed {seed}; log validation first")
            threshold = float(json.loads(v["params"])["threshold"])
        metrics = evaluate(t["isFraud"], score, threshold)
        log_run(log_path, model=BLEND_NAME, feature_set=BLEND_SET, cv_scheme="holdout", split=split, seed=seed,
                params={**BLEND_SPEC, "threshold": threshold}, runtime_s=round(time.time() - t0, 2), **metrics)
        _save(blend_preds_path(preds_dir, split, seed), t, score, seed)
        print(f"{BLEND_NAME} {split} seed {seed}: pr_auc={metrics['pr_auc']:.4f}", flush=True)
        out.append({"seed": seed, **metrics})
    return pd.DataFrame(out)


def uid_lookup():
    """(uid by TransactionID, set of train-block uids) from features_v2; no labels are read."""
    from src import feature_sets as fs
    from src.data import ROOT
    from src.split import holdout_indices
    spec = fs.load_spec()
    ids = pd.read_parquet(ROOT / spec["parquet"], columns=["TransactionID", "TransactionDT", "uid"])
    train_idx, _, _ = holdout_indices(ids)
    return (pd.Series(ids["uid"].to_numpy(), index=ids["TransactionID"].to_numpy()),
            set(ids["uid"].iloc[train_idx]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Log the segment blend's validation rows (before the freeze). "
                                             "Its test rows are logged by `python -m src.final_test --blend`.")
    ap.add_argument("--log-validation", action="store_true")
    if ap.parse_args().log_validation:
        log_blend("val", *uid_lookup())
    else:
        ap.print_help()
