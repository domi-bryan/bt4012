"""Role C: shared time-CV tuning and validation runs. Every fit appends one row to the log.

A model module exposes `NAME`, `fit(X, y, X_val, y_val, params, seed)` returning an object
with `predict_proba(X)`. `make_pipeline()` returns an unfitted pipeline from `src.data`;
it is refit on each fold's training part, so no fold sees its own validation rows.
"""
import random
import time
import json

import numpy as np
import pandas as pd

from src.metrics import evaluate, pick_threshold
from src.results import LOG_PATH, log_run
from src.split import time_cv_folds


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _xy(pipe, df, idx):
    part = df.iloc[idx]
    return pipe.transform(part), part["isFraud"].to_numpy()


def _fit_score(model, X, y, X_val, y_val, params, seed):
    set_seeds(seed)
    t0 = time.time()
    fitted = model.fit(X, y, X_val, y_val, params, seed=seed)
    score = fitted.predict_proba(X_val)
    return fitted, score, time.time() - t0


def logged_run(model_name, feature_set, split, seed, params, log_path=LOG_PATH):
    """Find an exact run, ignoring only recorded outcomes (threshold/model metadata).

    Callers must keep data, feature configuration and model defaults fixed when resuming.
    """
    try:
        log = pd.read_csv(log_path)
    except FileNotFoundError:
        return None
    scheme = "time" if split == "cv" else "holdout"
    candidates = log[(log.model == model_name) & (log.feature_set == feature_set)
                     & (log.split == split) & (log.seed == seed) & (log.cv_scheme == scheme)]
    matches = []
    for row in candidates.to_dict("records"):
        recorded = json.loads(row["params"])
        inputs = {k: v for k, v in recorded.items() if k not in ("threshold", "_fit")}
        if inputs == params:
            matches.append(row)
    if len(matches) > 1:
        raise ValueError(f"Ambiguous duplicate runs: {model_name}/{feature_set}/{split}/{params}")
    return matches[0] if matches else None


def time_cv(model, make_pipeline, df, train_idx, grid, feature_set, seed=0, log_path=LOG_PATH,
            resume=False):
    """Score every params dict in `grid` on every time fold; log one cv row per (params, fold).

    Returns a DataFrame with one row per (params, fold). runtime_s covers model fit and predict.
    """
    rows = []
    for k, (a, b) in enumerate(time_cv_folds(df, train_idx), start=1):
        prepared = False
        for params in grid:
            old = (logged_run(model.NAME, feature_set, "cv", seed, {**params, "fold": k},
                              log_path) if resume else None)
            if old is not None:
                rows.append({**old, "fold": k, "params": params})
                continue
            if not prepared:
                pipe = make_pipeline().fit(df.iloc[a])
                Xa, ya = _xy(pipe, df, a)
                Xb, yb = _xy(pipe, df, b)
                prepared = True
            fitted, score, runtime = _fit_score(model, Xa, ya, Xb, yb, params, seed)
            metrics = evaluate(yb, score)
            logged_params = {**params, "fold": k}
            if hasattr(fitted, "run_metadata"):
                logged_params["_fit"] = fitted.run_metadata
            log_run(log_path, model=model.NAME, feature_set=feature_set, cv_scheme="time",
                    split="cv", seed=seed, params=logged_params,
                    runtime_s=round(runtime, 2), **metrics)
            rows.append({"fold": k, "params": params, "runtime_s": runtime, **metrics})
            print(f"fold {k} {params} pr_auc={metrics['pr_auc']:.4f} ({runtime:.0f}s)", flush=True)
            del fitted
        if prepared:
            del Xa, Xb, pipe
    return pd.DataFrame(rows)


def best_params(cv_rows):
    """Params with the highest mean CV PR-AUC across folds."""
    keyed = cv_rows.assign(key=cv_rows["params"].map(lambda p: json.dumps(p, sort_keys=True)))
    best = keyed.groupby("key")["pr_auc"].mean().idxmax()
    return json.loads(best)


def fit_validate(model, make_pipeline, df, train_idx, val_idx, params, feature_set, seed=0,
                 log_path=LOG_PATH):
    """Refit on the whole train block, score validation, pick the F1 threshold there, log one val row.

    The threshold is stored in params so final test scoring can reuse it.
    """
    pipe = make_pipeline().fit(df.iloc[train_idx])
    X, y = _xy(pipe, df, train_idx)
    X_val, y_val = _xy(pipe, df, val_idx)
    fitted, score, runtime = _fit_score(model, X, y, X_val, y_val, params, seed)
    threshold = pick_threshold(y_val, score)
    metrics = evaluate(y_val, score, threshold)
    logged_params = {**params, "threshold": threshold}
    if hasattr(fitted, "run_metadata"):
        logged_params["_fit"] = fitted.run_metadata
    log_run(log_path, model=model.NAME, feature_set=feature_set, cv_scheme="holdout",
            split="val", seed=seed, params=logged_params,
            runtime_s=round(runtime, 2), **metrics)
    return {"model": fitted, "pipeline": pipe, "score": score, "threshold": threshold,
            "runtime_s": runtime, **metrics}


def already_logged(model_name, feature_set, log_path=LOG_PATH):
    """Number of log rows for this model and feature set (to avoid duplicate runs)."""
    try:
        log = pd.read_csv(log_path)
    except FileNotFoundError:
        return 0
    return int(((log["model"] == model_name) & (log["feature_set"] == feature_set)).sum())
