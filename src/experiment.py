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


def _fit_score(model, X, y, X_val, y_val, params, seed, X_score=None):
    """Fit (X_val/y_val drive early stopping, if the model has any) and score X_score (default X_val)."""
    set_seeds(seed)
    t0 = time.time()
    fitted = model.fit(X, y, X_val, y_val, params, seed=seed)
    score = fitted.predict_proba(X_val if X_score is None else X_score)
    return fitted, score, time.time() - t0


def _uses_early_stopping(model):
    return bool(getattr(model, "EARLY_STOPPING", False))


def split_early_stopping(idx, fraction):
    """(fit rows, early-stopping rows): the last `fraction` of `idx` in time order goes to early stopping."""
    idx = np.asarray(idx)
    n = max(1, int(round(len(idx) * fraction)))
    return idx[:-n], idx[-n:]


def _prepare(model, make_pipeline, df, train_part, score_idx, es_fraction):
    """Fit the pipeline and build the matrices for one fit.

    With `es_fraction` and a model that early-stops, the last `es_fraction` of the training part
    (by time) is held out for early stopping, so the scored rows never influence training.
    Without it, early stopping (if any) watches the scored rows, as in the F0-F2 runs.
    Returns (pipe, X, y, X_es, y_es, X_score, y_score, early_stopping_rows).
    """
    if es_fraction and _uses_early_stopping(model):
        fit_idx, es_idx = split_early_stopping(train_part, es_fraction)
    else:
        fit_idx, es_idx = train_part, None
    pipe = make_pipeline().fit(df.iloc[fit_idx])
    X, y = _xy(pipe, df, fit_idx)
    X_s, y_s = _xy(pipe, df, score_idx)
    if es_idx is None:
        return pipe, X, y, X_s, y_s, X_s, y_s, 0
    X_es, y_es = _xy(pipe, df, es_idx)
    return pipe, X, y, X_es, y_es, X_s, y_s, len(es_idx)


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
            resume=False, es_fraction=None):
    """Score every params dict in `grid` on every time fold; log one cv row per (params, fold).

    Returns a DataFrame with one row per (params, fold). runtime_s covers model fit and predict.
    `es_fraction`: see `_prepare` (None keeps the F0-F2 behaviour).
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
                pipe, Xa, ya, Xe, ye, Xb, yb, n_es = _prepare(model, make_pipeline, df, a, b, es_fraction)
                prepared = True
            fitted, score, runtime = _fit_score(model, Xa, ya, Xe, ye, params, seed, X_score=Xb)
            metrics = evaluate(yb, score)
            logged_params = {**params, "fold": k}
            if hasattr(fitted, "run_metadata"):
                logged_params["_fit"] = dict(fitted.run_metadata)
            if n_es:
                logged_params.setdefault("_fit", {})["early_stopping_rows"] = n_es
            log_run(log_path, model=model.NAME, feature_set=feature_set, cv_scheme="time",
                    split="cv", seed=seed, params=logged_params,
                    runtime_s=round(runtime, 2), **metrics)
            rows.append({"fold": k, "params": params, "runtime_s": runtime, **metrics})
            print(f"fold {k} {params} pr_auc={metrics['pr_auc']:.4f} ({runtime:.0f}s)", flush=True)
            del fitted
        if prepared:
            del Xa, Xb, Xe, pipe
    return pd.DataFrame(rows)


def best_params(cv_rows):
    """Params with the highest mean CV PR-AUC across folds."""
    keyed = cv_rows.assign(key=cv_rows["params"].map(lambda p: json.dumps(p, sort_keys=True)))
    best = keyed.groupby("key")["pr_auc"].mean().idxmax()
    return json.loads(best)


def fit_validate(model, make_pipeline, df, train_idx, val_idx, params, feature_set, seed=0,
                 log_path=LOG_PATH, es_fraction=None):
    """Refit on the whole train block, score validation, pick the F1 threshold there, log one val row.

    The threshold is stored in params so final test scoring can reuse it. With `es_fraction`, an
    early-stopping model trains on the train block minus its last `es_fraction` (by time), which
    is used for early stopping instead of the validation block.
    """
    pipe, X, y, X_es, y_es, X_val, y_val, n_es = _prepare(model, make_pipeline, df, train_idx, val_idx,
                                                          es_fraction)
    fitted, score, runtime = _fit_score(model, X, y, X_es, y_es, params, seed, X_score=X_val)
    threshold = pick_threshold(y_val, score)
    metrics = evaluate(y_val, score, threshold)
    logged_params = {**params, "threshold": threshold}
    if hasattr(fitted, "run_metadata"):
        logged_params["_fit"] = dict(fitted.run_metadata)
    if n_es:
        logged_params.setdefault("_fit", {})["early_stopping_rows"] = n_es
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


# ---------------------------------------------------------------------------
# G0-G3 runs used by the model notebooks (notebooks/04_*.ipynb to 09_*.ipynb).

from src.data import ROOT  # noqa: E402

PREDS_DIR = ROOT / "results" / "preds"
VAL_SEEDS = (0, 1, 2)
# Share of each training part (the last rows by time) held out for early stopping in the G0-G3 runs,
# so LightGBM, XGBoost, CatBoost and the MLP never early-stop on the rows they are scored on.
ES_FRACTION = 0.1


def save_predictions(path, df, idx, score, **meta):
    """One parquet per run: TransactionID, label and score for the scored rows, plus `meta` columns."""
    part = df.iloc[idx]
    out = pd.DataFrame({"TransactionID": part["TransactionID"].to_numpy(),
                        "isFraud": part["isFraud"].to_numpy(), "score": np.asarray(score, dtype="float64")})
    for key, value in meta.items():
        out[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def oof_predictions(model, make_pipeline, df, train_idx, params, feature_set, seed=0,
                    preds_dir=PREDS_DIR, es_fraction=ES_FRACTION):
    """Refit `params` on each time-CV fold and save the fold-validation scores (not logged).

    Covers the rows of folds 1-4 (days 31 to 141). Used to train a stacking model later.
    """
    path = preds_dir / f"{model.NAME}_{feature_set}_oof_seed{seed}.parquet"
    if path.exists():
        return path
    parts = []
    for k, (a, b) in enumerate(time_cv_folds(df, train_idx), start=1):
        _, Xa, ya, Xe, ye, Xb, yb, _ = _prepare(model, make_pipeline, df, a, b, es_fraction)
        _, score, _ = _fit_score(model, Xa, ya, Xe, ye, params, seed, X_score=Xb)
        parts.append(pd.DataFrame({"TransactionID": df["TransactionID"].iloc[b].to_numpy(),
                                   "isFraud": yb, "score": score, "fold": k}))
    out = pd.concat(parts, ignore_index=True)
    out["params"] = json.dumps(params, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def tune_and_validate(model, make_pipeline, df, train_idx, val_idx, feature_sets, grid,
                      tune_on=None, seeds=VAL_SEEDS, log_path=LOG_PATH, preds_dir=PREDS_DIR,
                      save_oof=False, es_fraction=ES_FRACTION):
    """Tune with time CV (seed 0), then validate the chosen settings with each seed.

    `feature_sets` maps a set name (e.g. "G3") to its column list; `make_pipeline(columns)`
    returns an unfitted pipeline. With `tune_on=None` every set is tuned separately; with a
    set name (e.g. "G3"), tuning runs on that set only and its best settings are validated on
    every set, so the sets are compared with the model settings held fixed.

    Models that early-stop (LightGBM, XGBoost, CatBoost, MLP) hold out the last `es_fraction` of
    each training part for it, so no model ever sees the rows it is scored on.

    Everything already in the log is skipped, so a notebook can be stopped and rerun.
    Validation scores are saved to `preds_dir`; with `save_oof` the time-CV fold scores of
    the chosen settings are saved too. Returns {set name: chosen params}.
    """
    if tune_on is not None and tune_on not in feature_sets:
        raise ValueError(f"tune_on={tune_on!r} is not one of {list(feature_sets)}")
    tuned = {}
    for name in ([tune_on] if tune_on else list(feature_sets)):
        cols = feature_sets[name]
        print(f"\n{model.NAME}/{name}: tuning {len(grid)} settings with time CV (seed 0)", flush=True)
        cv = time_cv(model, lambda cols=cols: make_pipeline(cols), df, train_idx, grid, name,
                     seed=0, log_path=log_path, resume=True, es_fraction=es_fraction)
        tuned[name] = best_params(cv)
        print(f"{model.NAME}/{name}: best {tuned[name]}", flush=True)
    chosen = {name: tuned[tune_on] if tune_on else tuned[name] for name in feature_sets}
    for name, params in chosen.items():
        cols = feature_sets[name]
        build = lambda cols=cols: make_pipeline(cols)
        for seed in seeds:
            path = preds_dir / f"{model.NAME}_{name}_val_seed{seed}.parquet"
            if logged_run(model.NAME, name, "val", seed, params, log_path) is not None:
                note = "" if path.exists() else " (scores not saved; delete its log row to rerun)"
                print(f"{model.NAME}/{name} seed {seed}: validation already logged{note}", flush=True)
                continue
            val = fit_validate(model, build, df, train_idx, val_idx, params, name, seed=seed,
                               log_path=log_path, es_fraction=es_fraction)
            save_predictions(path, df, val_idx, val["score"], seed=seed,
                             params=json.dumps(params, sort_keys=True))
            print(f"{model.NAME}/{name} seed {seed}: val pr_auc={val['pr_auc']:.4f} "
                  f"auroc={val['auroc']:.4f} ({val['runtime_s']:.0f}s)", flush=True)
            del val
        if save_oof:
            oof_predictions(model, build, df, train_idx, params, name, seed=0, preds_dir=preds_dir,
                            es_fraction=es_fraction)
    return chosen


def block_sizes(df, train_idx, val_idx):
    """{"val": (rows, fraud), "cv": {fold: (rows, fraud)}} for the scored rows of each block."""
    y = df["isFraud"].to_numpy()
    folds = {k: (len(b), int(y[b].sum())) for k, (_, b) in enumerate(time_cv_folds(df, train_idx), start=1)}
    return {"val": (len(val_idx), int(y[val_idx].sum())), "cv": folds}


def _precision_top1pct(recall, rows, fraud):
    from src.metrics import top_rows
    return recall * fraud / top_rows(rows)


def run_summary(model_name, feature_sets, log_path=LOG_PATH, sizes=None):
    """(cv, val) tables for one model from the log, for the G sets given.

    cv: per set and settings, mean over folds of PR-AUC (with sd), AUROC and recall in the top 1%.
    val: per set and settings, mean and sample sd over seeds of PR-AUC, AUROC, recall in the top 1%,
    precision and recall at the validation threshold, and the mean threshold.
    With `sizes` from `block_sizes`, both also get precision in the top 1%, derived from the logged
    recall (precision = recall * fraud rows / rows flagged), so past runs need no refit.
    """
    log = pd.read_csv(log_path)
    log = log[(log["model"] == model_name) & (log["feature_set"].isin(list(feature_sets)))].copy()
    params = log["params"].map(json.loads)
    log["settings"] = params.map(lambda d: json.dumps(
        {k: v for k, v in d.items() if k not in ("fold", "threshold", "_fit")}, sort_keys=True))
    log["fold"] = params.map(lambda d: d.get("fold"))
    log["threshold"] = params.map(lambda d: d.get("threshold"))
    if sizes is not None:
        def prec(r):
            rows, fraud = sizes["val"] if r["split"] == "val" else sizes["cv"][int(r["fold"])]
            return _precision_top1pct(r["recall_top1pct"], rows, fraud)
        log["precision_top1pct"] = log.apply(prec, axis=1) if len(log) else []
    top = ["recall_top1pct"] + (["precision_top1pct"] if sizes is not None else [])

    cv = log[log["split"] == "cv"]
    cv = (cv.groupby(["feature_set", "settings"])
          .agg(cv_pr_auc=("pr_auc", "mean"), cv_std=("pr_auc", "std"), cv_auroc=("auroc", "mean"),
               **{f"cv_{m}": (m, "mean") for m in top}, folds=("fold", "nunique"))
          .reset_index().sort_values(["feature_set", "cv_pr_auc"], ascending=[True, False]))
    val = log[log["split"] == "val"]
    metrics = ["pr_auc", "auroc"] + top + ["precision", "recall"]
    val = val.groupby(["feature_set", "settings"]).agg(
        **{f"{m}_{s}": (m, s) for m in metrics for s in ("mean", "std")},
        threshold_mean=("threshold", "mean"), seeds=("seed", "nunique"))
    return cv.reset_index(drop=True), val.reset_index()


MODELS_DIR = ROOT / "results" / "models"


def fit_final(model, make_pipeline, df, train_idx, params, feature_set, seed=0, es_fraction=ES_FRACTION,
              cache_dir=MODELS_DIR):
    """Refit `params` on the training block exactly as `fit_validate` does (same rows, same early-stopping
    hold-out, same seed), without logging. Returns (fitted model, fitted pipeline).

    The pair is cached as results/models/<model>_<set>_seed<seed>.pkl (gitignored) with the settings, so
    reruns, feature importance and the final test scoring reuse it instead of refitting.
    """
    import pickle
    path = None if cache_dir is None else cache_dir / f"{model.NAME}_{feature_set}_seed{seed}.pkl"
    key = json.dumps(params, sort_keys=True)
    if path is not None and path.exists():
        with open(path, "rb") as handle:
            saved = pickle.load(handle)
        if saved.get("params") == key and saved.get("es_fraction") == es_fraction:
            return saved["model"], saved["pipeline"]
    fit_idx, es_idx = (split_early_stopping(train_idx, es_fraction)
                       if es_fraction and _uses_early_stopping(model) else (np.asarray(train_idx), None))
    pipe = make_pipeline().fit(df.iloc[fit_idx])
    X, y = _xy(pipe, df, fit_idx)
    X_es, y_es = _xy(pipe, df, es_idx) if es_idx is not None else (None, None)
    set_seeds(seed)
    fitted = model.fit(X, y, X_es, y_es, params, seed=seed)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(path, "wb") as handle:
                pickle.dump({"model": fitted, "pipeline": pipe, "params": key, "es_fraction": es_fraction}, handle)
        except Exception as exc:   # a model that can't be pickled is simply not cached
            path.unlink(missing_ok=True)
            print(f"not cached ({type(exc).__name__}: {exc})")
    return fitted, pipe


def setting_sensitivity(model_name, feature_set, log_path=LOG_PATH, folds=None):
    """RQ4 from the tuning runs already in the log: for each tuned setting and each value it took,
    the mean, best and worst time-CV PR-AUC over the settings tried with that value.

    Each tried combination is first averaged over its folds (only combinations with all `folds`
    folds count; default: the most folds any combination has). With a full grid every value is
    paired with the same mix of other settings; with random search the mix differs by value, so the
    marginal means are indicative rather than a controlled one-at-a-time sweep.
    Returns one row per (setting, value): setting, value, mean, best, worst, combinations.
    """
    log = pd.read_csv(log_path)
    cv = log[(log["model"] == model_name) & (log["feature_set"] == feature_set) & (log["split"] == "cv")]
    if cv.empty:
        return pd.DataFrame(columns=["setting", "value", "mean", "best", "worst", "combinations"])
    params = cv["params"].map(json.loads)
    cv = cv.assign(settings=params.map(lambda d: json.dumps(
        {k: v for k, v in d.items() if k not in ("fold", "threshold", "_fit")}, sort_keys=True)))
    per = cv.groupby("settings")["pr_auc"].agg(["mean", "count"])
    per = per[per["count"] == (folds or per["count"].max())]
    # dtype=object keeps None (unlimited depth) and integers as they were logged
    table = pd.DataFrame([json.loads(s) for s in per.index], index=per.index, dtype=object)
    table["score"] = per["mean"].astype("float64")
    rows = []
    for setting in [c for c in table.columns if c != "score"]:
        values = table[setting].map(lambda v: "none" if v is None else v)
        if values.nunique() < 2:
            continue
        for value, part in table.groupby(values.astype(str), sort=False)["score"]:
            value = values[part.index[0]]
            rows.append({"setting": setting, "value": value, "mean": part.mean(), "best": part.max(),
                         "worst": part.min(), "combinations": len(part)})
    out = pd.DataFrame(rows, columns=["setting", "value", "mean", "best", "worst", "combinations"])
    out["value"] = pd.Series([r["value"] for r in rows], dtype=object)   # keep 15 as 15, not 15.0

    def order(v):   # numbers in numeric order, then text values (e.g. "balanced", "sqrt", "none")
        return (0, float(v), "") if isinstance(v, (int, float)) and not isinstance(v, bool) else (1, 0.0, str(v))
    return (out.assign(_k=out["value"].map(order)).sort_values(["setting", "_k"])
            .drop(columns="_k").reset_index(drop=True))
