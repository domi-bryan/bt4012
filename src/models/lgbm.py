"""LightGBM with fold-local class weights and PR-AUC early stopping."""
import lightgbm as lgb
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

NAME = "lgbm"
EARLY_STOPPING = True   # src.experiment holds out rows for it in the G0-G3 runs
# Row and column subsampling (80% each, a fresh row sample every tree) match src/models/xgb.py
# and src/models/catboost.py, so the three boosting models differ only in their algorithms.
# Added 2026-11-14 for the G0-G3 runs; the earlier F0-F2 runs used no subsampling.
# Cap on trees raised from 1,500 to 3,000 on 2026-11-16, the same for all three boosting models.
DEFAULTS = {
    "n_estimators": 3000, "num_leaves": 63, "learning_rate": 0.05,
    "min_child_samples": 100, "scale_pos_weight": "balanced", "n_jobs": 6,
    "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
}
# L2 on leaf weights (reg_lambda) is tuned over the same values as XGBoost's reg_lambda and CatBoost's
# l2_leaf_reg (added 2026-11-14; F0-F2 runs used LightGBM's default of 0).
GRID = list(ParameterSampler({
    "num_leaves": [31, 63, 127],
    "learning_rate": [0.03, 0.1],
    "min_child_samples": [20, 100, 300],
    "reg_lambda": [1.0, 3.0, 10.0],
    "scale_pos_weight": [1.0, "balanced"],
}, n_iter=20, random_state=0))


def _pr_auc(y_true, y_score):
    return "pr_auc", average_precision_score(y_true, y_score), True


class Model:
    def __init__(self, clf):
        self.clf = clf
        self.run_metadata = {
            "best_iteration": int(clf.best_iteration_ or clf.n_estimators_),
            "resolved_scale_pos_weight": float(clf.get_params()["scale_pos_weight"]),
        }

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]


def fit(X, y, X_val=None, y_val=None, params=None, seed=0):
    """Stop after 50 rounds without an improvement in validation average precision."""
    p = {**DEFAULTS, **(params or {})}
    y = np.asarray(y)
    if p["scale_pos_weight"] == "balanced":
        positives = np.count_nonzero(y == 1)
        if positives == 0 or positives == len(y):
            raise ValueError("LightGBM training requires both classes")
        p["scale_pos_weight"] = float((len(y) - positives) / positives)
    clf = lgb.LGBMClassifier(
        **p, objective="binary", metric="None", verbosity=-1,
        random_state=seed, deterministic=True, force_col_wise=True,
    )
    if (X_val is None) != (y_val is None):
        raise ValueError("Supply both X_val and y_val for early stopping")
    kwargs = {}
    if X_val is not None:
        kwargs = {
            "eval_set": [(X_val, np.asarray(y_val))],
            "eval_metric": _pr_auc,
            "callbacks": [lgb.early_stopping(50, first_metric_only=True, verbose=False)],
        }
    clf.fit(X, y, **kwargs)
    return Model(clf)


def refit_params(model):
    """The stopping point found by early stopping, as settings for a refit without it."""
    return {"n_estimators": model.run_metadata["best_iteration"]}


def predict_proba(model, X):
    return model.predict_proba(X)


IMPORTANCE = "total gain"


def feature_importance(model, columns):
    """Total gain of the splits on each input column."""
    import pandas as pd
    return pd.Series(model.clf.booster_.feature_importance("gain"), index=list(columns), dtype="float64")
