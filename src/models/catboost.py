"""CPU CatBoost following the shared model contract, with no target statistics."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

NAME = "catboost"
EARLY_STOPPING = True   # src.experiment holds out rows for it in the G0-G3 runs
# Row and column subsampling (80% each: Bernoulli row sampling, `rsm` for columns) match
# src/models/lgbm.py and src/models/xgb.py. Added 2026-11-14 for the G0-G3 runs; the earlier
# F0-F2 runs used CatBoost's default (MVS) row sampling and all columns.
# Cap on trees raised from 1,500 to 3,000 on 2026-11-16, the same for all three boosting models.
DEFAULTS = {
    "iterations": 3000, "depth": 6, "learning_rate": 0.05,
    "l2_leaf_reg": 3.0, "scale_pos_weight": "balanced",
    "thread_count": 6, "early_stopping_rounds": 50,
    "bootstrap_type": "Bernoulli", "subsample": 0.8, "rsm": 0.8,
}
GRID = list(ParameterSampler({
    "depth": [4, 6, 8], "learning_rate": [0.03, 0.1],
    "l2_leaf_reg": [1.0, 3.0, 10.0], "scale_pos_weight": [1.0, "balanced"],
}, n_iter=20, random_state=0))
SENSITIVITY = {
    "depth": [3, 4, 6, 8, 10],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.3],
    "l2_leaf_reg": [0.1, 1.0, 3.0, 10.0, 30.0],
    "scale_pos_weight": [1.0, 3.0, 10.0, 30.0, 100.0, "balanced"],
}


class AveragePrecision:
    """Match src.metrics exactly, ignoring training class weights for evaluation.

    Raw logits preserve score order, so sigmoid is unnecessary for average precision.
    CatBoost's built-in PRAUC integrates differently from sklearn average precision.
    """

    def is_max_optimal(self):
        return True

    def evaluate(self, approxes, target, weight):
        return float(average_precision_score(target, approxes[0])), 1.0

    def get_final_error(self, error, weight):
        return error


class Model:
    def __init__(self, clf, resolved_weight, one_hot_max_size):
        self.clf = clf
        self.run_metadata = {
            "best_iteration": int(clf.tree_count_),
            "resolved_scale_pos_weight": float(resolved_weight),
            "categorical_columns": list(clf.feature_names_[i] for i in clf.get_cat_feature_indices()),
            "categorical_mode": "one_hot_no_target_statistics",
            "one_hot_max_size": one_hot_max_size,
            "early_stopping_metric": "sklearn average_precision_score (unweighted)",
        }

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]


def fit(X, y, X_val=None, y_val=None, params=None, seed=0):
    """Fit with fold-local weighting and optional validation AP early stopping.

    All categorical features use internal one-hot splits, so no target statistics
    (including category combinations) can encode card1 or uid. Model configuration
    cannot override this policy. '_sweep' is logging provenance, not a CatBoost option.
    """
    supplied = dict(params or {})
    supplied.pop("_sweep", None)
    unknown = set(supplied) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unsupported CatBoost parameters: {sorted(unknown)}")
    p = {**DEFAULTS, **supplied}
    if (X_val is None) != (y_val is None):
        raise ValueError("Supply both X_val and y_val for early stopping")
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or X.empty:
        raise ValueError("Expected a nonempty DataFrame with unique feature names")
    if "uid" in X or "isFraud" in X:
        raise ValueError("Raw uid and the target must not be model inputs")
    if "card1" in X and not pd.api.types.is_numeric_dtype(X.card1):
        raise ValueError("card1 must remain numeric")
    cats = [c for c in X if not pd.api.types.is_numeric_dtype(X[c])]
    for c in cats:
        if X[c].isna().any() or not X[c].map(lambda v: isinstance(v, str)).all():
            raise ValueError(f"{c}: use CatBoostPipeline for nonmissing string categories")
    y = np.asarray(y)
    if len(y) != len(X) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("CatBoost training requires aligned binary labels with both classes")
    if X_val is not None and (list(X_val.columns) != list(X.columns) or len(X_val) != len(y_val)):
        raise ValueError("Validation features and labels must align with training columns")
    if p["scale_pos_weight"] == "balanced":
        p["scale_pos_weight"] = float(np.count_nonzero(y == 0) / np.count_nonzero(y == 1))
    if not np.isfinite(p["scale_pos_weight"]) or p["scale_pos_weight"] <= 0:
        raise ValueError("scale_pos_weight must be positive or 'balanced'")
    one_hot_max_size = max([2] + [X[c].nunique() for c in cats])
    if one_hot_max_size > 65535:
        raise ValueError("CPU one-hot category limit exceeded; group encoding decision required")
    patience = p.pop("early_stopping_rounds")
    clf = CatBoostClassifier(
        **p, loss_function="Logloss", eval_metric=AveragePrecision(),
        cat_features=cats, one_hot_max_size=one_hot_max_size,
        has_time=True, task_type="CPU", random_seed=seed,
        allow_writing_files=False, verbose=False,
    )
    kwargs = {}
    if X_val is not None:
        kwargs = {"eval_set": (X_val, np.asarray(y_val)), "use_best_model": True,
                  "early_stopping_rounds": patience}
    clf.fit(X, y, **kwargs)
    return Model(clf, p["scale_pos_weight"], one_hot_max_size)


def refit_params(model):
    """The stopping point found by early stopping, as settings for a refit without it."""
    return {"iterations": model.run_metadata["best_iteration"]}


def predict_proba(model, X):
    return model.predict_proba(X)


IMPORTANCE = "prediction values change"


def feature_importance(model, columns):
    """CatBoost's default importance: how much predictions change on average when a feature's split values change."""
    return pd.Series(model.clf.get_feature_importance(), index=model.clf.feature_names_,
                     dtype="float64").reindex(list(columns)).fillna(0.0)
