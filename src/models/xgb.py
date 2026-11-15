"""XGBoost on `TreePipeline(kind="lgbm")` output, following the shared model contract.

Categories stay pandas `category` columns and use XGBoost's native categorical splits
(`enable_categorical=True`, histogram method). Class weighting and early stopping match
`src.models.lgbm`: "balanced" resolves to negatives / positives of the training part, and
training stops after 50 rounds without a better validation average precision (the same
sklearn measure as `src.metrics`), keeping the best round.
"""
import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

NAME = "xgboost"
EARLY_STOPPING = True   # src.experiment holds out rows for it in the G0-G3 runs
# Cap on trees (raised from 1,500 on 2026-11-16: at learning rate 0.03 many fits hit 1,500 before early stopping).
DEFAULTS = {
    "n_estimators": 3000, "max_depth": 6, "learning_rate": 0.05, "min_child_weight": 1.0,
    # 80% of rows and columns per tree, the same as src/models/lgbm.py and catboost.py
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    "scale_pos_weight": "balanced", "n_jobs": 6,
}
# reg_lambda (L2 on leaf weights) is tuned over the same values as LightGBM's reg_lambda and
# CatBoost's l2_leaf_reg.
GRID = list(ParameterSampler({
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.03, 0.1],
    "min_child_weight": [1.0, 5.0, 20.0],
    "reg_lambda": [1.0, 3.0, 10.0],
    "scale_pos_weight": [1.0, "balanced"],
}, n_iter=20, random_state=0))
PATIENCE = 50


def _average_precision(y_true, y_score):
    return average_precision_score(y_true, y_score)


class Model:
    def __init__(self, clf, resolved_weight):
        self.clf = clf
        best = getattr(clf, "best_iteration", None)
        self.run_metadata = {
            "best_iteration": int(best) + 1 if best is not None else int(clf.n_estimators),
            "resolved_scale_pos_weight": float(resolved_weight),
        }

    def predict_proba(self, X):
        return self.clf.predict_proba(X)[:, 1]


def fit(X, y, X_val=None, y_val=None, params=None, seed=0):
    """Fit with fold-local class weights; early stopping needs both X_val and y_val."""
    p = {**DEFAULTS, **(params or {})}
    y = np.asarray(y)
    positives = np.count_nonzero(y == 1)
    if positives == 0 or positives == len(y):
        raise ValueError("XGBoost training requires both classes")
    if p["scale_pos_weight"] == "balanced":
        p["scale_pos_weight"] = float((len(y) - positives) / positives)
    if (X_val is None) != (y_val is None):
        raise ValueError("Supply both X_val and y_val for early stopping")
    callbacks = []
    if X_val is not None:
        callbacks = [xgb.callback.EarlyStopping(rounds=PATIENCE, metric_name="_average_precision",
                                                maximize=True, save_best=True)]
    clf = xgb.XGBClassifier(
        **p, objective="binary:logistic", tree_method="hist", enable_categorical=True,
        eval_metric=_average_precision, callbacks=callbacks, random_state=seed, verbosity=0,
    )
    kwargs = {"eval_set": [(X_val, np.asarray(y_val))], "verbose": False} if X_val is not None else {}
    clf.fit(X, y, **kwargs)
    return Model(clf, p["scale_pos_weight"])


def refit_params(model):
    """The stopping point found by early stopping, as settings for a refit without it."""
    return {"n_estimators": model.run_metadata["best_iteration"]}


def predict_proba(model, X):
    return model.predict_proba(X)


IMPORTANCE = "total gain"


def feature_importance(model, columns):
    """Total gain of the splits on each input column (columns never split on get 0)."""
    import pandas as pd
    gain = model.clf.get_booster().get_score(importance_type="total_gain")
    return pd.Series(gain, dtype="float64").reindex(list(columns)).fillna(0.0)
