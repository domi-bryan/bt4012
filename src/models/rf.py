"""Random Forest on TreePipeline(kind="rf") output."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

NAME = "rf"
DEFAULTS = {"n_estimators": 300, "max_depth": 24, "min_samples_leaf": 5, "max_features": "sqrt", "n_jobs": 6}
# 12 settings: depth x minimum leaf size x share of columns tried per split (widened 2026-11-14;
# F0-F2 runs used depth {12, 24} x leaf {5, 20} with sqrt columns). None = unlimited depth.
GRID = [{"max_depth": depth, "min_samples_leaf": leaf, "max_features": mf}
        for depth in (12, 24, None) for leaf in (5, 20) for mf in ("sqrt", 0.3)]


class Model:
    def __init__(self, clf):
        self.clf = clf

    def predict_proba(self, X):
        return self.clf.predict_proba(np.asarray(X, dtype="float32"))[:, 1]


def fit(X, y, X_val=None, y_val=None, params=None, seed=0):
    """Bootstrap class weights are computed from training labels only."""
    clf = RandomForestClassifier(
        **{**DEFAULTS, **(params or {})},
        class_weight="balanced_subsample", random_state=seed,
    )
    clf.fit(np.asarray(X, dtype="float32"), np.asarray(y))
    return Model(clf)


def predict_proba(model, X):
    return model.predict_proba(X)
