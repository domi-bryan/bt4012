"""Random Forest on TreePipeline(kind="rf") output."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier

NAME = "rf"
DEFAULTS = {"n_estimators": 300, "max_depth": 24, "min_samples_leaf": 5, "n_jobs": 6}
GRID = [{"max_depth": depth, "min_samples_leaf": leaf}
        for depth in (12, 24) for leaf in (5, 20)]


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
