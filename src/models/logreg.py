"""Role C: logistic regression baseline (Step 6). Input is `LinearPipeline` output."""
import numpy as np
from sklearn.linear_model import LogisticRegression

NAME = "logreg"
DEFAULTS = {"C": 1.0, "max_iter": 1000}


class Model:
    def __init__(self, clf):
        self.clf = clf

    def predict_proba(self, X):
        """Fraud probability per row (1-D)."""
        return self.clf.predict_proba(np.asarray(X))[:, 1]


def fit(X, y, X_val=None, y_val=None, params=None, seed=0):
    """Fit balanced-class-weight LR. X_val/y_val are unused (no early stopping); kept for the shared interface."""
    p = {**DEFAULTS, **(params or {})}
    clf = LogisticRegression(C=p["C"], max_iter=p["max_iter"], class_weight="balanced",
                             solver="lbfgs", random_state=seed)
    clf.fit(np.asarray(X), np.asarray(y))
    return Model(clf)


def predict_proba(model, X):
    return model.predict_proba(X)


# 11 values of C from 0.001 to 100 in half-decade steps (widened 2026-11-14; F0-F2 runs used 0.01-10)
GRID = [{"C": c} for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100)]


def main(feature_set="F0", seed=0, force=False):
    """Step 6: tune C with time CV, refit the best C on the train block, score validation."""
    import sys

    from src import data, experiment
    from src.features import build_feature_set
    from src.split import holdout_indices

    if experiment.already_logged(NAME, feature_set) and not force:
        sys.exit(f"{NAME}/{feature_set} already has log rows; pass --force to append another run")
    df = build_feature_set(data.load_cleaned(), feature_set)
    train_idx, val_idx, _ = holdout_indices(df)
    _, v_drop = data.load_v_columns()
    make_pipeline = lambda: data.LinearPipeline(v_drop=v_drop)
    cv = experiment.time_cv(sys.modules[__name__], make_pipeline, df, train_idx, GRID,
                            feature_set, seed=seed)
    best = experiment.best_params(cv)
    print("mean CV PR-AUC by C:\n", cv.groupby(cv["params"].map(lambda p: p["C"]))["pr_auc"].mean())
    val = experiment.fit_validate(sys.modules[__name__], make_pipeline, df, train_idx, val_idx,
                                  best, feature_set, seed=seed)
    print(f"best {best}; validation pr_auc={val['pr_auc']:.4f} auroc={val['auroc']:.4f} "
          f"threshold={val['threshold']:.4f} ({val['runtime_s']:.0f}s)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="F0", choices=["F0", "F1", "F2"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.feature_set, a.seed, a.force)


IMPORTANCE = "absolute coefficient (standardised inputs)"


def feature_importance(model, columns):
    """|coefficient| per input column; inputs are standardised, so sizes are comparable within the model."""
    import pandas as pd
    return pd.Series(np.abs(model.clf.coef_[0]), index=list(columns))
