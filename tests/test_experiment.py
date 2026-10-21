import json

import numpy as np
import pandas as pd
import pytest

from src import data, experiment
from src.models import logreg
from src.split import holdout_indices, time_cv_folds


@pytest.fixture
def toy():
    """3,000 rows over days 1-100 where fraud depends on amount and a category."""
    rng = np.random.default_rng(0)
    n = 3000
    dt = np.sort(rng.integers(86400, 101 * 86400, size=n)).astype("int32")
    amt = rng.lognormal(3, 1, n).astype("float32")
    cat = rng.choice(["W", "C", "H"], n)
    logit = -4 + 0.8 * (np.log(amt) - 3) + 1.5 * (cat == "C")
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype("int8")
    v1 = rng.normal(size=n).astype("float32")
    v1[rng.random(n) < 0.4] = np.nan
    return pd.DataFrame({
        "TransactionID": np.arange(n, dtype="int32"), "isFraud": y, "TransactionDT": dt,
        "TransactionDay": (dt // 86400).astype("int16"), "TransactionAmt": amt, "V1": v1,
        "ProductCD": pd.Categorical(cat),
    })


def test_logreg_cv_and_validation_logging(toy, tmp_path):
    log = tmp_path / "log.csv"
    tr, va, _ = holdout_indices(toy)
    make = lambda: data.LinearPipeline(v_drop=[])
    grid = [{"C": 0.1}, {"C": 1}]
    cv = experiment.time_cv(logreg, make, toy, tr, grid, "F0", log_path=log)
    n_folds = len(time_cv_folds(toy, tr))
    assert len(cv) == len(grid) * n_folds
    best = experiment.best_params(cv)
    assert best in grid
    val = experiment.fit_validate(logreg, make, toy, tr, va, best, "F0", log_path=log)
    assert val["pr_auc"] > toy["isFraud"].iloc[va].mean()

    df = pd.read_csv(log)
    assert len(df) == len(grid) * n_folds + 1
    assert (df["split"] != "test").all()
    assert df["split"].tolist().count("val") == 1
    folds = {json.loads(p)["fold"] for p in df.loc[df.split == "cv", "params"]}
    assert folds == set(range(1, n_folds + 1))
    val_params = json.loads(df.loc[df.split == "val", "params"].iloc[0])
    assert val_params["threshold"] == pytest.approx(val["threshold"])
    assert df.loc[df.split == "val", "precision"].notna().all()
    assert experiment.already_logged("logreg", "F0", log) == len(df)


def test_logreg_interface(toy):
    pipe = data.LinearPipeline(v_drop=[]).fit(toy)
    X = pipe.transform(toy)
    model = logreg.fit(X, toy["isFraud"], None, None, {"C": 1})
    p = logreg.predict_proba(model, X)
    assert p.shape == (len(toy),) and ((p >= 0) & (p <= 1)).all()
    assert model.clf.class_weight == "balanced"


def test_logreg_command_builds_requested_features(toy, monkeypatch):
    """An F2 label must never silently train on F0 inputs."""
    from src import features

    requested = []
    engineered = toy.assign(card1_prior_count=1, uid_prior_count=2)
    monkeypatch.setattr(data, "load_cleaned", lambda: toy)
    monkeypatch.setattr(data, "load_v_columns", lambda: ([], []))
    monkeypatch.setattr(experiment, "already_logged", lambda *args: 0)

    def build(frame, feature_set):
        assert frame is toy
        requested.append(feature_set)
        return engineered

    def cv(model, make, frame, *args, **kwargs):
        assert frame is engineered
        return pd.DataFrame({"params": [{"C": 0.1}], "pr_auc": [0.5]})

    def validate(model, make, frame, *args, **kwargs):
        assert frame is engineered
        return {"pr_auc": 0.5, "auroc": 0.8, "threshold": 0.5, "runtime_s": 1}

    monkeypatch.setattr(features, "build_feature_set", build)
    monkeypatch.setattr(experiment, "time_cv", cv)
    monkeypatch.setattr(experiment, "fit_validate", validate)
    logreg.main(feature_set="F2")
    assert requested == ["F2"]
