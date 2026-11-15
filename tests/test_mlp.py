"""MLP interface, training-only class weight, early stopping, determinism and logging."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src import data
from src.models import mlp
from src.split import holdout_indices, time_cv_folds
from src.train_trees import run_cell


@pytest.fixture
def transactions():
    rng = np.random.default_rng(3)
    n = 1500
    dt = np.linspace(86400, 151 * 86400, n).astype("int32")
    x = rng.normal(size=n).astype("float32")
    y = (rng.random(n) < 1 / (1 + np.exp(2.5 - 2 * x))).astype("int8")
    x[::11] = np.nan
    return pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionDT": dt, "TransactionDay": dt // 86400,
        "TransactionAmt": x, "ProductCD": pd.Categorical(np.where(rng.random(n) < 0.5, "C", "W")),
        "isFraud": y,
    })


@pytest.fixture
def matrices(transactions):
    tr, va, _ = holdout_indices(transactions)
    pipe = data.LinearPipeline(v_drop=[]).fit(transactions.iloc[tr])
    return (pipe.transform(transactions.iloc[tr]), transactions.isFraud.iloc[tr],
            pipe.transform(transactions.iloc[va]), transactions.isFraud.iloc[va])


def test_interface_and_training_weight(matrices):
    X, y, V, v = matrices
    model = mlp.fit(X, y, V, v, {"hidden": 16, "max_epochs": 8}, seed=0)
    scores = mlp.predict_proba(model, V)
    assert scores.shape == (len(V),) and np.isfinite(scores).all()
    assert ((scores >= 0) & (scores <= 1)).all()
    meta = model.run_metadata
    assert meta["pos_weight"] == pytest.approx((y == 0).sum() / y.sum(), rel=1e-5)
    assert len(meta["loss"]) == len(meta["val_pr_auc"]) == meta["epochs_run"]
    # the kept weights are the best epoch's: rescoring reproduces its PR-AUC
    from sklearn.metrics import average_precision_score
    best = meta["val_pr_auc"][meta["best_epoch"] - 1]
    assert average_precision_score(v, scores) == pytest.approx(best, abs=1e-6)
    assert best == max(meta["val_pr_auc"])


def test_early_stopping_patience(matrices, monkeypatch):
    X, y, V, v = matrices
    # a constant validation score never improves after epoch 1
    monkeypatch.setattr(mlp, "average_precision_score", lambda *a: 0.5)
    model = mlp.fit(X, y, V, v, {"hidden": 8, "max_epochs": 30, "patience": 3}, seed=0)
    assert model.run_metadata["best_epoch"] == 1
    assert model.run_metadata["epochs_run"] == 4


def test_same_seed_same_scores(matrices):
    X, y, V, v = matrices
    p = {"hidden": 16, "max_epochs": 3}
    a = mlp.predict_proba(mlp.fit(X, y, V, v, p, seed=1), V)
    b = mlp.predict_proba(mlp.fit(X, y, V, v, p, seed=1), V)
    c = mlp.predict_proba(mlp.fit(X, y, V, v, p, seed=2), V)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_validation_rows_come_in_pairs(matrices):
    X, y, V, _ = matrices
    with pytest.raises(ValueError, match="X_val"):
        mlp.fit(X, y, V, None, {})


def test_without_validation_runs_exactly_max_epochs(matrices):
    """The refit after early stopping: no validation rows, a fixed number of epochs, final weights kept."""
    X, y, V, v = matrices
    stopped = mlp.fit(X, y, V, v, {"hidden": 8, "max_epochs": 6}, seed=0)
    rounds = mlp.refit_params(stopped)
    assert rounds == {"max_epochs": stopped.run_metadata["best_epoch"]}
    refit = mlp.fit(X, y, None, None, {"hidden": 8, **rounds}, seed=0)
    assert refit.run_metadata["epochs_run"] == refit.run_metadata["best_epoch"] == rounds["max_epochs"]
    assert refit.run_metadata["val_pr_auc"] == []
    assert np.isfinite(mlp.predict_proba(refit, V)).all()


def test_run_cell_logs_cv_and_three_seeds(transactions, tmp_path):
    tr, va, _ = holdout_indices(transactions)
    log, best_path = tmp_path / "log.csv", tmp_path / "best.yaml"
    grid = [{"hidden": 8, "max_epochs": 3}, {"hidden": 16, "max_epochs": 3}]
    make = lambda: data.LinearPipeline(v_drop=[])
    best = run_cell(mlp, make, transactions, tr, va, "F0", grid, log_path=log, best_path=best_path)
    df = pd.read_csv(log)
    n_folds = len(time_cv_folds(transactions, tr))
    assert (df.split == "cv").sum() == len(grid) * n_folds
    assert sorted(df.loc[df.split == "val", "seed"]) == [0, 1, 2]
    assert (df.split != "test").all()
    fit = json.loads(df.loc[df.split == "val", "params"].iloc[0])["_fit"]
    assert len(fit["val_pr_auc"]) == fit["epochs_run"]
    assert yaml.safe_load(best_path.read_text())["models"]["mlp"]["F0"]["params"] == best
    # rerunning resumes: no new rows
    run_cell(mlp, make, transactions, tr, va, "F0", grid, log_path=log, best_path=best_path)
    assert len(pd.read_csv(log)) == len(df)

