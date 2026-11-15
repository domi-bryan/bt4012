"""Check model interfaces, fold-local weighting, stopping and interrupted-run recovery."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src import data, experiment
from src.models import lgbm, rf
from src.split import holdout_indices, time_cv_folds
from src.train_trees import run_cell


@pytest.fixture
def transactions():
    rng = np.random.default_rng(2)
    n = 1200
    dt = np.linspace(86400, 151 * 86400, n).astype("int32")
    x = rng.normal(size=n).astype("float32")
    y = (rng.random(n) < 1 / (1 + np.exp(2 - 2 * x))).astype("int8")
    x[::13] = np.nan
    return pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionDT": dt, "TransactionDay": dt // 86400,
        "TransactionAmt": x, "ProductCD": pd.Categorical(np.where(y, "C", "W")),
        "isFraud": y,
    })


@pytest.mark.parametrize("module", [rf, lgbm])
def test_model_contract_and_training_weights(transactions, module):
    tr, va, _ = holdout_indices(transactions)
    pipe = data.TreePipeline(module.NAME, v_drop=[]).fit(transactions.iloc[tr])
    X, V = pipe.transform(transactions.iloc[tr]), pipe.transform(transactions.iloc[va])
    y, v = transactions.isFraud.iloc[tr], transactions.isFraud.iloc[va]
    params = {"n_jobs": 1} if module is rf else {"n_jobs": 1, "n_estimators": 100}
    fitted = module.fit(X, y, V, v, params, seed=2)
    scores = module.predict_proba(fitted, V)
    assert scores.shape == (len(va),)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    if module is rf:
        assert fitted.clf.n_estimators == 300
        assert fitted.clf.class_weight == "balanced_subsample"
    else:
        assert fitted.run_metadata["resolved_scale_pos_weight"] == pytest.approx((y == 0).sum() / y.sum())
        assert 0 < fitted.run_metadata["best_iteration"] < 100
        # Only the explicitly supplied validation part drives early stopping.
        assert list(fitted.clf.evals_result_) == ["valid_0"]
        assert list(fitted.clf.evals_result_["valid_0"]) == ["pr_auc"]


def test_resume_missing_fold_and_seeds(transactions, tmp_path, monkeypatch):
    tr, va, _ = holdout_indices(transactions)
    log = tmp_path / "log.csv"
    best = tmp_path / "best.yaml"
    make = lambda: data.TreePipeline("lgbm", v_drop=[])
    grid = [{"num_leaves": 7, "n_estimators": 10, "n_jobs": 1}]
    original_fit = lgbm.fit
    train_lengths = []

    def track(X, y, *args, **kwargs):
        train_lengths.append(len(y))
        return original_fit(X, y, *args, **kwargs)

    monkeypatch.setattr(lgbm, "fit", track)
    selected = run_cell(lgbm, make, transactions, tr, va, "F0", grid, log, best)
    recorded = pd.read_csv(log)
    folds = len(time_cv_folds(transactions, tr))
    assert len(recorded) == folds + 3
    assert set(recorded.split) == {"cv", "val"}
    assert recorded.loc[recorded.split == "val", "seed"].tolist() == [0, 1, 2]
    assert max(train_lengths) == len(tr)
    assert yaml.safe_load(best.read_text())["models"]["lgbm"]["F0"]["params"] == selected
    assert "_fit" in json.loads(recorded.params.iloc[0])

    # A complete resume neither refits nor appends rows.
    train_lengths.clear()
    run_cell(lgbm, make, transactions, tr, va, "F0", grid, log, best)
    assert train_lengths == []
    assert len(pd.read_csv(log)) == len(recorded)

    # Simulate interruption: recover only a missing CV fold and validation seed.
    recorded.drop(index=[0, len(recorded) - 1]).to_csv(log, index=False)
    run_cell(lgbm, make, transactions, tr, va, "F0", grid, log, best)
    assert len(train_lengths) == 2
    assert len(pd.read_csv(log)) == len(recorded)


def test_best_params_with_mixed_weight_types():
    rows = pd.DataFrame({
        "params": [{"scale_pos_weight": "balanced"}, {"scale_pos_weight": 1.0}],
        "pr_auc": [0.5, 0.6],
    })
    assert experiment.best_params(rows) == {"scale_pos_weight": 1.0}


def test_manifest_rejects_changed_inputs(tmp_path, monkeypatch):
    from src.train_trees import check_manifest

    files = ["data/processed/cleaned.parquet", "config/features.yaml",
             "src/data.py", "src/features.py", "src/split.py", "src/metrics.py",
             "src/experiment.py", "src/models/logreg.py", "src/models/rf.py", "src/models/lgbm.py"]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original")
    monkeypatch.setattr(data, "ROOT", tmp_path)
    manifest = tmp_path / "manifest.json"
    check_manifest(manifest)
    original = manifest.read_bytes()
    check_manifest(manifest)
    assert manifest.read_bytes() == original
    (tmp_path / "config/features.yaml").write_text("changed")
    with pytest.raises(ValueError, match="inputs changed"):
        check_manifest(manifest)
    assert manifest.read_bytes() == original
