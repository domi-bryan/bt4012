"""G0-G3 feature sets, explicit pipeline columns, the XGBoost model and the notebook runner."""
import json
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from src import data, experiment, feature_sets as fs
from src.catboost_pipeline import CatBoostPipeline
from src.models import lgbm
from src.split import holdout_indices


@pytest.fixture
def transactions():
    rng = np.random.default_rng(3)
    n = 1500
    dt = np.linspace(86400, 151 * 86400, n).astype("int32")
    x = rng.normal(size=n).astype("float32")
    y = (rng.random(n) < 1 / (1 + np.exp(2.5 - 2 * x))).astype("int8")
    x2 = rng.normal(size=n).astype("float32")
    x[::13] = np.nan
    return pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionDT": dt, "TransactionDay": dt // 86400,
        "isFraud": y, "a": x, "b": x2, "noise": rng.normal(size=n).astype("float32"),
        "cat": pd.Categorical(np.where(rng.random(n) < y * 0.5 + 0.25, "p", "q")),
        "uid": [f"u{i % 40}" for i in range(n)],
    })


def test_feature_sets_are_cumulative_and_clean():
    spec = fs.load_spec()
    sizes = {g: len(fs.feature_list(g, spec)) for g in fs.FEATURE_SETS}
    assert sizes == {"G0": 197, "G1": 224, "G2": 234, "G3": 252}
    for small, big in zip(fs.FEATURE_SETS, fs.FEATURE_SETS[1:]):
        assert set(fs.feature_list(small, spec)) < set(fs.feature_list(big, spec))
    assert not fs.NOT_INPUTS & set(fs.feature_list("G3", spec))
    with pytest.raises(ValueError, match="Unknown feature set"):
        fs.feature_list("F2", spec)


def test_load_features_drops_test_block():
    spec = fs.load_spec()
    if not (data.ROOT / spec["parquet"]).exists():
        pytest.skip("features_v2.parquet not built")
    df, train_idx, val_idx = fs.load_features(spec)
    assert len(train_idx) == spec["train_block_rows"] == 472_432
    assert len(df) == val_idx[-1] + 1 == 472_432 + 59_054          # nothing after the validation block
    assert set(fs.feature_list("G3", spec)) <= set(df.columns)


def test_pipelines_use_exactly_the_given_columns(transactions):
    cols = ["b", "cat", "a"]
    for pipe in (data.TreePipeline("lgbm", columns=cols), data.TreePipeline("rf", columns=cols),
                 data.LinearPipeline(columns=cols), CatBoostPipeline(columns=cols)):
        pipe.fit(transactions)                                   # no V list or config/features.yaml needed
        out = pipe.transform(transactions)
        used = {c.split("==")[0].split("__")[0] for c in out.columns}
        assert used == set(cols)
    with pytest.raises(ValueError, match="raw uid"):
        CatBoostPipeline(columns=["a", "uid"]).fit(transactions)


def test_xgb_contract():
    """Runs in a separate process: on macOS, XGBoost's OpenMP runtime crashes once PyTorch is
    loaded first, which tests/conftest.py does for LightGBM's sake. Each model notebook runs
    one model per kernel, so the notebooks are not affected."""
    code = textwrap.dedent("""
        import numpy as np, pandas as pd
        from src.models import xgb
        from src import data
        from src.split import holdout_indices
        rng = np.random.default_rng(3); n = 1500
        dt = np.linspace(86400, 151 * 86400, n).astype("int32")
        x = rng.normal(size=n).astype("float32")
        y = (rng.random(n) < 1 / (1 + np.exp(2.5 - 2 * x))).astype("int8")
        x[::13] = np.nan
        df = pd.DataFrame({"TransactionID": np.arange(n), "TransactionDT": dt, "TransactionDay": dt // 86400,
                           "isFraud": y, "a": x, "b": rng.normal(size=n).astype("float32"),
                           "cat": pd.Categorical(np.where(rng.random(n) < y * 0.5 + 0.25, "p", "q"))})
        tr, va, _ = holdout_indices(df)
        pipe = data.TreePipeline("lgbm", columns=["a", "b", "cat"]).fit(df.iloc[tr])
        X, V = pipe.transform(df.iloc[tr]), pipe.transform(df.iloc[va])
        yt, yv = df.isFraud.iloc[tr], df.isFraud.iloc[va]
        params = {"n_jobs": 1, "n_estimators": 200}
        fitted = xgb.fit(X, yt, V, yv, params, seed=1)
        s = xgb.predict_proba(fitted, V)
        assert s.shape == (len(va),) and np.isfinite(s).all() and ((s >= 0) & (s <= 1)).all()
        assert abs(fitted.run_metadata["resolved_scale_pos_weight"] - (yt == 0).sum() / yt.sum()) < 1e-9
        assert 0 < fitted.run_metadata["best_iteration"] < 200
        assert np.array_equal(s, xgb.predict_proba(xgb.fit(X, yt, V, yv, params, seed=1), V))
        try:
            xgb.fit(X, yt, V, None, params)
            raise SystemExit("missing y_val was accepted")
        except ValueError as e:
            assert "Supply both" in str(e)
        print("ok")
    """)
    r = subprocess.run([sys.executable, "-c", code], cwd=data.ROOT, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-3000:]


def test_tune_and_validate_resumes_and_saves_scores(transactions, tmp_path):
    tr, va, _ = holdout_indices(transactions)
    sets = {"S1": ["a", "cat"], "S2": ["a", "b", "cat"]}
    grid = [{"num_leaves": 7, "n_estimators": 20, "n_jobs": 1},
            {"num_leaves": 15, "n_estimators": 20, "n_jobs": 1}]
    log, preds = tmp_path / "log.csv", tmp_path / "preds"
    make = lambda cols: data.TreePipeline("lgbm", columns=cols)
    kwargs = dict(tune_on="S2", seeds=(0, 1), log_path=log, preds_dir=preds, save_oof=True)
    chosen = experiment.tune_and_validate(lgbm, make, transactions, tr, va, sets, grid, **kwargs)
    assert chosen["S1"] == chosen["S2"] and chosen["S2"] in grid
    rows = pd.read_csv(log)
    folds = len(rows[rows.split == "cv"]) // len(grid)
    assert set(rows[rows.split == "cv"].feature_set) == {"S2"}            # tuned on S2 only
    assert len(rows[rows.split == "val"]) == 4                             # 2 sets x 2 seeds
    for name in sets:
        for seed in (0, 1):
            p = pd.read_parquet(preds / f"lgbm_{name}_val_seed{seed}.parquet")
            assert len(p) == len(va) and json.loads(p.params[0]) == chosen[name]
        oof = pd.read_parquet(preds / f"lgbm_{name}_oof_seed0.parquet")
        assert oof.fold.nunique() == folds and set(oof.TransactionID) <= set(transactions.TransactionID.iloc[tr])
    experiment.tune_and_validate(lgbm, make, transactions, tr, va, sets, grid, **kwargs)
    assert len(pd.read_csv(log)) == len(rows)                              # a rerun adds nothing
    cv, val = experiment.run_summary("lgbm", list(sets), log)
    assert set(val.feature_set) == set(sets) and (val.seeds == 2).all()
    assert cv.folds.eq(folds).all()
