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


def test_known_uid_and_fold_references(transactions):
    from src import segments
    tr, va, _ = holdout_indices(transactions)
    known = segments.known_uid(transactions["uid"].iloc[va], set(transactions["uid"].iloc[tr]))
    assert known.dtype == bool and len(known) == len(va)
    assert list(segments.segment_labels([True, False])) == ["known", "unknown"]
    ref = segments.fold_reference_uids(transactions, tr)
    from src.split import time_cv_folds
    for k, (a, _) in enumerate(time_cv_folds(transactions, tr), start=1):
        assert ref[k] == set(transactions["uid"].iloc[a])       # only that fold's training months
    assert all(ref[k] <= ref[k + 1] for k in range(1, len(ref)))


def test_segment_pr_auc_with_bootstrap():
    from src import segments
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.1).astype(int)
    scores = [y * 0.5 + rng.random(4000), y * 0.4 + rng.random(4000)]
    known = rng.random(4000) < 0.5
    t = segments.segment_pr_auc(y, scores, known, n_boot=50, seed=1)
    assert list(t["segment"]) == ["known", "unknown"]
    assert t["rows"].sum() == 4000 and t["fraud"].sum() == y.sum()
    assert ((t["low"] <= t["pr_auc"]) & (t["pr_auc"] <= t["high"])).all()
    assert t.equals(segments.segment_pr_auc(y, scores, known, n_boot=50, seed=1))   # reproducible
    no_boot = segments.segment_pr_auc(y, scores[0], known, n_boot=0)
    assert no_boot["low"].isna().all() and no_boot["pr_auc"].notna().all()


def test_early_stopping_rows_are_held_out_of_scored_rows(transactions, tmp_path, monkeypatch):
    """G0-G3 runs: early-stopping models stop on the last 10% of their training part, never on scored rows."""
    from src.models import logreg
    from src.split import time_cv_folds
    tr, va, _ = holdout_indices(transactions)
    seen = []

    def spy(module):
        original = module.fit
        def fit(X, y, X_val=None, y_val=None, params=None, seed=0):
            seen.append((module.NAME, X.index.to_numpy(), None if X_val is None else X_val.index.to_numpy()))
            return original(X, y, X_val, y_val, params, seed)
        monkeypatch.setattr(module, "fit", fit)

    spy(lgbm)
    spy(logreg)
    sets = {"S": ["a", "b", "cat"]}
    tree = lambda cols: data.TreePipeline("lgbm", columns=cols)
    linear = lambda cols: data.LinearPipeline(columns=cols)
    experiment.tune_and_validate(lgbm, tree, transactions, tr, va, sets,
                                 [{"num_leaves": 7, "n_estimators": 20, "n_jobs": 1}], seeds=(0,),
                                 log_path=tmp_path / "log.csv", preds_dir=tmp_path / "p")
    experiment.tune_and_validate(logreg, linear, transactions, tr, va, sets, [{"C": 1.0}], seeds=(0,),
                                 log_path=tmp_path / "log.csv", preds_dir=tmp_path / "p")
    parts = [a for a, _ in time_cv_folds(transactions, tr)] + [tr]
    scored = [b for _, b in time_cv_folds(transactions, tr)] + [va]
    lg = [s for s in seen if s[0] == "lgbm"]
    lr = [s for s in seen if s[0] == "logreg"]
    assert len(lg) == len(lr) == len(parts)
    for (_, fit_rows, es_rows), part, score in zip(lg, parts, scored):
        fit_pos, es_pos = experiment.split_early_stopping(part, experiment.ES_FRACTION)
        assert np.array_equal(fit_rows, transactions.index[fit_pos])          # first 90% of the part
        assert np.array_equal(es_rows, transactions.index[es_pos])            # last 10%, by time
        assert not set(es_rows) & set(transactions.index[score])              # never the scored rows
    for (_, fit_rows, _), part in zip(lr, parts):
        assert np.array_equal(fit_rows, transactions.index[part])             # no early stopping: all rows
    fit_meta = [json.loads(p)["_fit"] for p in pd.read_csv(tmp_path / "log.csv").query("model == 'lgbm'").params]
    assert all(m["early_stopping_rows"] > 0 for m in fit_meta)
