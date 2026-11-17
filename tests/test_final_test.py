"""Step 19 final test scoring on synthetic data: freeze guard, settings and threshold taken from the
validation rows, the refit reproduces validation, one test row per cell, resuming, and the loader."""
import json
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from src import data, experiment, feature_sets as fs, final_test
from src.split import holdout_indices

SETS = {"A": ["a", "cat"], "B": ["a", "b", "cat"]}


@pytest.fixture
def transactions():
    rng = np.random.default_rng(5)
    n = 2000
    dt = np.linspace(86400, 151 * 86400, n).astype("int32")
    x = rng.normal(size=n).astype("float32")
    y = (rng.random(n) < 1 / (1 + np.exp(2.5 - 2 * x))).astype("int8")
    x[::13] = np.nan
    return pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionDT": dt, "TransactionDay": dt // 86400, "isFraud": y,
        "a": x, "b": rng.normal(size=n).astype("float32"),
        "cat": pd.Categorical(np.where(rng.random(n) < y * 0.5 + 0.25, "p", "q")),
        "uid": [f"u{i % 50}" for i in range(n)],
    })


def _validate(name, df, tmp_path, grid):
    """Run tuning and validation as a model notebook would, into a scratch log."""
    model = __import__(final_test.MODULES[name], fromlist=["x"])
    tr, va, _ = holdout_indices(df)
    log = tmp_path / "log.csv"
    experiment.tune_and_validate(model, lambda cols: final_test.make_pipeline(name, cols), df.iloc[:va[-1] + 1],
                                 tr, va, SETS, grid, tune_on="B", seeds=(0, 1), log_path=log,
                                 preds_dir=tmp_path / "preds")
    return log


def _run(name, df, tmp_path, log, **kw):
    tr, va, te = holdout_indices(df)
    final_test.run(name, sets=tuple(SETS), seeds=(0, 1), log_path=log, preds_dir=tmp_path / "preds",
                   data=(df, tr, va, te, SETS), **kw)
    return te


def test_freeze_guard(transactions, tmp_path):
    log = _validate("logreg", transactions, tmp_path, [{"C": 0.1}])
    before = log.read_text()
    with pytest.raises(PermissionError, match="freeze"):
        _run("logreg", transactions, tmp_path, log, cache_dir=None)
    assert log.read_text() == before


def test_one_test_row_per_cell_with_the_validated_settings(transactions, tmp_path, name="logreg",
                                                          grid=({"C": 0.1}, {"C": 1.0})):
    log = _validate(name, transactions, tmp_path, list(grid))
    te = _run(name, transactions, tmp_path, log, freeze_confirmed=True, cache_dir=None)
    rows = pd.read_csv(log)
    test, val = rows[rows.split == "test"], rows[rows.split == "val"]
    assert len(test) == len(SETS) * 2 and not test.duplicated(["feature_set", "seed"]).any()
    for r in test.itertuples():
        v = val[(val.feature_set == r.feature_set) & (val.seed == r.seed)].iloc[0]
        tp, vp = json.loads(r.params), json.loads(v.params)
        assert final_test.settings_of(tp) == final_test.settings_of(vp)
        assert tp["threshold"] == vp["threshold"]                          # stored threshold, not re-picked
        assert tp.get("es_setup") == vp.get("es_setup")                    # same early-stopping setup
        assert tp["_test"]["val_pr_auc_refit"] == pytest.approx(v.pr_auc, abs=1e-6)
        saved = pd.read_parquet(tmp_path / "preds" / f"{name}_{r.feature_set}_test_seed{r.seed}.parquet")
        assert saved["TransactionID"].tolist() == transactions["TransactionID"].iloc[te].tolist()
        from sklearn.metrics import average_precision_score
        assert average_precision_score(saved["isFraud"], saved["score"]) == pytest.approx(r.pr_auc)
    # rerunning adds nothing
    _run(name, transactions, tmp_path, log, freeze_confirmed=True, cache_dir=None)
    assert len(pd.read_csv(log)) == len(rows)


def test_refit_must_reproduce_validation(transactions, tmp_path):
    log = _validate("logreg", transactions, tmp_path, [{"C": 0.1}])
    rows = pd.read_csv(log)
    i = rows.index[(rows.split == "val") & (rows.feature_set == "A") & (rows.seed == 0)][0]
    rows.loc[i, "pr_auc"] += 0.01                         # as if the code or data changed since validation
    rows.to_csv(log, index=False)
    with pytest.raises(RuntimeError, match="would not belong"):
        _run("logreg", transactions, tmp_path, log, freeze_confirmed=True, cache_dir=None)
    assert (pd.read_csv(log).split != "test").all()


def test_early_stopping_model_in_a_subprocess(tmp_path):
    """LightGBM (two-step early stopping) end to end, plus reuse of a cached refit. Runs in its own
    process: on macOS LightGBM can crash once PyTorch (loaded by tests/conftest.py) is in the process;
    the final-test script runs one model per process, so it is not affected."""
    code = textwrap.dedent(f"""
        import json, pathlib
        from src.models import lgbm
        import numpy as np, pandas as pd
        from src import experiment, final_test
        from src.split import holdout_indices
        rng = np.random.default_rng(5); n = 2000
        dt = np.linspace(86400, 151 * 86400, n).astype("int32")
        x = rng.normal(size=n).astype("float32")
        y = (rng.random(n) < 1 / (1 + np.exp(2.5 - 2 * x))).astype("int8"); x[::13] = np.nan
        df = pd.DataFrame({{"TransactionID": np.arange(n), "TransactionDT": dt, "TransactionDay": dt // 86400,
            "isFraud": y, "a": x, "b": rng.normal(size=n).astype("float32"),
            "cat": pd.Categorical(np.where(rng.random(n) < y * 0.5 + 0.25, "p", "q")),
            "uid": [f"u{{i % 50}}" for i in range(n)]}})
        SETS = {SETS!r}
        tmp = pathlib.Path({str(tmp_path)!r}); log = tmp / "log.csv"
        tr, va, te = holdout_indices(df)
        build = lambda c: final_test.make_pipeline("lgbm", c)
        experiment.tune_and_validate(lgbm, build, df.iloc[:va[-1] + 1], tr, va, SETS,
                                     [{{"num_leaves": 7, "learning_rate": 0.1}}], tune_on="B", seeds=(0, 1),
                                     log_path=log, preds_dir=tmp / "preds")
        v = pd.read_csv(log); v = v[v.split == "val"]
        p = final_test.settings_of(v[(v.feature_set == "B") & (v.seed == 0)].params.iloc[0])
        experiment.fit_final(lgbm, lambda c=SETS["B"]: build(c), df, tr, p, "B", seed=0, cache_dir=tmp / "models")
        final_test.run("lgbm", True, sets=tuple(SETS), seeds=(0, 1), log_path=log, preds_dir=tmp / "preds",
                       cache_dir=tmp / "models", data=(df, tr, va, te, SETS))
        rows = pd.read_csv(log); t = rows[rows.split == "test"]
        assert len(t) == 4
        for r in t.itertuples():
            tp = json.loads(r.params)
            vp = json.loads(v[(v.feature_set == r.feature_set) & (v.seed == r.seed)].params.iloc[0])
            assert tp["es_setup"] == vp["es_setup"] == experiment.ES_SETUP
            assert tp["threshold"] == vp["threshold"]
            assert final_test.settings_of(tp) == final_test.settings_of(vp)
            assert tp["_test"]["reused_cached_model"] == (r.feature_set == "B" and r.seed == 0)
        assert len(list((tmp / "models").glob("*.pkl"))) == 1
        print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(data.ROOT), timeout=600)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr[-3000:]


def test_missing_validation_row_stops_the_run(transactions, tmp_path):
    log = _validate("logreg", transactions, tmp_path, [{"C": 0.1}])
    rows = pd.read_csv(log)
    rows = rows.drop(rows.index[(rows.split == "val") & (rows.feature_set == "B") & (rows.seed == 1)])
    rows.to_csv(log, index=False)
    with pytest.raises(ValueError, match="expected 1 current validation row"):
        _run("logreg", transactions, tmp_path, log, freeze_confirmed=True, cache_dir=None)


def test_report_tables(transactions, tmp_path):
    log = _validate("logreg", transactions, tmp_path, [{"C": 0.1}])
    _run("logreg", transactions, tmp_path, log, freeze_confirmed=True, cache_dir=None)
    rows = pd.read_csv(log).assign(feature_set=lambda d: d.feature_set.map({"A": "G0", "B": "G3"}).fillna(d.feature_set))
    t = final_test.results_table(rows)
    assert set(t.index) == {("logreg", "G0"), ("logreg", "G3")} and (t["count"] == 2).all()
    text = final_test.render(rows)
    assert "Logistic regression" in text and "not run" in text


def test_loader_with_test_block():
    spec = fs.load_spec()
    if not (data.ROOT / spec["parquet"]).exists():
        pytest.skip("features_v2.parquet not built")
    df, tr, va, te = fs.load_features_with_test(spec)
    assert len(tr) == 472_432 and len(va) == len(te) == 59_054 and te[-1] == len(df) - 1
    assert df["TransactionDT"].is_monotonic_increasing
    base, tr2, va2 = fs.load_features(spec)
    assert len(base) == te[0] and np.array_equal(tr, tr2) and np.array_equal(va, va2)
