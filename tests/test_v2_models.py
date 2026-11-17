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
        rounds = xgb.refit_params(fitted)                  # refit without early stopping, fixed tree count
        assert rounds == {"n_estimators": fitted.run_metadata["best_iteration"]}
        refit = xgb.fit(X, yt, None, None, {**params, **rounds}, seed=1)
        assert refit.clf.get_booster().num_boosted_rounds() == rounds["n_estimators"]
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
    """G0-G3 runs: early-stopping models stop on the last 10% of their training part, never on scored rows.

    Tuning fits stop there and are scored as they are; the validation fit then refits on the whole
    training block with the stopping point it found (two fits). Logistic regression trains on all rows.
    """
    from src.models import logreg
    from src.split import time_cv_folds
    tr, va, _ = holdout_indices(transactions)
    seen = []

    def spy(module):
        original = module.fit
        def fit(X, y, X_val=None, y_val=None, params=None, seed=0):
            fitted = original(X, y, X_val, y_val, params, seed)
            seen.append((module.NAME, X.index.to_numpy(), None if X_val is None else X_val.index.to_numpy(),
                         dict(params or {}), getattr(fitted, "run_metadata", {})))
            return fitted
        monkeypatch.setattr(module, "fit", fit)

    spy(lgbm)
    spy(logreg)
    sets = {"S": ["a", "b", "cat"]}
    tree = lambda cols: data.TreePipeline("lgbm", columns=cols)
    linear = lambda cols: data.LinearPipeline(columns=cols)
    experiment.tune_and_validate(lgbm, tree, transactions, tr, va, sets,
                                 [{"num_leaves": 7, "n_estimators": 200, "n_jobs": 1}], seeds=(0,),
                                 log_path=tmp_path / "log.csv", preds_dir=tmp_path / "p")
    experiment.tune_and_validate(logreg, linear, transactions, tr, va, sets, [{"C": 1.0}], seeds=(0,),
                                 log_path=tmp_path / "log.csv", preds_dir=tmp_path / "p")
    folds = time_cv_folds(transactions, tr)
    lg = [s for s in seen if s[0] == "lgbm"]
    lr = [s for s in seen if s[0] == "logreg"]
    assert len(lg) == len(folds) + 2 and len(lr) == len(folds) + 1
    # tuning folds and step 1 of the validation fit: first 90% of the part, stop on the last 10%
    for (_, fit_rows, es_rows, _, _), (part, score) in zip(lg, folds + [(tr, va)]):
        fit_pos, es_pos = experiment.split_early_stopping(part, experiment.ES_FRACTION)
        assert np.array_equal(fit_rows, transactions.index[fit_pos])
        assert np.array_equal(es_rows, transactions.index[es_pos])
        assert not set(es_rows) & set(transactions.index[score])              # never the scored rows
    # step 2: the whole training block, no early stopping, the stopping point from step 1
    _, fit_rows, es_rows, params, _ = lg[-1]
    step1_meta = lg[-2][4]
    assert np.array_equal(fit_rows, transactions.index[tr]) and es_rows is None
    assert params["n_estimators"] == step1_meta["best_iteration"] < 200
    for (_, fit_rows, _, _, _), part in zip(lr, [a for a, _ in folds] + [tr]):
        assert np.array_equal(fit_rows, transactions.index[part])             # no early stopping: all rows
    log = pd.read_csv(tmp_path / "log.csv")
    lg_params = [json.loads(p) for p in log.query("model == 'lgbm'").params]
    assert all(p["es_setup"] == experiment.ES_SETUP and p["_fit"]["early_stopping_rows"] > 0 for p in lg_params)
    val = [p for p in lg_params if "threshold" in p][0]
    assert val["_fit"]["refit"] == {"n_estimators": step1_meta["best_iteration"]}
    assert all("es_setup" not in json.loads(p) for p in log.query("model == 'logreg'").params)


def test_refit_params_fix_the_stopping_point(transactions):
    """LightGBM and CatBoost: the refit trains exactly the trees early stopping kept (XGBoost: subprocess test)."""
    from src.models import catboost
    tr, va, _ = holdout_indices(transactions)
    cols = ["a", "b", "cat"]
    tp = data.TreePipeline("lgbm", columns=cols).fit(transactions.iloc[tr])
    X, V = tp.transform(transactions.iloc[tr]), tp.transform(transactions.iloc[va])
    y, v = transactions.isFraud.iloc[tr], transactions.isFraud.iloc[va]
    stopped = lgbm.fit(X, y, V, v, {"n_estimators": 300, "n_jobs": 1})
    rounds = lgbm.refit_params(stopped)
    assert rounds == {"n_estimators": stopped.run_metadata["best_iteration"]} and rounds["n_estimators"] < 300
    assert lgbm.fit(X, y, None, None, {"n_jobs": 1, **rounds}).clf.booster_.num_trees() == rounds["n_estimators"]
    cp = CatBoostPipeline(columns=cols).fit(transactions.iloc[tr])
    X, V = cp.transform(transactions.iloc[tr]), cp.transform(transactions.iloc[va])
    stopped = catboost.fit(X, y, V, v, {"iterations": 300, "thread_count": 1})
    rounds = catboost.refit_params(stopped)
    assert rounds == {"iterations": stopped.run_metadata["best_iteration"]} and rounds["iterations"] < 300
    assert catboost.fit(X, y, None, None, {"thread_count": 1, **rounds}).clf.tree_count_ == rounds["iterations"]
    assert lgbm.DEFAULTS["n_estimators"] == catboost.DEFAULTS["iterations"] == 3000


def test_older_early_stopping_runs_are_superseded(transactions, tmp_path):
    """Rows from the 1,500-tree, no-refit setup are rerun, not reused, and left out of the summaries."""
    tr, va, _ = holdout_indices(transactions)
    sets = {"S": ["a", "b", "cat"]}
    grid = [{"num_leaves": 7, "n_estimators": 20, "n_jobs": 1}]
    log, preds = tmp_path / "log.csv", tmp_path / "p"
    make = lambda cols: data.TreePipeline("lgbm", columns=cols)
    experiment.tune_and_validate(lgbm, make, transactions, tr, va, sets, grid, seeds=(0,), log_path=log,
                                 preds_dir=preds, save_oof=True)
    rows = pd.read_csv(log)
    old = rows.copy()                                  # the same runs as logged before es_setup existed
    old["params"] = old.params.map(lambda p: json.dumps({k: v for k, v in json.loads(p).items() if k != "es_setup"}))
    old["pr_auc"] = 0.0
    lr_row = rows.iloc[[0]].assign(model="logreg", params=json.dumps({"C": 1.0, "fold": 1, "_fit": {}}))
    pd.concat([old, lr_row]).to_csv(log, index=False)
    pd.read_parquet(preds / "lgbm_S_oof_seed0.parquet").drop(columns="es_setup").to_parquet(
        preds / "lgbm_S_oof_seed0.parquet")
    experiment.tune_and_validate(lgbm, make, transactions, tr, va, sets, grid, seeds=(0,), log_path=log,
                                 preds_dir=preds, save_oof=True)
    after = pd.read_csv(log)
    assert len(after) == len(old) + 1 + len(rows)                         # everything rerun once
    assert len(experiment.current_runs(after)) == len(rows) + 1           # old lgbm rows dropped, LR kept
    cv, val = experiment.run_summary("lgbm", ["S"], log)
    assert (cv.cv_pr_auc > 0).all() and (val.pr_auc_mean > 0).all() and len(val) == 1
    assert "es_setup" not in val.settings[0]
    assert (pd.read_parquet(preds / "lgbm_S_oof_seed0.parquet").es_setup == experiment.ES_SETUP).all()


def test_fit_final_matches_validation_run_and_is_cached(transactions, tmp_path):
    from sklearn.metrics import average_precision_score
    tr, va, _ = holdout_indices(transactions)
    cols = ["a", "b", "cat"]
    make = lambda: data.TreePipeline("lgbm", columns=cols)
    params = {"num_leaves": 7, "n_estimators": 30, "n_jobs": 1}
    val = experiment.fit_validate(lgbm, make, transactions, tr, va, params, "S", log_path=tmp_path / "log.csv",
                                  es_fraction=experiment.ES_FRACTION)
    fitted, pipe = experiment.fit_final(lgbm, make, transactions, tr, params, "S", cache_dir=tmp_path)
    X = pipe.transform(transactions.iloc[va])
    assert average_precision_score(transactions.isFraud.iloc[va], fitted.predict_proba(X)) == pytest.approx(val["pr_auc"])
    assert (tmp_path / "lgbm_S_seed0.pkl").exists()
    cached, _ = experiment.fit_final(lgbm, make, transactions, tr, params, "S", cache_dir=tmp_path)
    assert np.array_equal(cached.predict_proba(X), fitted.predict_proba(X))


def test_importance_maps_pipeline_columns_and_ranks_the_signal(transactions):
    from src import importance
    from src.models import logreg
    tr, va, _ = holdout_indices(transactions)
    cols = ["a", "noise", "cat"]
    pipe = data.LinearPipeline(columns=cols).fit(transactions.iloc[tr])
    assert importance.column_blocks(pipe.columns_, cols) == {
        "a": ["a", "a__missing"], "noise": ["noise"], "cat": ["cat==p", "cat==q", "cat==__nan__"]}
    X, y = pipe.transform(transactions.iloc[tr]), transactions.isFraud.iloc[tr]
    fitted = logreg.fit(X, y)
    built_in = importance.intrinsic(logreg, fitted, list(X.columns), cols)
    assert list(built_in.index) == cols and built_in.idxmax() == "a"
    V, v = pipe.transform(transactions.iloc[va]), transactions.isFraud.iloc[va]
    perm = importance.permutation(fitted, V, v, cols, n_rows=len(va), repeats=3, seed=1)
    assert perm["pr_auc_drop"].idxmax() == "a" and perm["drop_sd"].notna().all()
    # tree input keeps category dtypes when shuffled
    tp = data.TreePipeline("lgbm", columns=cols).fit(transactions.iloc[tr])
    T = tp.transform(transactions.iloc[va])
    f2 = lgbm.fit(tp.transform(transactions.iloc[tr]), y, T, v, {"n_estimators": 30, "n_jobs": 1})
    assert set(importance.permutation(f2, T, v, cols, n_rows=len(va)).index) == set(cols)


def test_setting_sensitivity_from_the_tuning_log(tmp_path):
    rows = []
    for c, score in [(0.01, 0.30), (0.1, 0.40), (1.0, 0.35)]:
        for leaf, bump in [(5, 0.02), (20, 0.0)]:
            for fold in (1, 2):
                rows.append({"model": "m", "feature_set": "G3", "cv_scheme": "time", "split": "cv", "seed": 0,
                             "params": json.dumps({"C": c, "leaf": leaf, "fold": fold}),
                             "pr_auc": score + bump + 0.01 * fold})
    rows.append({"model": "m", "feature_set": "G3", "cv_scheme": "time", "split": "cv", "seed": 0,       # incomplete: 1 fold only
                 "params": json.dumps({"C": 10.0, "leaf": 5, "fold": 1}), "pr_auc": 0.9})
    pd.DataFrame(rows).to_csv(tmp_path / "log.csv", index=False)
    s = experiment.setting_sensitivity("m", "G3", tmp_path / "log.csv")
    c = s[s.setting == "C"].set_index("value")
    assert list(c.index) == [0.01, 0.1, 1.0]                       # numeric order; the 1-fold run is ignored
    assert c.loc[0.1, "mean"] == pytest.approx(0.425)              # mean over leaf {5, 20}, folds averaged
    assert c.loc[0.1, "best"] == pytest.approx(0.435) and c.loc[0.1, "combinations"] == 2
    leaf = s[s.setting == "leaf"].set_index("value")
    assert leaf.loc[5, "mean"] - leaf.loc[20, "mean"] == pytest.approx(0.02)


def test_tuning_summaries_leave_out_leakage_check_rows(tmp_path):
    """Random and group folds logged by the leakage check share the chosen settings; they must not
    enter the time-CV means or change which combinations count as complete."""
    rows = [{"model": "m", "feature_set": "G3", "cv_scheme": "time", "split": "cv", "seed": 0,
             "params": json.dumps({"C": c, "fold": f}), "pr_auc": score}
            for c, score in [(0.1, 0.4), (1.0, 0.3)] for f in (1, 2)]
    rows += [{"model": "m", "feature_set": "G3", "cv_scheme": scheme, "split": "cv", "seed": 0,
              "params": json.dumps({"C": 0.1, "fold": f}), "pr_auc": 0.9}
             for scheme in ("random", "group_uid") for f in (1, 2, 3)]
    metrics = {"auroc": 0.9, "recall_top1pct": 0.3, "precision": None, "recall": None}
    pd.DataFrame([{**r, **metrics} for r in rows]).to_csv(tmp_path / "log.csv", index=False)
    s = experiment.setting_sensitivity("m", "G3", tmp_path / "log.csv").set_index("value")
    assert list(s.index) == [0.1, 1.0] and s.loc[0.1, "mean"] == pytest.approx(0.4)
    cv, _ = experiment.run_summary("m", ["G3"], tmp_path / "log.csv")
    assert cv.set_index("settings").loc['{"C": 0.1}', "cv_pr_auc"] == pytest.approx(0.4)
    assert (cv["folds"] == 2).all()


def test_setting_sensitivity_keeps_none_and_integers(tmp_path):
    rows = [{"model": "rf", "feature_set": "G3", "cv_scheme": "time", "split": "cv", "seed": 0,
             "params": json.dumps({"max_depth": d, "leaf": 5, "fold": f}), "pr_auc": score}
            for d, score in [(12, 0.3), (None, 0.4)] for f in (1, 2)]
    pd.DataFrame(rows).to_csv(tmp_path / "log.csv", index=False)
    s = experiment.setting_sensitivity("rf", "G3", tmp_path / "log.csv")
    assert list(s["value"]) == [12, "none"] and isinstance(s["value"][0], int)


def test_setting_sensitivity_value_column_keeps_types(tmp_path):
    rows = [{"model": "m", "feature_set": "G3", "cv_scheme": "time", "split": "cv", "seed": 0,
             "params": json.dumps({"leaves": lv, "lr": lr, "fold": 1}), "pr_auc": 0.1 * lv / 15 + lr}
            for lv in (15, 31) for lr in (0.03, 0.1)]
    pd.DataFrame(rows).to_csv(tmp_path / "log.csv", index=False)
    s = experiment.setting_sensitivity("m", "G3", tmp_path / "log.csv")
    assert all(isinstance(v, (int, np.integer)) for v in s[s.setting == "leaves"]["value"])   # not 15.0
