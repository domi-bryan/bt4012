"""Blend weights, ranking within folds, the fold-score weight search and segment routing."""
import numpy as np
import pandas as pd
import pytest

from src import ensemble


def test_weight_grid_sums_to_one():
    grid = ensemble.weight_grid(["a", "b", "c", "d"], step=0.1)
    assert len(grid) == 286                      # compositions of 10 into 4 parts
    assert all(np.isclose(sum(w.values()), 1) for w in grid)
    assert all(min(w.values()) >= 0 for w in grid)
    with pytest.raises(ValueError):
        ensemble.weight_grid(["a"], step=0.3)


def test_rank_within_fold():
    df = pd.DataFrame({"fold": [1, 1, 2, 2], "m": [10.0, 20.0, 0.1, 0.2]})
    r = ensemble.rank_within(df, ["m"], by="fold")
    assert r["m"].tolist() == [0.5, 1.0, 0.5, 1.0]
    assert df["m"].tolist() == [10.0, 20.0, 0.1, 0.2]      # input unchanged


@pytest.fixture
def scores():
    rng = np.random.default_rng(0)
    n = 4000
    y = (rng.random(n) < 0.1).astype(int)
    known = rng.random(n) < 0.5
    noise = lambda: rng.normal(size=n)
    # "good" ranks fraud well everywhere, "known_only" only on known rows, "unknown_only" only on unknown rows
    df = pd.DataFrame({
        "isFraud": y, "fold": np.repeat([1, 2, 3, 4], n // 4), "known": known,
        "good": y * 1.0 + noise(),
        "known_only": np.where(known, y * 3.0, 0) + noise(),
        "unknown_only": np.where(~known, y * 3.0, 0) + noise(),
    })
    return ensemble.rank_within(df, ["good", "known_only", "unknown_only"], by="fold")


def test_search_finds_the_model_that_works_on_each_segment(scores):
    models = ["good", "known_only", "unknown_only"]
    k = scores["known"].to_numpy()
    best_k = ensemble.best_weights(ensemble.search(scores, models, mask=k), models)
    best_u = ensemble.best_weights(ensemble.search(scores, models, mask=~k), models)
    assert max(best_k, key=best_k.get) == "known_only"
    assert max(best_u, key=best_u.get) == "unknown_only"
    table = ensemble.search(scores, models)
    assert table["oof_pr_auc"].is_monotonic_decreasing and len(table) == 66


def test_segment_blend_routes_rows(scores):
    k = scores["known"].to_numpy()
    s = ensemble.segment_blend(scores, k, {"known_only": 1.0}, {"unknown_only": 1.0})
    assert np.array_equal(s[k], scores["known_only"].to_numpy()[k])
    assert np.array_equal(s[~k], scores["unknown_only"].to_numpy()[~k])


def test_fold_pr_auc_is_the_mean_over_folds(scores):
    from sklearn.metrics import average_precision_score as ap
    per = [ap(p["isFraud"], p["good"]) for _, p in scores.groupby("fold")]
    assert ensemble.fold_pr_auc(scores["isFraud"], scores["good"], scores["fold"]) == pytest.approx(np.mean(per))


def test_load_scores_checks_rows(tmp_path):
    a = pd.DataFrame({"TransactionID": [1, 2, 3], "isFraud": [0, 1, 0], "score": [.1, .9, .2]})
    a.to_parquet(tmp_path / "m1_G3_val_seed0.parquet")
    a.assign(score=[.3, .8, .1]).to_parquet(tmp_path / "m2_G3_val_seed0.parquet")
    t = ensemble.load_scores(tmp_path, ["m1", "m2"], "G3", "val", seed=0)
    assert list(t.columns) == ["TransactionID", "isFraud", "m1", "m2"] and t["m2"].tolist() == [.3, .8, .1]
    a.assign(TransactionID=[1, 2, 4]).to_parquet(tmp_path / "m3_G3_val_seed0.parquet")
    with pytest.raises(ValueError, match="different rows"):
        ensemble.load_scores(tmp_path, ["m1", "m3"], "G3", "val", seed=0)
    a.assign(isFraud=[1, 1, 0]).to_parquet(tmp_path / "m4_G3_val_seed0.parquet")
    with pytest.raises(ValueError, match="isFraud"):
        ensemble.load_scores(tmp_path, ["m1", "m4"], "G3", "val", seed=0)


# --- the adopted segment blend: validation and test rows from saved scores --------------------------

@pytest.fixture
def blend_inputs(tmp_path):
    """Saved val and test scores of the four input models (seeds 0, 1) and matching log rows."""
    import json
    from sklearn.metrics import average_precision_score as ap
    from src import results
    rng = np.random.default_rng(1)
    preds, log = tmp_path / "preds", tmp_path / "log.csv"
    preds.mkdir()
    uid = {}
    for split, start in (("val", 1000), ("test", 2000)):
        ids = np.arange(start, start + 600)
        y = (rng.random(600) < 0.1).astype(int)
        for i in ids:
            uid[i] = f"u{i % 80}"
        for seed in (0, 1):
            for m in ensemble.BLEND_MODELS:
                s = y * rng.random() + rng.normal(size=600)
                pd.DataFrame({"TransactionID": ids, "isFraud": y, "score": s}).to_parquet(
                    preds / f"{m}_G3_{split}_seed{seed}.parquet")
                results.log_run(log, model=m, feature_set="G3", cv_scheme="holdout", split=split, seed=seed,
                                params=json.dumps({"threshold": 0.5}), pr_auc=ap(y, s))
    uid_by_id = pd.Series(uid)
    train_uids = {f"u{i}" for i in range(40)}           # half the customers are "known"
    return log, preds, uid_by_id, train_uids


def test_blend_weights_are_fixed():
    assert sum(ensemble.BLEND_KNOWN.values()) == pytest.approx(1)
    assert sum(ensemble.BLEND_UNKNOWN.values()) == pytest.approx(1)
    assert ensemble.BLEND_KNOWN == {"xgboost": 1.0}
    assert ensemble.BLEND_UNKNOWN == {"catboost": 0.6, "lgbm": 0.2, "xgboost": 0.1, "rf": 0.1}


def test_blend_validation_rows(blend_inputs):
    import json
    from src.metrics import pick_threshold
    log, preds, uid_by_id, train_uids = blend_inputs
    ensemble.log_blend("val", uid_by_id, train_uids, seeds=(0, 1), log_path=log, preds_dir=preds)
    rows = pd.read_csv(log).query("model == 'segment_blend'")
    assert len(rows) == 2 and set(rows.split) == {"val"} and set(rows.feature_set) == {"G3"}
    for r in rows.itertuples():
        t, s = ensemble.blend_scores(preds, "val", r.seed, uid_by_id, train_uids)
        p = json.loads(r.params)
        assert p["threshold"] == pytest.approx(pick_threshold(t["isFraud"], s))
        assert p["known"] == ensemble.BLEND_KNOWN and p["unknown"] == ensemble.BLEND_UNKNOWN
        saved = pd.read_parquet(ensemble.blend_preds_path(preds, "val", r.seed))
        from sklearn.metrics import average_precision_score as ap
        assert ap(saved["isFraud"], saved["score"]) == pytest.approx(r.pr_auc)
    ensemble.log_blend("val", uid_by_id, train_uids, seeds=(0, 1), log_path=log, preds_dir=preds)
    assert (pd.read_csv(log).model == "segment_blend").sum() == 2          # rerun adds nothing


def test_blend_inputs_must_match_the_log(blend_inputs):
    log, preds, uid_by_id, train_uids = blend_inputs
    p = preds / "catboost_G3_val_seed0.parquet"
    pd.read_parquet(p).assign(score=lambda d: d["score"][::-1].to_numpy()).to_parquet(p)
    with pytest.raises(ValueError, match="does not reproduce"):
        ensemble.log_blend("val", uid_by_id, train_uids, seeds=(0,), log_path=log, preds_dir=preds)
    assert not (pd.read_csv(log).model == "segment_blend").any()


def test_blend_test_rows_use_the_stored_threshold(blend_inputs):
    import json
    from src import final_test
    log, preds, uid_by_id, train_uids = blend_inputs
    with pytest.raises(PermissionError, match="freeze"):
        final_test.run_blend(seeds=(0, 1), log_path=log, preds_dir=preds, lookup=(uid_by_id, train_uids))
    with pytest.raises(ValueError, match="no validation row"):
        final_test.run_blend(True, seeds=(0,), log_path=log, preds_dir=preds, lookup=(uid_by_id, train_uids))
    ensemble.log_blend("val", uid_by_id, train_uids, seeds=(0, 1), log_path=log, preds_dir=preds)
    final_test.run_blend(True, seeds=(0, 1), log_path=log, preds_dir=preds, lookup=(uid_by_id, train_uids))
    rows = pd.read_csv(log).query("model == 'segment_blend'")
    val, test = rows[rows.split == "val"], rows[rows.split == "test"]
    assert len(test) == 2
    for r in test.itertuples():
        v = val[val.seed == r.seed].iloc[0]
        assert json.loads(r.params)["threshold"] == json.loads(v.params)["threshold"]
    assert "Segment blend (G3)" in final_test.render(pd.read_csv(log))
