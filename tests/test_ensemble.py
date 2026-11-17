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
