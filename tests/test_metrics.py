import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src import metrics, results


@pytest.fixture
def scored():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.05).astype(int)
    score = rng.random(5000) + 0.5 * y
    return y, score


def test_pr_auc_matches_sklearn(scored):
    y, score = scored
    out = metrics.evaluate(y, score)
    assert out["pr_auc"] == pytest.approx(average_precision_score(y, score), abs=1e-12)
    assert out["auroc"] == pytest.approx(roc_auc_score(y, score), abs=1e-12)


def test_random_scores_give_pr_auc_near_positive_rate():
    rng = np.random.default_rng(1)
    y = (rng.random(200_000) < 0.035).astype(int)
    out = metrics.evaluate(y, rng.random(200_000))
    assert out["pr_auc"] == pytest.approx(y.mean(), abs=0.003)
    assert out["auroc"] == pytest.approx(0.5, abs=0.01)


def test_no_threshold_leaves_precision_recall_empty(scored):
    out = metrics.evaluate(*scored)
    assert out["precision"] is None and out["recall"] is None
    assert set(out) == {"pr_auc", "auroc", "precision", "recall", "recall_top1pct"}


def test_threshold_precision_recall():
    y = np.array([0, 0, 1, 1, 0, 1])
    s = np.array([0.1, 0.6, 0.7, 0.4, 0.2, 0.9])
    out = metrics.evaluate(y, s, threshold=0.5)
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(2 / 3)


def test_pick_threshold_maximises_f1(scored):
    y, score = scored
    t = metrics.pick_threshold(y, score)
    best = f1_score(y, score >= t)
    # brute force over a grid of candidate thresholds
    for c in np.quantile(score, np.linspace(0.5, 0.999, 200)):
        assert f1_score(y, score >= c) <= best + 1e-12


def test_recall_top1pct_exact():
    y = np.zeros(1000, dtype=int)
    y[[0, 1, 2, 500]] = 1
    s = np.linspace(1, 0, 1000)  # rows 0..9 are the top 1%
    assert metrics.recall_top1pct(y, s) == pytest.approx(3 / 4)


def test_recall_top1pct_ties_ignore_row_order():
    # 10 slots, 5 rows strictly above, 10 rows tied at the cutoff (2 of them fraud)
    y = np.zeros(1000, dtype=int)
    s = np.zeros(1000)
    s[:5] = 2.0
    s[5:15] = 1.0
    y[[0, 5, 6, 900]] = 1
    expected = (1 + 5 * 2 / 10) / 4
    assert metrics.recall_top1pct(y, s) == pytest.approx(expected)
    perm = np.random.default_rng(0).permutation(1000)
    assert metrics.recall_top1pct(y[perm], s[perm]) == pytest.approx(expected)


def run_fields(**over):
    fields = dict(author="tester", model="logreg", feature_set="F0", cv_scheme="time",
                  split="cv", seed=0, params={"C": 0.1, "fold": 1}, runtime_s=1.5,
                  **metrics.evaluate([0, 1, 0, 1], [0.1, 0.8, 0.3, 0.6]))
    fields.update(over)
    return fields


def test_log_run_twice_gives_header_and_two_rows(tmp_path):
    path = tmp_path / "results" / "log.csv"
    results.log_run(path, **run_fields())
    results.log_run(path, **run_fields(split="val", params={"C": 0.1}))
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(results.COLUMNS)
    assert len(lines) == 3
    df = pd.read_csv(path)
    assert list(df.columns) == results.COLUMNS
    assert df["split"].tolist() == ["cv", "val"]
    assert json.loads(df["params"][0]) == {"C": 0.1, "fold": 1}
    assert df["precision"].isna().all()
    assert df["date"].notna().all()
    assert b"\r" not in path.read_bytes()        # LF line endings, like the rest of results/log.csv


@pytest.mark.parametrize("over,message", [
    ({"fold": 1}, "Unknown"),
    ({"split": "train"}, "split must be"),
    ({"model": None}, "Missing"),
])
def test_log_run_rejects_bad_rows(tmp_path, over, message):
    path = tmp_path / "log.csv"
    with pytest.raises(ValueError, match=message):
        results.log_run(path, **run_fields(**over))
    assert not path.exists()


def test_log_run_rejects_foreign_header(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="header"):
        results.log_run(path, **run_fields())


def test_precision_top1pct_matches_count_and_recall():
    from src.metrics import precision_top1pct, top_rows
    rng = np.random.default_rng(5)
    y = (rng.random(20_000) < 0.03).astype(int)
    s = y * 0.6 + rng.random(20_000)
    k = top_rows(len(y))
    assert k == 200
    assert precision_top1pct(y, s) == pytest.approx(y[np.argsort(-s)[:k]].mean())
    assert precision_top1pct(y, s) == pytest.approx(metrics.recall_top1pct(y, s) * y.sum() / k)
