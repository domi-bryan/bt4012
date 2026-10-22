import json

import numpy as np
import pandas as pd
import pytest

from src import data, leakage
from src.models import logreg
from src.results import COLUMNS
from src.split import group_cv_folds, holdout_indices

BEST = {"logreg": {"C": 0.1}, "lgbm": {"num_leaves": 7}}


@pytest.fixture
def toy():
    rng = np.random.default_rng(4)
    n = 1200
    dt = np.linspace(86400, 151 * 86400, n).astype("int32")
    x = rng.normal(size=n).astype("float32")
    return pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionDT": dt, "TransactionDay": dt // 86400,
        "TransactionAmt": x, "isFraud": (rng.random(n) < 1 / (1 + np.exp(2 - 2 * x))).astype("int8"),
        "group": rng.integers(0, 150, size=n),
    })


def test_run_folds_logs_scheme_and_resumes(toy, tmp_path):
    log = tmp_path / "log.csv"
    tr, _, _ = holdout_indices(toy)
    y, g = toy.isFraud.to_numpy()[tr], toy.group.to_numpy()[tr]
    folds = group_cv_folds(tr, y, g, k=5, seed=0)
    for a, b in folds:
        assert set(toy.group.to_numpy()[a]).isdisjoint(toy.group.to_numpy()[b])
    make = lambda: data.LinearPipeline(v_drop=[])
    leakage.run_folds(logreg, make, toy.drop(columns="group"), folds, {"C": 0.1}, "group_uid", log)
    df = pd.read_csv(log)
    assert len(df) == 5 and (df.cv_scheme == "group_uid").all() and (df.split == "cv").all()
    assert [json.loads(p)["fold"] for p in df.params] == [1, 2, 3, 4, 5]
    assert (df.feature_set == "F2").all()
    # rerun skips every logged fold
    leakage.run_folds(logreg, make, toy.drop(columns="group"), folds, {"C": 0.1}, "group_uid", log)
    assert len(pd.read_csv(log)) == 5


def fake_log():
    rows = []
    base = {"date": "2026-10-22", "author": "t", "feature_set": "F2", "split": "cv", "seed": 0,
            "auroc": 0.9, "precision": None, "recall": None, "recall_top1pct": 0.3, "runtime_s": 1}
    score = {"random": 0.7, "group_uid": 0.6, "time": 0.5}
    for model, params in BEST.items():
        for scheme, n in leakage.N_FOLDS.items():
            for k in range(1, n + 1):
                p = {**params, "fold": k, "_fit": {"best_iteration": 3}} if model == "lgbm" else {**params, "fold": k}
                rows.append({**base, "model": model, "cv_scheme": scheme, "params": json.dumps(p),
                             "pr_auc": score[scheme] + 0.01 * k})
        # a time row for another config must be ignored
        rows.append({**base, "model": model, "cv_scheme": "time",
                     "params": json.dumps({**{k: 99 for k in params}, "fold": 1}), "pr_auc": 0.0})
    return pd.DataFrame(rows, columns=COLUMNS)


def test_table_and_report():
    t = leakage.leakage_table(fake_log(), BEST).set_index(["model", "scheme"])
    assert t.loc[("logreg", "random"), "mean"] == pytest.approx(0.73)
    assert t.loc[("lgbm", "time"), "n"] == 4
    assert t.loc[("lgbm", "time"), "folds"] == pytest.approx([0.51, 0.52, 0.53, 0.54])
    text = leakage.render(fake_log(), BEST)
    assert text.count("**random** scores highest") == 2
    assert "+0.2050 over time" in text            # 0.73 - 0.525


def test_incomplete_cell_rejected():
    log = fake_log()
    log = log.drop(index=log.index[(log.model == "logreg") & (log.cv_scheme == "group_uid")][0])
    with pytest.raises(ValueError, match="one row per fold"):
        leakage.leakage_table(log, BEST)


def test_table_from_csv_roundtrip(tmp_path):
    """The real log is read from CSV (Arrow string columns), including empty matches."""
    path = tmp_path / "log.csv"
    fake_log().to_csv(path, index=False)
    log = pd.read_csv(path)
    assert leakage.leakage_table(log, BEST).shape[0] == 6
    assert leakage.scheme_rows(log, "logreg", "random", {"C": 123}).empty
    # no candidate rows at all (a scheme never run yet) still returns rows with every column
    empty = leakage.scheme_rows(log, "logreg", "no_such_scheme", {"C": 0.1})
    assert empty.empty and "params" in empty.columns
