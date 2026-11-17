import json

import numpy as np
import pandas as pd
import pytest

from src import data, experiment, leakage
from src.models import lgbm, logreg
from src.results import COLUMNS
from src.split import holdout_indices

BEST = {"logreg": {"C": 0.1}, "lgbm": {"num_leaves": 7}}
ES = {"es_setup": experiment.ES_SETUP}


@pytest.fixture
def toy():
    rng = np.random.default_rng(4)
    n = 1200
    dt = np.linspace(86400, 151 * 86400, n).astype("int32")
    x = rng.normal(size=n).astype("float32")
    return pd.DataFrame({
        "TransactionID": np.arange(n), "TransactionDT": dt, "TransactionDay": dt // 86400,
        "TransactionAmt": x, "isFraud": (rng.random(n) < 1 / (1 + np.exp(2 - 2 * x))).astype("int8"),
        "uid": [f"u{g}" for g in rng.integers(0, 150, size=n)],
    })


def test_make_folds_keep_each_uid_on_one_side(toy):
    tr, _, _ = holdout_indices(toy)
    folds = leakage.make_folds(toy, tr)
    assert {k: len(v) for k, v in folds.items()} == {"random": 5, "group_uid": 5}
    uid = toy.uid.to_numpy()
    for scheme in folds:
        for a, b in folds[scheme]:
            assert set(a) | set(b) == set(tr) and not set(a) & set(b)
    for a, b in folds["group_uid"]:
        assert set(uid[a]).isdisjoint(uid[b])
    # random folds do share customers, which the check rejects
    with pytest.raises(ValueError, match="both training and scored rows"):
        leakage.check_group_folds(folds["random"], uid)


def test_run_folds_logs_scheme_and_resumes(toy, tmp_path):
    log = tmp_path / "log.csv"
    tr, _, _ = holdout_indices(toy)
    folds = leakage.make_folds(toy, tr)["group_uid"]
    make = lambda: data.LinearPipeline(columns=["TransactionAmt"])
    leakage.run_folds(logreg, make, toy, folds, {"C": 0.1}, "group_uid", log)
    df = pd.read_csv(log)
    assert len(df) == 5 and (df.cv_scheme == "group_uid").all() and (df.split == "cv").all()
    assert [json.loads(p)["fold"] for p in df.params] == [1, 2, 3, 4, 5]
    assert (df.feature_set == "G3").all()
    assert all("es_setup" not in json.loads(p) for p in df.params)      # LR has no early stopping
    # rerun skips every logged fold
    leakage.run_folds(logreg, make, toy, folds, {"C": 0.1}, "group_uid", log)
    assert len(pd.read_csv(log)) == 5


def test_early_stopping_uses_held_out_training_rows(toy, tmp_path, monkeypatch):
    """LightGBM folds are fitted as tuning folds: one fit on the first 90% of the training part by
    time, stopping on the last 10%, and never on the scored rows."""
    tr, _, _ = holdout_indices(toy)
    folds = leakage.make_folds(toy, tr)["random"]
    seen = []
    original = lgbm.fit

    def fit(X, y, X_val=None, y_val=None, params=None, seed=0):
        seen.append((X.index.to_numpy(), X_val.index.to_numpy()))
        return original(X, y, X_val, y_val, params, seed)
    monkeypatch.setattr(lgbm, "fit", fit)

    params = {"num_leaves": 7, "n_estimators": 200, "n_jobs": 1}
    make = lambda: data.TreePipeline("lgbm", columns=["TransactionAmt"])
    leakage.run_folds(lgbm, make, toy, folds, params, "random", tmp_path / "log.csv")
    assert len(seen) == len(folds)                                       # one fit per fold, no refit
    for (fit_rows, es_rows), (a, b) in zip(seen, folds):
        fit_pos, es_pos = experiment.split_early_stopping(np.sort(a), experiment.ES_FRACTION)
        assert np.array_equal(fit_rows, toy.index[fit_pos])
        assert np.array_equal(es_rows, toy.index[es_pos])
        assert fit_rows.max() < es_rows.min()                            # the latest training rows
        assert not set(es_rows) & set(toy.index[b])                      # never the scored rows
    logged = [json.loads(p) for p in pd.read_csv(tmp_path / "log.csv").params]
    assert all(p["es_setup"] == experiment.ES_SETUP and p["_fit"]["early_stopping_rows"] > 0 for p in logged)
    assert all("refit" not in p["_fit"] for p in logged)
    # the logged rows are found again, so a rerun fits nothing
    leakage.run_folds(lgbm, make, toy, folds, params, "random", tmp_path / "log.csv")
    assert len(seen) == len(folds)


def fake_log():
    rows = []
    base = {"date": "2026-11-17", "author": "t", "feature_set": "G3", "seed": 0,
            "auroc": 0.9, "precision": None, "recall": None, "recall_top1pct": 0.3, "runtime_s": 1}
    score = {"random": 0.7, "group_uid": 0.6, "time": 0.5}
    held_out = {"_fit": {"best_iteration": 3, "early_stopping_rows": 9}}
    for model, params in BEST.items():
        run = {**params, **ES, **held_out} if model == "lgbm" else dict(params)
        for scheme, n in leakage.N_FOLDS.items():
            for k in range(1, n + 1):
                rows.append({**base, "model": model, "cv_scheme": scheme, "split": "cv",
                             "params": json.dumps({**run, "fold": k}), "pr_auc": score[scheme] + 0.01 * k})
        # a time row for other settings must be ignored
        rows.append({**base, "model": model, "cv_scheme": "time", "split": "cv",
                     "params": json.dumps({**{k: 99 for k in params}, "fold": 1}), "pr_auc": 0.0})
        # the validated settings
        for seed in (0, 1, 2):
            rows.append({**base, "model": model, "cv_scheme": "holdout", "split": "val", "seed": seed,
                         "params": json.dumps({**run, "threshold": 0.5}), "pr_auc": 0.4})
    # LightGBM rows from the older early-stopping setup: same settings in tuning, other settings validated
    rows.append({**base, "model": "lgbm", "cv_scheme": "time", "split": "cv",
                 "params": json.dumps({**BEST["lgbm"], **held_out, "fold": 1}), "pr_auc": 0.0})
    rows.append({**base, "model": "lgbm", "cv_scheme": "holdout", "split": "val",
                 "params": json.dumps({"num_leaves": 31, **held_out, "threshold": 0.5}), "pr_auc": 0.0})
    # another feature set is never read
    rows.append({**base, "model": "logreg", "feature_set": "F2", "cv_scheme": "random", "split": "cv",
                 "params": json.dumps({"C": 0.1, "fold": 1}), "pr_auc": 0.0})
    return pd.DataFrame(rows, columns=COLUMNS)


def test_chosen_settings_come_from_current_validation_rows():
    log = fake_log()
    assert leakage.chosen_settings(log) == BEST
    extra = log[(log.model == "logreg") & (log.split == "val")].head(1).assign(
        params=json.dumps({"C": 1.0, "threshold": 0.5}))
    with pytest.raises(ValueError, match="expected one set of validated settings"):
        leakage.chosen_settings(pd.concat([log, extra], ignore_index=True))
    with pytest.raises(ValueError, match="found 0"):
        leakage.chosen_settings(log[log.split != "val"])


def test_table_and_report():
    t = leakage.leakage_table(fake_log(), BEST).set_index(["model", "scheme"])
    assert t.loc[("logreg", "random"), "mean"] == pytest.approx(0.73)
    assert t.loc[("lgbm", "time"), "n"] == 4                # the older-setup time row is not counted
    assert t.loc[("lgbm", "time"), "folds"] == pytest.approx([0.51, 0.52, 0.53, 0.54])
    text = leakage.render(fake_log(), BEST)
    assert text.count("**random** scores highest") == 2
    assert "+0.2050 over time" in text            # 0.73 - 0.525
    assert "G3" in text and "F2" not in text


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
    assert leakage.chosen_settings(log) == BEST
    assert leakage.leakage_table(log, BEST).shape[0] == 6
    assert leakage.scheme_rows(log, "logreg", "random", {"C": 123}).empty
    # no candidate rows at all (a scheme never run yet) still returns rows with every column
    empty = leakage.scheme_rows(log, "logreg", "no_such_scheme", {"C": 0.1})
    assert empty.empty and "params" in empty.columns
    report = tmp_path / "leakage_check.md"
    assert leakage.write_report(path, report) == report.read_text()
