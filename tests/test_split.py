import numpy as np
import pandas as pd
import pytest

from src import split
from src.data import CLEANED_PATH


@pytest.fixture
def toy():
    """1,000 rows over days 1-130, sorted by time, with ties on TransactionDT."""
    rng = np.random.default_rng(0)
    dt = np.sort(rng.integers(86400, 131 * 86400, size=1000))
    dt[100:110] = dt[100]
    return pd.DataFrame({
        "TransactionDT": dt,
        "TransactionDay": dt // 86400,
        "isFraud": (rng.random(1000) < 0.1).astype("int8"),
        "group": rng.integers(0, 60, size=1000),
    })


def check_holdout(df):
    tr, va, te = split.holdout_indices(df)
    allidx = np.concatenate([tr, va, te])
    assert len(allidx) == len(df) and len(np.unique(allidx)) == len(df)
    assert set(tr).isdisjoint(va) and set(tr).isdisjoint(te) and set(va).isdisjoint(te)
    dt = df["TransactionDT"].to_numpy()
    assert dt[tr].max() <= dt[va].min() <= dt[te].min()
    assert dt[va].max() <= dt[te].min()
    return tr, va, te


def check_time_folds(df, tr):
    folds = split.time_cv_folds(df, tr)
    dt = df["TransactionDT"].to_numpy()
    seen_val = []
    for a, b in folds:
        assert len(a) and len(b)
        assert dt[a].max() < dt[b].min()
        assert set(a).isdisjoint(b)
        seen_val.append(b)
    # expanding window: each fold's train is the previous fold's train plus its val
    for (a0, b0), (a1, _) in zip(folds, folds[1:]):
        assert np.array_equal(np.sort(np.concatenate([a0, b0])), np.sort(a1))
    # the last fold's train plus val is the whole train block
    a, b = folds[-1]
    assert np.array_equal(np.sort(np.concatenate([a, b])), np.sort(tr))
    return folds


def test_holdout_toy(toy):
    tr, va, te = check_holdout(toy)
    assert (len(tr), len(va), len(te)) == (800, 100, 100)


def test_holdout_rejects_unsorted(toy):
    with pytest.raises(ValueError, match="sorted"):
        split.holdout_indices(toy.iloc[::-1])


def test_time_folds_toy(toy):
    tr, _, _ = split.holdout_indices(toy)
    folds = check_time_folds(toy, tr)
    day = toy["TransactionDay"].to_numpy()
    first = day[tr].min()
    for k, (a, b) in enumerate(folds, start=1):
        assert set((day[b] - first) // 30) == {k}
        assert ((day[a] - first) // 30 < k).all()


def test_random_folds_toy(toy):
    tr, _, _ = split.holdout_indices(toy)
    y = toy["isFraud"].to_numpy()[tr]
    folds = split.random_cv_folds(tr, y, k=5, seed=0)
    assert len(folds) == 5
    vals = np.concatenate([b for _, b in folds])
    assert np.array_equal(np.sort(vals), tr)
    for a, b in folds:
        assert set(a).isdisjoint(b)
    again = split.random_cv_folds(tr, y, k=5, seed=0)
    assert all(np.array_equal(b0, b1) for (_, b0), (_, b1) in zip(folds, again))


def test_group_folds_never_share_a_group(toy):
    tr, _, _ = split.holdout_indices(toy)
    y = toy["isFraud"].to_numpy()[tr]
    g = toy["group"].to_numpy()[tr]
    folds = split.group_cv_folds(tr, y, g, k=5, seed=0)
    groups = toy["group"].to_numpy()
    vals = np.concatenate([b for _, b in folds])
    assert np.array_equal(np.sort(vals), tr)
    for a, b in folds:
        assert set(groups[a]).isdisjoint(groups[b])


def test_misaligned_labels_rejected(toy):
    tr, _, _ = split.holdout_indices(toy)
    with pytest.raises(ValueError, match="aligned"):
        split.random_cv_folds(tr, toy["isFraud"].to_numpy(), seed=0)


@pytest.fixture(scope="module")
def real():
    if not CLEANED_PATH.exists():
        pytest.skip("cleaned.parquet not built")
    cols = ["TransactionDT", "TransactionDay", "isFraud", "card1"]
    return pd.read_parquet(CLEANED_PATH, columns=cols)


def test_real_holdout_and_time_folds(real):
    tr, va, te = check_holdout(real)
    assert (len(tr), len(va), len(te)) == (472_432, 59_054, 59_054)
    assert tr[-1] == 472_431
    folds = check_time_folds(real, tr)
    assert len(folds) == 4


def test_real_group_folds(real):
    # uid is not built until Step 7; card1 stands in as the group key here
    tr, _, _ = split.holdout_indices(real)
    y = real["isFraud"].to_numpy()[tr]
    card1 = real["card1"].to_numpy()
    folds = split.group_cv_folds(tr, y, card1[tr], k=5, seed=0)
    for a, b in folds:
        assert np.intersect1d(card1[a], card1[b]).size == 0
