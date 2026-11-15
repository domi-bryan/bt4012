"""Checks for src/feature_selection.py on small synthetic tables."""
import numpy as np
import pandas as pd

from src import feature_selection as fsel


def _frame(n=40_000, seed=0):
    rng = np.random.default_rng(seed)
    day = np.sort(rng.integers(1, 121, n))
    late = day > 60
    a = rng.integers(0, 2, n)                       # risky value 1 throughout
    b = rng.integers(0, 2, n)                       # risky value 1 early, value 0 late
    p = 0.02 + 0.08 * a + np.where(late, 0.08 * (1 - b), 0.08 * b)
    y = (rng.random(n) < p).astype(int)
    noise = rng.normal(size=n)
    return pd.DataFrame({"TransactionDay": day, "isFraud": y, "steady": a, "flips": b,
                         "noise": noise, "constant": 1.0})


def test_time_stability_categories():
    out = fsel.time_stability(_frame(), ["steady", "flips", "noise", "constant"])
    assert out.loc["steady", "result"] == "holds later"
    assert out.loc["flips", "result"] == "reversed later"
    assert out.loc["noise", "result"] == "weak in both windows"
    assert out.loc["constant", "result"] == "constant early"


def test_prune_v_drops_copies_within_missing_group():
    rng = np.random.default_rng(1)
    n = 500
    df = pd.DataFrame({f"V{i}": rng.normal(size=n) for i in range(1, 340)})
    df["V2"] = df["V1"] * 2 + 0.001 * rng.normal(size=n)     # near copy of V1, same missing pattern
    keep, drop = fsel.prune_v(df, threshold=0.75)
    assert len(keep) + len(drop) == 339
    assert ("V1" in drop) != ("V2" in drop)
    assert len(drop) == 1
