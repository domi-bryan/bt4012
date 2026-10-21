import numpy as np
import pandas as pd
import pytest

from src import features


@pytest.fixture
def toy():
    return pd.DataFrame({
        "TransactionDay": np.array([10, 12, 12, 20, 21], dtype="int16"),
        "D1": np.array([5, 7, np.nan, 15, 0], dtype="float32"),
        "card1": np.array([1000, 1000, 1000, 2000, 2000], dtype="float32"),
        "addr1": np.array([300, 300, 300, np.nan, 315], dtype="float32"),
    })


def test_uid_parts_and_missing(toy):
    out = features.add_uid(toy)
    assert out["account_open_day"].tolist()[:2] == [5.0, 5.0]
    assert np.isnan(out["account_open_day"].iloc[2])
    assert out["uid"].tolist() == ["1000_300_5", "1000_300_5", "1000_300_nan",
                                   "2000_nan_5", "2000_315_21"]
    assert pd.api.types.is_string_dtype(out["uid"])
    # input is not modified
    assert "uid" not in toy


def test_account_open_day_floors():
    df = pd.DataFrame({"TransactionDay": [10, 10], "D1": [2.5, -0.5]})
    assert features.account_open_day(df).tolist() == [7.0, 10.0]


def test_group_stats_hand_case():
    keys = np.array(["a", "a", "a", "b", "c", "c"])
    fraud = np.array([1, 0, 0, 0, 1, 1])
    s = features.group_stats(keys, fraud)
    assert s["groups"] == 3 and s["size_max"] == 3
    assert s["rows_in_single_groups_pct"] == pytest.approx(100 / 6)
    assert s["groups_with_fraud_pct"] == pytest.approx(200 / 3)
    assert s["fraud_rate_in_fraud_groups_pct"] == pytest.approx(60.0)   # 3 fraud of 5 rows
    assert s["all_fraud_groups_pct_of_fraud_groups"] == pytest.approx(50.0)
    assert s["groups_size_1_pct"] + s["groups_size_2-5_pct"] == pytest.approx(100)


def test_returning():
    keys = np.array(["a", "b", "a", "c", "c", "d"])
    r = features.returning(keys, {"seen": np.array([0, 1]), "target": np.array([2, 3, 4, 5])})
    assert (r["rows"], r["target_rows"], r["keys"], r["target_keys"]) == (1, 4, 1, 3)
    assert r["rows_pct"] == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Step 8: past-only aggregates, checked against a brute-force recomputation

from src.data import CLEANED_PATH  # noqa: E402

AGG_INPUTS = ["TransactionDT", "TransactionDay", "TransactionAmt", "card1", "addr1", "D1",
              "P_emaildomain", "DeviceInfo"]


def brute_force(df, key, row, name):
    """Recompute every aggregate for one row from df.iloc[:row] only."""
    prior = df.iloc[:row][key[:row] == key[row]]
    amt, t = float(df["TransactionAmt"].iloc[row]), int(df["TransactionDT"].iloc[row])
    a = prior["TransactionAmt"].astype("float64")
    n = len(prior)
    mean = a.mean() if n else np.nan
    out = {
        f"{name}_prior_count": n,
        f"{name}_prior_amt_mean": mean,
        f"{name}_prior_amt_std": a.std(ddof=1) if n > 1 else np.nan,
        f"{name}_amt_to_prior_mean": amt / mean if n else np.nan,
        f"{name}_secs_since_prev": t - int(prior["TransactionDT"].iloc[-1]) if n else np.nan,
    }
    for label, w in features.WINDOWS.items():
        out[f"{name}_count_{label}"] = int((prior["TransactionDT"] > t - w).sum())
    for col in features.DISTINCT_COLS:
        out[f"{name}_n_{col}"] = prior[col].nunique(dropna=True)
    return out


def assert_matches(df, key, name, rows):
    agg = features.past_aggregates(df, key, name)
    assert list(agg.columns) == features.agg_columns(name)
    for row in rows:
        expected = brute_force(df, key, row, name)
        got = agg.iloc[row]
        for col, val in expected.items():
            if np.isnan(val):
                assert np.isnan(got[col]), (row, col, got[col])
            else:
                assert abs(got[col] - val) <= 1e-5 * max(1.0, abs(val)), (row, col, got[col], val)


@pytest.fixture
def toy_tx():
    """400 rows, 12 cards, tied timestamps, repeated amounts and missing distinct-count values."""
    rng = np.random.default_rng(0)
    n = 400
    dt = np.sort(rng.integers(86400, 86400 * 40, size=n))
    dt[50:56] = dt[50]
    amt = rng.choice([10.0, 25.5, 100.0, 1234.5], size=n).astype("float32")
    email = rng.choice(["a.com", "b.com", "c.com", None], size=n)
    device = rng.choice(["x", "y", None], size=n)
    return pd.DataFrame({
        "TransactionDT": dt.astype("int32"), "TransactionDay": (dt // 86400).astype("int16"),
        "TransactionAmt": amt, "card1": rng.integers(1000, 1012, size=n).astype("float32"),
        "addr1": rng.choice([100.0, 200.0, np.nan], size=n).astype("float32"),
        "D1": rng.integers(0, 5, size=n).astype("float32"),
        "P_emaildomain": pd.Categorical(email), "DeviceInfo": pd.Categorical(device),
    })


def test_aggregates_match_brute_force_toy(toy_tx):
    assert_matches(toy_tx, toy_tx["card1"].to_numpy(), "card1", range(len(toy_tx)))
    uid = features.make_uid(toy_tx).to_numpy()
    assert_matches(toy_tx, uid, "uid", range(len(toy_tx)))


def test_first_transaction_and_ties(toy_tx):
    agg = features.past_aggregates(toy_tx, toy_tx["card1"].to_numpy(), "card1")
    first = ~toy_tx["card1"].duplicated()
    assert (agg.loc[first, "card1_prior_count"] == 0).all()
    assert agg.loc[first, "card1_prior_amt_mean"].isna().all()
    assert (agg.loc[first, [f"card1_count_{w}" for w in features.WINDOWS]] == 0).all().all()
    # a row never counts itself: prior count equals the number of earlier same-key rows
    assert np.array_equal(agg["card1_prior_count"], toy_tx.groupby("card1").cumcount())


def test_constant_amounts_give_zero_std():
    df = pd.DataFrame({"TransactionDT": np.arange(5) * 10 + 86400, "TransactionAmt": [1e5] * 5,
                       "P_emaildomain": None, "addr1": np.nan, "DeviceInfo": None})
    agg = features.past_aggregates(df, np.zeros(5), "k")
    assert (agg["k_prior_amt_std"].iloc[2:] == 0).all()


def test_build_feature_sets(toy_tx):
    f0 = features.build_feature_set(toy_tx, "F0")
    f1 = features.build_feature_set(toy_tx, "F1")
    f2 = features.build_feature_set(toy_tx, "F2")
    assert f0 is toy_tx
    assert list(f1.columns) == list(toy_tx.columns) + features.agg_columns("card1")
    assert list(f2.columns) == list(f1.columns) + features.agg_columns("uid")
    assert "uid" not in f2
    assert (f2[features.agg_columns("uid")].dtypes == "float32").all()


def test_feature_dictionary_lists_every_column():
    text = (features.ROOT / "reports" / "feature_dictionary.md").read_text()
    for key in features.AGG_KEYS:
        for col in features.agg_columns(key):
            assert f"`{col}`" in text, col


@pytest.fixture(scope="module")
def real_tx():
    if not CLEANED_PATH.exists():
        pytest.skip("cleaned.parquet not built")
    return pd.read_parquet(CLEANED_PATH, columns=AGG_INPUTS)


@pytest.mark.parametrize("name", features.AGG_KEYS)
def test_aggregates_match_brute_force_real(real_tx, name):
    key = (real_tx["card1"].to_numpy() if name == "card1"
           else features.make_uid(real_tx).to_numpy())
    rows = np.random.default_rng(0).choice(len(real_tx), size=200, replace=False)
    assert_matches(real_tx, key, name, rows)


@pytest.mark.parametrize("name", features.AGG_KEYS)
def test_real_first_transactions(real_tx, name):
    key = (real_tx["card1"].to_numpy() if name == "card1"
           else features.make_uid(real_tx).to_numpy())
    agg = features.past_aggregates(real_tx, key, name)
    first = ~pd.Series(key).duplicated().to_numpy()
    assert (agg.loc[first, f"{name}_prior_count"] == 0).all()
    assert agg.loc[first, f"{name}_prior_amt_mean"].isna().all()
    assert agg.loc[~first, f"{name}_prior_amt_mean"].notna().all()
