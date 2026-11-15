"""Past-only checks for src/domain_features.py.

1. Each helper matches a brute-force loop over earlier rows on a small synthetic table.
2. On the real table, features for the first rows do not change when later rows are added
   (if they did, a feature would be using future information).
"""
import numpy as np
import pandas as pd
import pytest

from src import domain_features as dfe
from src.data import JOINED_PATH

rng = np.random.default_rng(0)
N = 400
KEY = rng.integers(0, 12, N)
T = np.sort(rng.integers(0, 10 * dfe.DAY, N))
X = rng.gamma(2.0, 50.0, N)
X[rng.random(N) < 0.1] = np.nan
VAL = pd.Series(rng.choice(["a", "b", "c", None], N))


def _earlier(i):
    return [j for j in range(i) if KEY[j] == KEY[i]]


def test_prior_count_mean_max_prev():
    cnt, mean, mx, prev = (dfe.prior_count(KEY), dfe.prior_mean(KEY, X),
                           dfe.prior_max(KEY, X), dfe.prev_value(KEY, X))
    for i in range(N):
        e = _earlier(i)
        xs = [X[j] for j in e if not np.isnan(X[j])]
        assert cnt[i] == len(e)
        assert np.isclose(mean[i], np.mean(xs)) if xs else np.isnan(mean[i])
        assert np.isclose(mx[i], max(xs)) if xs else np.isnan(mx[i])
        assert (np.isnan(prev[i]) and (not e or np.isnan(X[e[-1]]))) or prev[i] == X[e[-1]]


def test_distinct_prior_and_new_value_flag():
    distinct, _ = dfe.distinct_prior(KEY, VAL)
    flag = dfe.new_value_flag(KEY, VAL)
    for i in range(N):
        seen = {VAL[j] for j in _earlier(i) if not pd.isna(VAL[j])}
        assert distinct[i] == len(seen)
        if pd.isna(VAL[i]) or not seen:
            assert np.isnan(flag[i])
        else:
            assert flag[i] == float(VAL[i] not in seen)


@pytest.mark.parametrize("window", [dfe.HOUR, dfe.DAY, 7 * dfe.DAY])
def test_window_prior(window):
    cnt, tot = dfe.window_prior(KEY, T, X, window)
    for i in range(N):
        e = [j for j in _earlier(i) if T[j] >= T[i] - window]
        assert cnt[i] == len(e)
        assert np.isclose(tot[i], np.nansum([X[j] for j in e]))


def test_frequency_features_amount_matches_brute_force():
    amt = np.round(rng.choice([1.0, 2.5, 2.499, 10.0, 49.99], N), 3)
    cols = {c: pd.Series(rng.choice(["x", "y", None], N)) for c in
            ["card1", "addr1", "P_emaildomain", "DeviceInfo", "id_31"]}
    df = pd.DataFrame({**cols, "TransactionAmt": amt})
    freq = dfe.frequency_features(df)["fe_freq_TransactionAmt"].to_numpy()
    cents = np.round(amt, 2)
    assert np.isnan(freq[0])
    for i in range(1, N):
        assert np.isclose(freq[i], np.sum(cents[:i] == cents[i]) / i)


def test_identity_os_missing():
    df = pd.DataFrame({"has_identity": [0, 0, 1, 1],
                       "id_30": [None, "Windows 10", None, "iOS 11.2.0"]})
    out = dfe.identity_os_missing(df)
    assert np.isnan(out[0]) and np.isnan(out[1])
    assert out[2] == 1.0 and out[3] == 0.0


def test_d_norm_includes_d11():
    df = pd.DataFrame({"TransactionDay": [10, 20], **{c: [1.0, np.nan] for c in dfe.D_NORM_COLS}})
    out = dfe.d_norm_features(df)
    assert list(out.columns) == ["fe_D1n", "fe_D4n", "fe_D10n", "fe_D11n", "fe_D15n"]
    assert out["fe_D11n"].iloc[0] == 9.0 and np.isnan(out["fe_D11n"].iloc[1])


# Candidate features left out because their link to fraud reversed, faded or never showed over time, or the
# EDA found no signal for them (notebooks/02_feature_engineering.ipynb section 10). They must not be built.
LEFT_OUT = ["fe_amt_round_dollar", "fe_amt_round_10", "fe_dow", "fe_uid_hour_concentration",
                 "fe_uid_new_os", "fe_uid_new_screen", "fe_os_family", "fe_screen_w", "fe_screen_h",
                 "fe_screen_area", "fe_screen_aspect", "fe_devicetype_os_mismatch",
                 "fe_uid_dist1_rel_prior_mean", "fe_freq_id_33",
                 "fe_card1_new_addr1", "fe_card1_n_prior_addr1", "fe_card1_n_prior_emails",
                 "fe_uid_amt_rel_prev", "fe_uid_amt_rel_prior_mean", "fe_uid_amt_rel_prior_max", "fe_uid_spend_24h"]
# Features backed by EDA 7.2, 9.15 and 6.1/11.1 that must be built.
FROM_EDA = ["fe_D11n", "fe_freq_TransactionAmt", "fe_identity_os_missing"]


def _s(series):
    return series.astype(object).fillna("<NA>").astype(str).to_numpy()


@pytest.mark.skipif(not JOINED_PATH.exists(), reason="joined.parquet not built")
def test_features_do_not_change_when_later_rows_are_added():
    df = pd.read_parquet(JOINED_PATH).iloc[:120_000]
    consts = dfe.fit_constants(df.iloc[:100_000])          # fixed constants for both runs
    short, long = 60_000, 120_000
    uid_s, fam_s = dfe.build_features(df.iloc[:short], consts)
    uid_l, fam_l = dfe.build_features(df, consts)
    names = [c for f in fam_l.values() for c in f.columns]
    assert len(names) == 55 and len(set(names)) == 55
    assert not set(LEFT_OUT) & set(names)
    assert set(FROM_EDA) <= set(names)
    assert (_s(uid_s) == _s(uid_l.iloc[:short])).all()
    for name in dfe.FAMILIES:
        a, b = fam_s[name], fam_l[name].iloc[:short]
        for c in a.columns:
            if isinstance(a[c].dtype, pd.CategoricalDtype):
                assert (_s(a[c]) == _s(b[c])).all(), c
            else:
                np.testing.assert_allclose(a[c].to_numpy(), b[c].to_numpy(), rtol=1e-5,
                                           equal_nan=True, err_msg=c)
