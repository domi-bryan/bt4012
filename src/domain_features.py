"""Domain-knowledge features for IEEE-CIS, built past-only on the time-sorted joined table.

Used by `notebooks/02_feature_engineering.ipynb`. Every behavioural feature for a row uses only rows
earlier in the sort order (TransactionDT, TransactionID). The only values fitted on data are a few
constants (most common addr2 / card3, V-column missing groups), and those come from the training
block (first 80% by time) via `fit_constants`.

All new columns are prefixed `fe_`. `uid` is added as an identifier for grouping, not as a feature.

Only features whose link to fraud holds between the first and last 30 days of the training block are
built here; candidates that reversed, faded, showed no signal or had no EDA support are listed in
`notebooks/02_feature_engineering.ipynb` section 10.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import ROOT

N_TRAIN = 472_432          # train block = rows 0..472,431 (Context.md: first 80% by time)
FEATURES_PATH = ROOT / "data" / "processed" / "features_v2.parquet"
FEATURE_LIST_PATH = ROOT / "config" / "features_v2.yaml"

DAY = 86_400
HOUR = 3_600
M_TF_COLS = ["M1", "M2", "M3", "M5", "M6", "M7", "M8", "M9"]   # M4 holds M0/M1/M2, not T/F
D_NORM_COLS = ["D1", "D4", "D10", "D11", "D15"]
ID_COLS = ["TransactionID", "TransactionDT", "TransactionDay", "uid"]
TARGET = "isFraud"


# --------------------------------------------------------------------------------------------
# Past-only helpers. `key` is an integer array (one code per group); rows must be in time order.
# --------------------------------------------------------------------------------------------

def combine(*arrays):
    """One int64 code per distinct combination of the inputs; NaN counts as its own value."""
    parts = {i: pd.factorize(pd.Series(a), use_na_sentinel=False)[0] for i, a in enumerate(arrays)}
    if len(parts) == 1:
        return parts[0].astype(np.int64)
    return pd.DataFrame(parts).groupby(list(parts), sort=False).ngroup().to_numpy(np.int64)


def prior_count(key):
    """Number of earlier rows with the same key."""
    return pd.Series(key).groupby(key, sort=False).cumcount().to_numpy(np.float64)


def prior_mean(key, x):
    """Mean of x over earlier rows with the same key (NaN x ignored; NaN if none)."""
    x = pd.Series(np.asarray(x, dtype=np.float64))
    filled = x.fillna(0.0)
    present = x.notna().astype(np.float64)
    s = (filled.groupby(key, sort=False).cumsum() - filled).to_numpy()
    n = (present.groupby(key, sort=False).cumsum() - present).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, s / n, np.nan)


def prior_max(key, x):
    """Max of x over earlier rows with the same key (NaN if none)."""
    x = pd.Series(np.asarray(x, dtype=np.float64)).fillna(-np.inf)
    cm = x.groupby(key, sort=False).cummax()
    prev = cm.groupby(key, sort=False).shift(1).to_numpy()
    return np.where(np.isfinite(prev), prev, np.nan)


def prev_value(key, x):
    """x on the previous row with the same key (NaN if none)."""
    return pd.Series(np.asarray(x, dtype=np.float64)).groupby(key, sort=False).shift(1).to_numpy()


def distinct_prior(key, val):
    """Number of distinct non-missing `val` values among earlier rows with the same key,
    and a flag for whether this row's (key, val) pair is a first occurrence."""
    val = pd.Series(val)
    valid = val.notna().to_numpy()
    pair = combine(key, val)
    first = (prior_count(pair) == 0) & valid
    firstf = pd.Series(first.astype(np.float64))
    distinct = (firstf.groupby(key, sort=False).cumsum() - firstf).to_numpy()
    return distinct, first


def new_value_flag(key, val):
    """1 if this key has used other values before but never this one, 0 if it has used this one
    before, NaN if the value is missing or the key has no earlier non-missing value."""
    distinct, first = distinct_prior(key, val)
    valid = pd.Series(val).notna().to_numpy()
    return np.where(valid & (distinct > 0), first.astype(np.float64), np.nan)


def window_prior(key, t, x, window):
    """Count and sum of x over earlier rows with the same key and t >= t_now - window.

    Rows sharing the same timestamp count as earlier only if they come earlier in the sort
    order. Vectorised with searchsorted on (key, t); no per-row Python loop.
    """
    key = np.asarray(key, dtype=np.int64)
    t = np.asarray(t, dtype=np.int64)
    x = np.nan_to_num(np.asarray(x, dtype=np.float64))
    n = len(key)
    order = np.lexsort((np.arange(n), key))          # by key, then original (time) order
    scale = np.int64(1 << 26)                          # > max TransactionDT + any window used
    comb = key[order] * scale + t[order]
    pos = np.arange(n)
    left = np.searchsorted(comb, comb - window, side="left")
    csum = np.concatenate([[0.0], np.cumsum(x[order])])
    count = np.empty(n)
    total = np.empty(n)
    count[order] = pos - left
    total[order] = csum[pos] - csum[left]
    return count, total


def _ratio(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.asarray(a, dtype=np.float64) / np.asarray(b, dtype=np.float64)
    out[~np.isfinite(out)] = np.nan
    return out


def _flag(cond, valid):
    return np.where(valid, np.asarray(cond, dtype=np.float64), np.nan)


def _int_str(x):
    return pd.Series(x).astype("Int64").astype("string").fillna("nan")


# --------------------------------------------------------------------------------------------
# Constants fitted on the training block only
# --------------------------------------------------------------------------------------------

def fit_constants(train):
    """Values learned from the training block. Nothing here sees validation or test rows."""
    v_cols = [f"V{i}" for i in range(1, 340)]
    counts = train[v_cols].isna().sum()
    groups = {}
    for c in v_cols:
        groups.setdefault(int(counts[c]), []).append(c)
    reps = [cols[0] for n_miss, cols in sorted(groups.items()) if n_miss > 0]
    return {
        "addr2_mode": float(train["addr2"].mode().iloc[0]),
        "card3_mode": float(train["card3"].mode().iloc[0]),
        "v_group_representatives": reps,
        "n_train_rows": int(len(train)),
    }


# --------------------------------------------------------------------------------------------
# Feature families
# --------------------------------------------------------------------------------------------

def make_uid(df):
    """uid = card1 _ addr1 _ floor(TransactionDay - D1), as in Context.md."""
    open_day = np.floor(df["TransactionDay"].to_numpy(np.float64) - df["D1"].to_numpy(np.float64))
    uid = (_int_str(df["card1"].to_numpy()) + "_" + _int_str(df["addr1"].to_numpy()) + "_"
           + _int_str(open_day))
    return uid.astype("category").set_axis(df.index)


def amount_features(df):
    amt = df["TransactionAmt"].to_numpy(np.float64)
    a3 = np.round(amt, 3)
    decimals = np.full(len(amt), 3.0)
    for d in (2, 1, 0):
        decimals[np.isclose(a3, np.round(a3, d), atol=1e-6)] = d
    return pd.DataFrame({
        "fe_amt_log": np.log1p(amt),
        "fe_amt_cents": np.round(amt * 100) % 100,
        "fe_amt_decimals": decimals,
        "fe_amt_lt5": (amt < 5).astype(np.float64),
    }, index=df.index)


def time_features(df, uid_key):
    dt = df["TransactionDT"].to_numpy(np.int64)
    hour = ((dt % DAY) // HOUR).astype(np.float64)
    ang = 2 * np.pi * hour / 24
    ms = prior_mean(uid_key, np.sin(ang))
    mc = prior_mean(uid_key, np.cos(ang))
    mean_hour = (np.arctan2(ms, mc) * 24 / (2 * np.pi)) % 24
    dev = np.abs(hour - mean_hour)
    dev = np.minimum(dev, 24 - dev)
    return pd.DataFrame({
        "fe_hour": hour,
        "fe_uid_hour_dev": dev,
    }, index=df.index)


def d_norm_features(df):
    day = df["TransactionDay"].to_numpy(np.float64)
    return pd.DataFrame({f"fe_{c}n": day - df[c].to_numpy(np.float64) for c in D_NORM_COLS},
                        index=df.index)


def novelty_features(df, uid_key):
    out = {}
    for col, name in [("DeviceInfo", "device"), ("id_31", "browser"), ("P_emaildomain", "email"),
                      ("ProductCD", "product")]:
        out[f"fe_uid_new_{name}"] = new_value_flag(uid_key, df[col])
    for col, name in [("DeviceInfo", "device"), ("P_emaildomain", "email"), ("id_31", "browser")]:
        out[f"fe_uid_n_prior_{name}s"] = distinct_prior(uid_key, df[col])[0]

    # Device fingerprint: device string + OS + browser + screen; only where a device is recorded.
    has_dev = df["DeviceInfo"].notna().to_numpy()
    fp = combine(df["DeviceInfo"], df["id_30"], df["id_31"], df["id_33"])
    n_cards = distinct_prior(fp, df["card1"])[0]
    n_uids = distinct_prior(fp, pd.Series(uid_key))[0]
    out["fe_fp_n_prior_cards"] = np.where(has_dev, n_cards, np.nan)
    out["fe_fp_n_prior_uids"] = np.where(has_dev, n_uids, np.nan)
    out["fe_fp_prior_count"] = np.where(has_dev, prior_count(fp), np.nan)
    return pd.DataFrame(out, index=df.index)


def velocity_features(df, uid_key):
    amt = df["TransactionAmt"].to_numpy(np.float64)
    dt = df["TransactionDT"].to_numpy(np.int64)
    day = df["TransactionDay"].to_numpy(np.float64)
    out = {}
    out["fe_uid_prior_count"] = prior_count(uid_key)
    prev_amt = prev_value(uid_key, amt)
    prev_t = prev_value(uid_key, dt)
    out["fe_uid_secs_since_prev"] = dt - prev_t
    has_prev = ~np.isnan(prev_amt)
    out["fe_uid_small_then_large"] = _flag((prev_amt < 5) & (amt >= 10 * prev_amt), has_prev)
    out["fe_uid_repeat_amt_count"] = prior_count(combine(uid_key, np.round(amt, 2)))
    for label, w in [("1h", HOUR), ("24h", DAY), ("7d", 7 * DAY)]:
        cnt, tot = window_prior(uid_key, dt, amt, w)
        out[f"fe_uid_count_{label}"] = cnt
        if label == "24h":
            spent = (pd.Series(amt).groupby(uid_key, sort=False).cumsum().to_numpy() - amt)
            first_day = pd.Series(day).groupby(uid_key, sort=False).transform("min").to_numpy()
            daily_avg = spent / (day - first_day + 1)
            out["fe_uid_spend_24h_rel_daily"] = np.where(
                out["fe_uid_prior_count"] > 0, _ratio(tot + amt, daily_avg), np.nan)
    return pd.DataFrame(out, index=df.index)


EMAIL_PROVIDERS = {
    "google": ["gmail"], "yahoo": ["yahoo", "ymail", "rocketmail"],
    "microsoft": ["hotmail", "outlook", "live", "msn"], "apple": ["icloud", "me.com", "mac.com"],
    "aol": ["aol", "aim.com"], "anonymous": ["anonymous", "protonmail"],
}


def _email_provider(domain):
    if pd.isna(domain):
        return np.nan
    d = str(domain).lower()
    for provider, keys in EMAIL_PROVIDERS.items():
        if any(d.startswith(k) or d == k for k in keys):
            return provider
    return "other"


def _map_categories(s, fn):
    """Apply fn to each category once, then map rows (fast on 590k rows)."""
    s = s.astype("category")
    mapping = {c: fn(c) for c in s.cat.categories}
    return s.astype(object).map(mapping)


def _browser_parts(v):
    v = str(v).lower()
    m = re.search(r"\d+(?:\.\d+)?", v)
    name = (v[:m.start()] if m else v.replace("generic", "")).strip() or "other"
    version = float(m.group()) if m else np.nan
    return name, version


DEVICE_BRAND_RULES = [
    ("windows", ("windows",)), ("apple", ("ios device", "macos")),
    ("samsung", ("sm-", "samsung", "gt-", "sgh-")), ("motorola", ("moto", "xt")),
    ("lg", ("lg",)), ("huawei", ("huawei", "ale-", "pra-", "was-", "hi6210", "cam-", "vns-")),
    ("browser_token", ("trident", "rv:")), ("htc", ("htc",)), ("lenovo", ("lenovo",)),
    ("google", ("pixel", "nexus")), ("xiaomi", ("redmi", "mi ")), ("zte", ("zte", "blade")),
    ("linux", ("linux",)),
]


def _device_brand(v):
    v = str(v).lower()
    for brand, prefixes in DEVICE_BRAND_RULES:
        if any(v.startswith(p) or (len(p) > 3 and p in v) for p in prefixes):
            return brand
    return "other"


def identity_os_missing(df):
    """1 if the row has an identity record but no OS (id_30), 0 if it has both, NaN without an
    identity record. OS, screen and related identity fields go missing together (EDA 6.1)."""
    has_id = df["has_identity"].to_numpy() == 1
    return _flag(df["id_30"].isna().to_numpy(), has_id)


def identity_features(df):
    out = {}
    p, r = df["P_emaildomain"], df["R_emaildomain"]
    out["fe_P_email_provider"] = _map_categories(p, _email_provider)
    out["fe_R_email_provider"] = _map_categories(r, _email_provider)
    out["fe_P_email_tld"] = _map_categories(p, lambda d: str(d).rsplit(".", 1)[-1])
    out["fe_email_anonymous"] = _flag(out["fe_P_email_provider"] == "anonymous", p.notna())
    both = (p.notna() & r.notna()).to_numpy()
    out["fe_email_match"] = _flag(p.astype(object).to_numpy() == r.astype(object).to_numpy(), both)

    b = df["id_31"]
    parts = _map_categories(b, _browser_parts)
    has_b = b.notna().to_numpy()
    name = parts.map(lambda t: t[0] if isinstance(t, tuple) else np.nan)
    version = parts.map(lambda t: t[1] if isinstance(t, tuple) else np.nan).astype(np.float64)
    out["fe_browser_name"] = name
    out["fe_browser_generic"] = _flag(b.astype(str).str.contains("generic").to_numpy(), has_b)
    # Versions behind the newest version of that browser seen so far (this row included).
    bkey = combine(name)
    newest = pd.Series(version.fillna(-np.inf).to_numpy()).groupby(bkey, sort=False).cummax()
    newest = newest.where(np.isfinite(newest)).to_numpy()
    out["fe_browser_versions_behind"] = newest - version.to_numpy()

    out["fe_device_brand"] = _map_categories(df["DeviceInfo"], _device_brand)
    out["fe_identity_os_missing"] = identity_os_missing(df)
    return pd.DataFrame(out, index=df.index)


def geo_card_features(df, consts):
    addr2, card3 = df["addr2"], df["card3"]
    combo = df["card4"].astype(object).fillna("nan") + "_" + df["card6"].astype(object).fillna("nan")
    return pd.DataFrame({
        "fe_addr2_not_mode": _flag(addr2.to_numpy() != consts["addr2_mode"], addr2.notna().to_numpy()),
        "fe_card3_not_mode": _flag(card3.to_numpy() != consts["card3_mode"], card3.notna().to_numpy()),
        "fe_card4_card6": combo.to_numpy(),
    }, index=df.index)


def missing_features(df, uid_key, consts, raw_cols):
    reps = consts["v_group_representatives"]
    bits = np.zeros(len(df), dtype=np.int64)
    for i, c in enumerate(reps):
        bits |= df[c].isna().to_numpy().astype(np.int64) << i
    m = df[M_TF_COLS].astype(object)
    out = {
        "fe_n_missing": df[raw_cols].isna().sum(axis=1).to_numpy(np.float64),
        "fe_v_missing_pattern": bits.astype(str),
        "fe_m_true": (m == "T").sum(axis=1).to_numpy(np.float64),
        "fe_m_false": (m == "F").sum(axis=1).to_numpy(np.float64),
        "fe_m_missing": m.isna().sum(axis=1).to_numpy(np.float64),
        "fe_uid_c1_rel_prior_mean": _ratio(df["C1"], prior_mean(uid_key, df["C1"])),
        "fe_uid_c13_rel_prior_mean": _ratio(df["C13"], prior_mean(uid_key, df["C13"])),
    }
    return pd.DataFrame(out, index=df.index)


def frequency_features(df):
    """Share of all earlier rows that have the same value (NaN for a missing value or the first row).
    The amount is compared rounded to cents."""
    position = np.arange(len(df), dtype=np.float64)
    out = {}
    for col in ["card1", "addr1", "P_emaildomain", "DeviceInfo", "id_31", "TransactionAmt"]:
        vals = df[col]
        if col == "TransactionAmt":
            vals = pd.Series(np.round(vals.to_numpy(np.float64), 2), index=df.index)
        valid = vals.notna().to_numpy()
        share = _ratio(prior_count(combine(vals)), position)
        out[f"fe_freq_{col}"] = np.where(valid, share, np.nan)
    return pd.DataFrame(out, index=df.index)


FAMILIES = ["amount", "time", "d_norm", "novelty", "velocity", "identity", "geo_card",
            "missing", "frequency"]


def build_features(df, consts):
    """Return (uid Series, {family: DataFrame of fe_ columns}). `df` must be time-sorted."""
    raw_cols = [c for c in df.columns if c not in ID_COLS and c != TARGET]
    uid = make_uid(df)
    uid_key = uid.cat.codes.to_numpy(np.int64)
    fams = {
        "amount": amount_features(df),
        "time": time_features(df, uid_key),
        "d_norm": d_norm_features(df),
        "novelty": novelty_features(df, uid_key),
        "velocity": velocity_features(df, uid_key),
        "identity": identity_features(df),
        "geo_card": geo_card_features(df, consts),
        "missing": missing_features(df, uid_key, consts, raw_cols),
        "frequency": frequency_features(df),
    }
    for f in fams.values():
        for c in f.columns:
            if pd.api.types.is_numeric_dtype(f[c]) or pd.api.types.is_bool_dtype(f[c]):
                f[c] = f[c].astype(np.float32)
            else:
                f[c] = f[c].astype("category")
    return uid, fams


def assemble(df, uid, fams):
    """Raw joined columns + uid + all fe_ columns, row order unchanged."""
    return pd.concat([df, uid.rename("uid")] + list(fams.values()), axis=1)
