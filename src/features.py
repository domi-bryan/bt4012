"""Role B: customer ID (uid) and feature-set builders.

uid = card1 + "_" + addr1 + "_" + account_open_day, where account_open_day =
floor(TransactionDay - D1). It is approximate; `reports/uid_check.md` validates it.
"""
import numpy as np
import pandas as pd

from src.data import CLEANED_PATH, ROOT
from src.split import holdout_indices

UID_REPORT = ROOT / "reports" / "uid_check.md"
SIZE_BINS = [(1, 1), (2, 5), (6, 20), (21, 100), (101, 1000), (1001, None)]


def _part(series):
    """Integer string per row, "nan" when missing."""
    return np.floor(series.astype("float64")).astype("Int64").astype("string").fillna("nan")


def account_open_day(df):
    return np.floor(df["TransactionDay"].astype("float64") - df["D1"].astype("float64"))


def make_uid(df):
    """uid string per row; missing parts are written "nan"."""
    return (_part(df["card1"]) + "_" + _part(df["addr1"]) + "_"
            + _part(account_open_day(df))).astype(str)


def add_uid(df):
    """Return a copy of df with account_open_day (float, NaN when D1 missing) and uid (string)."""
    out = df.copy()
    out["account_open_day"] = account_open_day(df).astype("float32")
    out["uid"] = make_uid(df)
    return out


# ---------------------------------------------------------------------------
# Past-only aggregates (Step 8). For each row, only rows earlier in the time-sorted table
# with the same key are used, so no feature sees its own row or the future. Rows tied on
# TransactionDT count as earlier when they come first in the sort order (lower TransactionID).

AGG_KEYS = ("card1", "uid")
WINDOWS = {"1h": 3600, "24h": 86400, "7d": 7 * 86400}
DISTINCT_COLS = ("P_emaildomain", "addr1", "DeviceInfo")
_OFFSET = max(WINDOWS.values())   # keeps (TransactionDT - window) non-negative
_BIG = 2 ** 26                    # > max TransactionDT + _OFFSET, separates keys


def agg_columns(key):
    return ([f"{key}_prior_count", f"{key}_prior_amt_mean", f"{key}_prior_amt_std",
             f"{key}_amt_to_prior_mean", f"{key}_secs_since_prev"]
            + [f"{key}_count_{w}" for w in WINDOWS]
            + [f"{key}_n_{c}" for c in DISTINCT_COLS])


def past_aggregates(df, key, name):
    """Past-only aggregates of `df` grouped by the array `key`, as float64 columns named `name_*`.

    Counts are 0 for a key's first row; mean, ratio and seconds-since-previous are NaN until one
    prior row exists, std (ddof=1) until two do. Distinct counts ignore missing values.
    """
    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Table must be sorted by TransactionDT")
    codes, _ = pd.factorize(np.asarray(key))
    if (codes < 0).any():
        raise ValueError(f"{name}: key has missing values")
    n = len(df)
    order = np.argsort(codes, kind="stable")        # groups contiguous, time order kept inside
    c = codes[order]
    dt = df["TransactionDT"].to_numpy(dtype="int64")[order]
    amt = df["TransactionAmt"].to_numpy(dtype="float64")[order]
    pos = np.arange(n)
    start = np.r_[True, c[1:] != c[:-1]]
    first = np.maximum.accumulate(np.where(start, pos, 0))
    count = (pos - first).astype("float64")
    has1, has2 = count >= 1, count >= 2

    # amount mean/std from per-group cumulative sums, centred on the group's first amount
    xc = amt - amt[first]
    s1 = pd.Series(xc).groupby(c).cumsum().to_numpy() - xc
    s2 = pd.Series(xc * xc).groupby(c).cumsum().to_numpy() - xc * xc
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(has1, s1 / count + amt[first], np.nan)
        var = np.where(has2, (s2 - s1 * s1 / count) / (count - 1), np.nan)
    std = np.sqrt(np.clip(var, 0, None))

    prev_dt = np.r_[0, dt[:-1]]
    out = {
        f"{name}_prior_count": count,
        f"{name}_prior_amt_mean": mean,
        f"{name}_prior_amt_std": std,
        f"{name}_amt_to_prior_mean": amt / mean,
        f"{name}_secs_since_prev": np.where(has1, dt - prev_dt, np.nan),
    }
    combined = c.astype("int64") * _BIG + dt + _OFFSET
    for label, w in WINDOWS.items():
        left = np.searchsorted(combined, combined - w, side="right")
        out[f"{name}_count_{label}"] = (pos - left).astype("float64")
    for col in DISTINCT_COLS:
        vals = df[col].to_numpy()[order]
        present = ~pd.isna(vals)
        new = present & ~pd.DataFrame({"k": c, "v": vals}).duplicated().to_numpy()
        seen = pd.Series(new.astype("int64")).groupby(c).cumsum().to_numpy() - new
        out[f"{name}_n_{col}"] = seen.astype("float64")

    result = np.empty((n, len(out)))
    result[order] = np.column_stack(list(out.values()))
    return pd.DataFrame(result, columns=list(out), index=df.index)


def build_feature_set(df, feature_set):
    """F0 = df unchanged; F1 = F0 + card1 aggregates; F2 = F1 + uid aggregates (float32).

    The uid string itself is not added, so it never becomes a model input.
    """
    if feature_set not in ("F0", "F1", "F2"):
        raise ValueError("feature_set must be F0, F1 or F2")
    if feature_set == "F0":
        return df
    parts = [df, past_aggregates(df, df["card1"].to_numpy(), "card1")]
    if feature_set == "F2":
        parts.append(past_aggregates(df, make_uid(df).to_numpy(), "uid"))
    return pd.concat([parts[0]] + [p.astype("float32") for p in parts[1:]], axis=1)


# ---------------------------------------------------------------------------
# uid check (Step 7). Fraud statistics use the train block only, so validation and
# test labels stay unread; returning-customer counts use keys, not labels.

def group_stats(keys, fraud):
    """Distinct groups, size distribution and fraud concentration for one key."""
    g = pd.DataFrame({"key": keys, "fraud": fraud}).groupby("key", sort=False)["fraud"]
    size, n_fraud = g.size(), g.sum()
    has_fraud = n_fraud > 0
    out = {
        "rows": len(keys),
        "groups": len(size),
        "size_mean": size.mean(),
        "size_median": size.median(),
        "size_p90": size.quantile(0.9),
        "size_p99": size.quantile(0.99),
        "size_max": int(size.max()),
        "rows_in_single_groups_pct": 100 * size[size == 1].sum() / len(keys),
        "groups_with_fraud_pct": 100 * has_fraud.mean(),
        "fraud_rate_in_fraud_groups_pct": 100 * n_fraud[has_fraud].sum() / size[has_fraud].sum(),
        "fraud_rows_in_fraud_groups": int(n_fraud[has_fraud].sum()),
        "all_fraud_groups_pct_of_fraud_groups": 100 * (n_fraud[has_fraud] == size[has_fraud]).mean(),
    }
    for lo, hi in SIZE_BINS:
        mask = (size >= lo) & (size <= hi if hi else True)
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi else f">{lo - 1}")
        out[f"groups_size_{label}_pct"] = 100 * mask.mean()
        out[f"rows_size_{label}_pct"] = 100 * size[mask].sum() / len(keys)
    return out


def returning(keys, blocks):
    """Share of rows and distinct keys in `blocks['target']` whose key appears in `blocks['seen']`."""
    seen = set(keys[blocks["seen"]])
    target = keys[blocks["target"]]
    in_seen = np.fromiter((k in seen for k in target), bool, len(target))
    distinct = pd.unique(target)
    return {"rows_pct": 100 * in_seen.mean(),
            "keys_pct": 100 * np.mean([k in seen for k in distinct]),
            "rows": int(in_seen.sum()), "keys": int(sum(k in seen for k in distinct)),
            "target_rows": len(target), "target_keys": len(distinct)}


def uid_check(df):
    """All Step 7 statistics for card1 and uid, as nested dicts."""
    df = add_uid(df)
    tr, va, te = holdout_indices(df)
    keys = {"card1": df["card1"].astype("int64").astype(str).to_numpy(), "uid": df["uid"].to_numpy()}
    fraud = df["isFraud"].to_numpy()
    stats = {k: group_stats(v[tr], fraud[tr]) for k, v in keys.items()}
    ret = {}
    for k, v in keys.items():
        ret[k] = {
            "test_in_train": returning(v, {"seen": tr, "target": te}),
            "test_in_train_or_val": returning(v, {"seen": np.concatenate([tr, va]), "target": te}),
            "val_in_train": returning(v, {"seen": tr, "target": va}),
        }
    missing = {
        "D1": int(df["D1"].isna().sum()),
        "addr1": int(df["addr1"].isna().sum()),
        "D1_or_addr1": int((df["D1"].isna() | df["addr1"].isna()).sum()),
        "D1_and_addr1": int((df["D1"].isna() & df["addr1"].isna()).sum()),
        "D1_or_addr1_train": int((df["D1"].isna() | df["addr1"].isna()).to_numpy()[tr].sum()),
        "uids_with_nan_part": int(pd.Series(df["uid"]).str.contains("nan").sum()),
    }
    return {"stats": stats, "returning": ret, "missing": missing, "n_rows": len(df), "n_train": len(tr)}


def _fmt(x):
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    return f"{x:,.2f}"


def write_uid_report(result, path=UID_REPORT):
    s, r, m = result["stats"], result["returning"], result["missing"]
    lines = [
        "# uid check (Step 7)",
        "",
        "Generated by `.venv/bin/python -m src.features`. "
        "uid = card1 + \"_\" + addr1 + \"_\" + floor(TransactionDay - D1); missing parts are written \"nan\".",
        "",
        f"Group and fraud statistics use the **train block only** (first {result['n_train']:,} rows), "
        "so no validation or test labels are read. Returning-customer counts compare keys only.",
        "",
        "## Groups and fraud concentration (train block)",
        "",
        "| Statistic | card1 | uid |",
        "| --- | --- | --- |",
    ]
    names = {
        "rows": "Rows", "groups": "Distinct groups", "size_mean": "Group size: mean",
        "size_median": "Group size: median", "size_p90": "Group size: 90th percentile",
        "size_p99": "Group size: 99th percentile", "size_max": "Group size: max",
        "rows_in_single_groups_pct": "Rows in single-transaction groups (%)",
        "groups_with_fraud_pct": "Groups with any fraud (%)",
        "fraud_rate_in_fraud_groups_pct": "Fraud rate within groups that have any fraud (%)",
        "fraud_rows_in_fraud_groups": "Fraud rows (all fall in fraud groups)",
        "all_fraud_groups_pct_of_fraud_groups": "Fraud groups where every row is fraud (%)",
    }
    for key, label in names.items():
        lines.append(f"| {label} | {_fmt(s['card1'][key])} | {_fmt(s['uid'][key])} |")
    lines += ["", "## Group-size distribution (train block)", "",
              "| Group size | card1 groups (%) | card1 rows (%) | uid groups (%) | uid rows (%) |",
              "| --- | --- | --- | --- | --- |"]
    for lo, hi in SIZE_BINS:
        label = f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi else f">{lo - 1}")
        lines.append(f"| {label} | {_fmt(s['card1'][f'groups_size_{label}_pct'])} | "
                     f"{_fmt(s['card1'][f'rows_size_{label}_pct'])} | "
                     f"{_fmt(s['uid'][f'groups_size_{label}_pct'])} | "
                     f"{_fmt(s['uid'][f'rows_size_{label}_pct'])} |")
    lines += ["", "## Returning customers", "",
              "A key is returning if it already appears in the earlier block(s). Step 15 splits test "
              "PR-AUC by \"seen in train or validation\".", "",
              "| Target block | Seen in | card1 rows (%) | card1 keys (%) | uid rows (%) | uid keys (%) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for name, target, seen in [("test_in_train", "test", "train"),
                               ("test_in_train_or_val", "test", "train + validation"),
                               ("val_in_train", "validation", "train")]:
        c, u = r["card1"][name], r["uid"][name]
        lines.append(f"| {target} | {seen} | {_fmt(c['rows_pct'])} | {_fmt(c['keys_pct'])} "
                     f"| {_fmt(u['rows_pct'])} | {_fmt(u['keys_pct'])} |")
    lines += ["", "Counts behind the percentages:", ""]
    for k in ("card1", "uid"):
        for name, v in r[k].items():
            lines.append(f"- {k} {name}: {v['rows']:,} of {v['target_rows']:,} rows; "
                         f"{v['keys']:,} of {v['target_keys']:,} distinct keys")
    lines += ["", "## Rows with missing uid parts (all 590,540 rows)", "",
              "| Missing | Rows |", "| --- | --- |",
              f"| D1 | {m['D1']:,} |", f"| addr1 | {m['addr1']:,} |",
              f"| D1 or addr1 | {m['D1_or_addr1']:,} |", f"| D1 and addr1 | {m['D1_and_addr1']:,} |",
              f"| D1 or addr1, train block only | {m['D1_or_addr1_train']:,} |",
              f"| uid contains a \"nan\" part | {m['uids_with_nan_part']:,} |",
              "", "card1 is never missing. Rows with a \"nan\" part still get a uid, so customers "
              "with missing addr1 or D1 that share the other parts fall into one coarser group.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


if __name__ == "__main__":
    cols = ["TransactionDT", "TransactionDay", "isFraud", "card1", "addr1", "D1"]
    result = uid_check(pd.read_parquet(CLEANED_PATH, columns=cols))
    print(write_uid_report(result).read_text())
