"""Role A: load, join and save the time-sorted table; tree and linear preprocessing pipelines."""
import warnings
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
JOINED_PATH = ROOT / "data" / "processed" / "joined.parquet"
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned.parquet"

CATEGORICAL_COLS = (
    ["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
    + [f"M{i}" for i in range(1, 10)]
    # string-valued identity columns among id_12-id_38
    + [f"id_{i}" for i in (12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38)]
    + ["DeviceType", "DeviceInfo"]
)
INT_COLS = {"TransactionID": "int32", "isFraud": "int8", "TransactionDT": "int32"}


def _read(path, id_cols=()):
    """Read a CSV with categoricals as `category` and every other column float32."""
    header = pd.read_csv(path, nrows=0).columns
    dtype = {}
    for col in header:
        if col in INT_COLS:
            dtype[col] = INT_COLS[col]
        elif col in CATEGORICAL_COLS:
            dtype[col] = "category"
        else:
            dtype[col] = "float32"
    return pd.read_csv(path, dtype=dtype)


def load_raw(raw_dir=RAW_DIR):
    """Return transaction left-joined with identity, sorted by (TransactionDT, TransactionID)."""
    raw_dir = Path(raw_dir)
    trans = _read(raw_dir / "train_transaction.csv")
    ident = _read(raw_dir / "train_identity.csv")
    ident_ids = set(ident["TransactionID"])

    df = trans.merge(ident, on="TransactionID", how="left", validate="one_to_one").copy()
    df["TransactionDay"] = (df["TransactionDT"] // 86400).astype("int16")
    df["has_identity"] = df["TransactionID"].isin(ident_ids).astype("int8")

    df = df.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
    return df


def save_joined(df, path=JOINED_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_joined(path=JOINED_PATH):
    return pd.read_parquet(path)


def validate_cleaned(df):
    """Reject malformed data without guessing repairs or fitting any statistics.

    Missing predictors, repeated predictors with distinct IDs, negative anonymized
    features, and extreme positive amounts are intentionally allowed.
    """
    required = {*INT_COLS, "TransactionAmt", "TransactionDay", "has_identity"}
    if df.empty or not df.columns.is_unique or not required.issubset(df.columns):
        raise ValueError("Expected nonempty data with unique columns and all required fields")
    expected_ints = {**INT_COLS, "TransactionDay": "int16", "has_identity": "int8"}
    for col in df:
        expected = expected_ints.get(col, "category" if col in CATEGORICAL_COLS else "float32")
        if str(df[col].dtype) != expected:
            raise ValueError(f"{col}: expected dtype {expected}, got {df[col].dtype}")
        if expected == "category":
            values = df[col].dropna().unique()
            if any(not isinstance(v, str) or not v.strip() or v != v.strip() for v in values):
                raise ValueError(f"{col}: blank, padded, or non-string category")
        elif np.isinf(df[col].to_numpy()).any():
            raise ValueError(f"{col}: infinite values")
    if not df.TransactionID.is_unique or df.TransactionID.isna().any():
        raise ValueError("TransactionID must be unique and nonmissing")
    if not df.isFraud.isin([0, 1]).all() or not df.has_identity.isin([0, 1]).all():
        raise ValueError("isFraud and has_identity must be binary and nonmissing")
    if df.TransactionAmt.isna().any() or (df.TransactionAmt <= 0).any():
        raise ValueError("TransactionAmt must be finite, nonmissing, and positive")
    if (df.TransactionDT < 0).any():
        raise ValueError("TransactionDT must be nonnegative")
    order = pd.MultiIndex.from_frame(df[["TransactionDT", "TransactionID"]])
    if not order.is_monotonic_increasing:
        raise ValueError("Rows must be ordered by TransactionDT then TransactionID")
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError("Expected consecutive row positions starting at zero")
    if not (df.TransactionDay == df.TransactionDT // 86400).all():
        raise ValueError("TransactionDay does not match TransactionDT")
    identity_cols = [c for c in df if c.startswith("id_") or c in ("DeviceType", "DeviceInfo")]
    if not df.loc[df.has_identity == 0, identity_cols].isna().all().all():
        raise ValueError("Rows without identity matches contain identity values")


def save_cleaned(df, path=CLEANED_PATH):
    """Save a validated, unchanged table after verifying an exact parquet roundtrip."""
    path = Path(path).resolve()
    if path == JOINED_PATH.resolve() or path.is_relative_to(RAW_DIR.resolve()):
        raise ValueError("The cleaned output must not overwrite joined or raw data")
    validate_cleaned(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        df.to_parquet(temporary, index=False)
        pd.testing.assert_frame_equal(df, pd.read_parquet(temporary), check_exact=True)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_cleaned(path=CLEANED_PATH):
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Preprocessing pipelines (Step 5). Both are fit on training rows only.

FEATURES_YAML = ROOT / "config" / "features.yaml"
# IDs, label and raw time are never model inputs; time is the split axis, not a feature.
NON_FEATURES = ["TransactionID", "isFraud", "TransactionDT", "TransactionDay"]
SENTINEL = -999.0
MAX_ONEHOT_LEVELS = 20


def load_v_columns(path=FEATURES_YAML):
    """Return (keep, drop) V-column lists. Warns while the list is not group-approved."""
    with open(path) as handle:
        section = yaml.safe_load(handle)["v_columns_proposed"]
    if not str(section.get("status", "")).startswith("approved"):
        warnings.warn(f"V-column list is not approved yet (status: {section.get('status')!r})")
    return list(section["keep"]), list(section["drop"])


def feature_columns(df, v_drop=None):
    """Model input columns in table order: everything except NON_FEATURES and pruned V columns."""
    if v_drop is None:
        v_drop = load_v_columns()[1]
    excluded = set(NON_FEATURES) | set(v_drop)
    return [c for c in df.columns if c not in excluded]


def _is_cat(series):
    return isinstance(series.dtype, pd.CategoricalDtype)


class TreePipeline:
    """Tree input. kind="lgbm" keeps NaN and `category` dtypes; kind="rf" fills NaN with -999
    and turns categoricals into integer codes. Category levels are the ones seen in training
    rows; levels first seen later become missing."""

    def __init__(self, kind="lgbm", v_drop=None):
        if kind not in ("lgbm", "rf"):
            raise ValueError("kind must be 'lgbm' or 'rf'")
        self.kind = kind
        self.v_drop = v_drop

    def fit(self, train_df):
        self.columns_ = feature_columns(train_df, self.v_drop)
        self.categories_ = {c: train_df[c].dropna().unique().astype(str).tolist()
                            for c in self.columns_ if _is_cat(train_df[c])}
        for levels in self.categories_.values():
            levels.sort()
        return self

    def transform(self, df):
        out = {}
        for c in self.columns_:
            s = df[c]
            if c in self.categories_:
                s = s.cat.set_categories(self.categories_[c])
                if self.kind == "rf":
                    codes = s.cat.codes.to_numpy().astype("float32")
                    codes[codes < 0] = SENTINEL
                    s = codes
            elif self.kind == "rf":
                s = s.astype("float32").fillna(SENTINEL)
            out[c] = s
        return pd.DataFrame(out, index=df.index)


class LinearPipeline:
    """LR and MLP input. Numeric columns: train median imputation, a missing indicator (for
    columns with any missing training value) and standard scaling. Categoricals with <= 20
    training levels: one-hot plus a missing column; the rest: training frequency encoding.
    Output is a float32 DataFrame with no NaN."""

    def __init__(self, v_drop=None):
        self.v_drop = v_drop

    def fit(self, train_df):
        cols = feature_columns(train_df, self.v_drop)
        self.numeric_ = [c for c in cols if not _is_cat(train_df[c])]
        cats = [c for c in cols if _is_cat(train_df[c])]
        self.medians_, self.means_, self.stds_, self.indicators_ = {}, {}, {}, []
        for c in self.numeric_:
            x = train_df[c].to_numpy(dtype="float64")
            miss = np.isnan(x)
            median = np.median(x[~miss]) if (~miss).any() else 0.0
            filled = np.where(miss, median, x)
            std = filled.std()
            self.medians_[c] = median
            self.means_[c] = filled.mean()
            self.stds_[c] = std if std > 0 else 1.0
            if miss.any():
                self.indicators_.append(c)
        self.onehot_, self.freq_ = {}, {}
        for c in cats:
            s = train_df[c].astype("string")
            levels = sorted(s.dropna().unique())
            if len(levels) <= MAX_ONEHOT_LEVELS:
                self.onehot_[c] = levels
            else:
                self.freq_[c] = s.fillna("__nan__").value_counts(normalize=True).to_dict()
        self.columns_ = (self.numeric_ + [f"{c}__missing" for c in self.indicators_]
                         + [f"{c}=={lv}" for c, lvs in self.onehot_.items() for lv in lvs + ["__nan__"]]
                         + [f"{c}__freq" for c in self.freq_])
        return self

    def transform(self, df):
        out = np.empty((len(df), len(self.columns_)), dtype="float32")
        j = 0
        for c in self.numeric_:
            x = df[c].to_numpy(dtype="float64")
            x = np.where(np.isnan(x), self.medians_[c], x)
            out[:, j] = (x - self.means_[c]) / self.stds_[c]
            j += 1
        for c in self.indicators_:
            out[:, j] = df[c].isna().to_numpy()
            j += 1
        for c, levels in self.onehot_.items():
            s = df[c].astype("string").fillna("__nan__").to_numpy(dtype=object)
            for lv in levels + ["__nan__"]:
                out[:, j] = s == lv
                j += 1
        for c, freq in self.freq_.items():
            s = df[c].astype("string").fillna("__nan__")
            out[:, j] = s.map(freq).astype("float64").fillna(0.0).to_numpy()
            j += 1
        return pd.DataFrame(out, columns=self.columns_, index=df.index, copy=False)


if __name__ == "__main__":
    import resource
    import time

    t0 = time.time()
    joined = load_raw()
    out = save_joined(joined)
    # ru_maxrss is bytes on macOS, kilobytes on Linux
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 1024**3 if peak > 1e8 else peak / 1024**2
    print(f"shape={joined.shape} fraud={int(joined['isFraud'].sum())} "
          f"has_identity={int(joined['has_identity'].sum())}")
    print(f"in-memory={joined.memory_usage(deep=True).sum() / 1024**3:.2f} GB "
          f"peak_rss={peak_gb:.2f} GB file={out.stat().st_size / 1024**2:.0f} MB "
          f"time={time.time() - t0:.0f}s")
