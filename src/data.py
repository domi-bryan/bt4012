"""Role A: load the raw IEEE-CIS tables, join them and save the time-sorted table."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
JOINED_PATH = ROOT / "data" / "processed" / "joined.parquet"

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
