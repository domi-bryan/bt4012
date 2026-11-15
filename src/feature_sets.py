"""The G0-G3 feature sets from `config/features_v2.yaml` (built by notebooks/02_feature_engineering.ipynb).

G0 raw columns (197) -> G1 + row-level features (224) -> G2 + card/device history (234)
-> G3 + customer history (252). Each set includes the ones before it.

`load_features()` returns the features_v2 table with the test block already removed, so a
model notebook can never touch it before the experiment freeze.
"""
import pandas as pd
import yaml

from src.data import NON_FEATURES, ROOT
from src.split import holdout_indices

SPEC_PATH = ROOT / "config" / "features_v2.yaml"
FEATURE_SETS = ("G0", "G1", "G2", "G3")
NOT_INPUTS = set(NON_FEATURES) | {"uid"}


def load_spec(path=SPEC_PATH):
    with open(path) as handle:
        return yaml.safe_load(handle)


def feature_list(name, spec=None):
    """Input columns of one feature set, in the order stored in the YAML."""
    spec = spec or load_spec()
    if name not in spec["feature_sets"]:
        raise ValueError(f"Unknown feature set {name!r}; expected one of {sorted(spec['feature_sets'])}")
    cols = list(spec["feature_sets"][name])
    bad = sorted(NOT_INPUTS & set(cols))
    if bad:
        raise ValueError(f"{name} contains identifiers, time or the target: {bad}")
    return cols


def load_features(spec=None, path=None):
    """(df, train_idx, val_idx): features_v2 rows of the train and validation blocks only.

    Splits come from `src.split.holdout_indices` on the full table, then the test block is
    dropped, so the positions are the same as in every other part of the project.
    """
    spec = spec or load_spec()
    df = pd.read_parquet(path or ROOT / spec["parquet"])
    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError("features_v2 must stay in time order")
    train_idx, val_idx, _ = holdout_indices(df)
    if len(train_idx) != spec["train_block_rows"]:
        raise ValueError("Training block size differs from the one the features were built with")
    df = df.iloc[:val_idx[-1] + 1].copy()
    return df, train_idx, val_idx
