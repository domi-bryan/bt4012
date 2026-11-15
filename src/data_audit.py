"""Read-only data checks; write audit evidence without changing the input tables.

Run with: .venv/bin/python -m src.data_audit
Structural checks cover the saved join. Distribution and cleaning diagnostics
use only the first 80% of rows, as required by the project split.
"""
import json

import numpy as np
import pandas as pd

from src.data import RAW_DIR, ROOT, load_joined
from src.data_profile import N_TRAIN


def main():
    df = load_joined()
    raw = pd.read_csv(
        RAW_DIR / "train_transaction.csv",
        usecols=["TransactionID", "isFraud", "TransactionDT", "TransactionAmt"],
        dtype={"TransactionID": "int32", "isFraud": "int8",
               "TransactionDT": "int32", "TransactionAmt": "float32"},
    )
    identity = pd.read_csv(RAW_DIR / "train_identity.csv", usecols=["TransactionID"])
    expected = raw.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    checks = {
        "expected_transaction_count": len(raw) == len(df) == 590_540,
        "expected_identity_count": len(identity) == 144_233,
        "raw_transaction_ids_unique_nonmissing": bool(
            raw.TransactionID.is_unique and raw.TransactionID.notna().all()),
        "raw_identity_ids_unique_nonmissing": bool(
            identity.TransactionID.is_unique and identity.TransactionID.notna().all()),
        "joined_ids_unique_nonmissing": bool(
            df.TransactionID.is_unique and df.TransactionID.notna().all()),
        "identity_ids_all_in_transactions": bool(identity.TransactionID.isin(raw.TransactionID).all()),
        "raw_core_columns_match_saved_join": df[raw.columns].equals(expected),
        "time_and_id_order_correct": df.TransactionID.equals(expected.TransactionID),
        "time_nondecreasing": bool(df.TransactionDT.is_monotonic_increasing),
        "transaction_day_correct": bool((df.TransactionDay == df.TransactionDT // 86400).all()),
        "identity_flag_correct": bool(
            (df.has_identity == df.TransactionID.isin(identity.TransactionID).astype("int8")).all()),
        "binary_nonmissing_target": bool(df.isFraud.isin([0, 1]).all()),
        "column_names_unique": bool(df.columns.is_unique),
    }
    identity_cols = [c for c in df if c.startswith("id_") or c in ("DeviceType", "DeviceInfo")]
    checks["unmatched_identity_fields_all_missing"] = bool(
        df.loc[df.has_identity == 0, identity_cols].isna().all().all())
    print("Structural checks complete", flush=True)

    train = df.iloc[:N_TRAIN]
    rows = []
    for col in train:
        s = train[col]
        numeric = pd.api.types.is_numeric_dtype(s)
        strings = s.cat.categories if isinstance(s.dtype, pd.CategoricalDtype) else []
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "missing_count": int(s.isna().sum()),
            "missing_pct": float(100 * s.isna().mean()),
            "distinct_nonmissing": int(s.nunique()),
            "infinite_count": int(np.isinf(s.to_numpy()).sum()) if numeric else 0,
            # Negatives are diagnostic only: anonymized feature semantics vary.
            "negative_count": int((s < 0).sum()) if numeric else 0,
            "blank_string_rows": int(s.isin([x for x in strings if isinstance(x, str) and not x.strip()]).sum()),
            "padded_string_rows": int(s.isin([x for x in strings if isinstance(x, str) and x != x.strip()]).sum()),
        })
    profile = pd.DataFrame(rows).sort_values("missing_pct", ascending=False, kind="stable")
    predictors = train.drop(columns=["TransactionID", "isFraud"])
    repeated_predictors = predictors.duplicated(keep=False)
    amount = train.TransactionAmt
    summary = {
        "scope": {
            "structural_checks": "raw key/core columns and full saved join",
            "quality_statistics": "first 472432 rows by time (training only)",
            "data_modified": False,
            "model_fitted_or_test_scored": False,
        },
        "rows": len(df), "columns": len(df.columns), "train_rows": len(train),
        "checks": checks,
        "all_structural_checks_pass": all(checks.values()),
        "training": {
            "fraud_count": int(train.isFraud.sum()),
            "fraud_rate": float(train.isFraud.mean()),
            "duplicate_full_rows_beyond_first": int(train.duplicated().sum()),
            "duplicate_predictor_rows_beyond_first": int(predictors.duplicated().sum()),
            "rows_in_repeated_predictor_groups": int(repeated_predictors.sum()),
            "columns_with_missing": int((profile.missing_count > 0).sum()),
            "columns_over_90pct_missing": profile.loc[profile.missing_pct > 90, "column"].tolist(),
            "all_missing_columns": profile.loc[profile.distinct_nonmissing == 0, "column"].tolist(),
            "constant_nonmissing_columns": profile.loc[profile.distinct_nonmissing == 1, "column"].tolist(),
            "infinite_values": int(profile.infinite_count.sum()),
            "blank_string_cells": int(profile.blank_string_rows.sum()),
            "padded_string_cells": int(profile.padded_string_rows.sum()),
            "missing_identity_rows": int((train.has_identity == 0).sum()),
            "invalid_amount_rows": int((amount.isna() | ~np.isfinite(amount) | (amount <= 0)).sum()),
            "negative_time_rows": int((train.TransactionDT < 0).sum()),
            "amount_quantiles": {str(q): float(v) for q, v in amount.quantile([0, .25, .5, .75, .95, .99, 1]).items()},
            "most_missing_columns": profile[["column", "missing_pct"]].head(10).to_dict("records"),
        },
        "interpretation": [
            "Missing identity is expected from the left join; retain missingness.",
            "Negative anonymized feature values and extreme amounts are not automatically errors.",
            "Repeated predictors are diagnostic only, not authorization to delete transactions.",
            "V-column pruning remains proposed; no columns were removed.",
        ],
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    profile.to_csv(out / "data_quality_columns.csv", index=False)
    payload = json.dumps(summary, indent=2, allow_nan=False)
    (out / "data_quality_audit.json").write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
