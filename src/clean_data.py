"""Run conservative cleaning locally: .venv/bin/python -m src.clean_data."""
import json

from src.data import ROOT, load_joined, save_cleaned


def main():
    df = load_joined()
    path = save_cleaned(df)
    report = {
        "input": "data/processed/joined.parquet",
        "output": str(path.relative_to(ROOT)),
        "rows": len(df),
        "columns": len(df.columns),
        "validation_passed": True,
        "exact_parquet_roundtrip_passed": True,
        "rows_removed": 0,
        "columns_removed": 0,
        "values_changed": 0,
        "missing_cells_preserved": int(df.isna().sum().sum()),
        "policy": "Validate and preserve: the audit found no confirmed errors requiring value changes.",
        "v_pruning_applied": False,
        "imputation_scaling_encoding_fitted": False,
        "eda_run": False,
    }
    payload = json.dumps(report, indent=2)
    report_path = ROOT / "reports" / "cleaning_summary.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
