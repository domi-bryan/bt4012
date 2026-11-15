"""Training-only EDA tables and report; plots live in notebooks/01_eda.ipynb."""
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data import CLEANED_PATH, ROOT, validate_cleaned
from src.data_profile import N_TRAIN, column_profile

OUTPUT = ROOT / "reports" / "eda"
CATEGORY_COLUMNS = ["ProductCD", "card4", "card6", "DeviceType",
                    "P_emaildomain", "R_emaildomain", "has_identity"]
AMOUNT_EDGES = [0, 10, 25, 50, 100, 250, 500, 1000, np.inf]
AMOUNT_LABELS = ["<10", "10–<25", "25–<50", "50–<100", "100–<250",
                 "250–<500", "500–<1000", "1000+"]


def load_training():
    """Materialize only the approved first 472,432 rows of the cleaned parquet."""
    parquet = pq.ParquetFile(CLEANED_PATH)
    if parquet.metadata.num_rows != 590_540:
        raise ValueError("Unexpected dataset size; reconsider the chronological split")
    batches, remaining = [], N_TRAIN
    for batch in parquet.iter_batches(batch_size=65_536):
        batch = batch.slice(0, min(remaining, batch.num_rows))
        batches.append(batch)
        remaining -= batch.num_rows
        if remaining == 0:
            break
    train = pa.Table.from_batches(batches).to_pandas().reset_index(drop=True)
    validate_cleaned(train)
    return train


def rates(labels, target):
    """Observed group counts and rates, keeping missingness as an explicit group."""
    labels = labels.astype("string").fillna("<missing>")
    result = pd.DataFrame({"group": labels, "isFraud": target}).groupby(
        "group", sort=False, observed=True
    ).agg(rows=("isFraud", "size"), fraud=("isFraud", "sum")).reset_index()
    result["fraud_rate"] = result.fraud / result.rows
    result["share"] = result.rows / len(target)
    return result


def markdown_table(frame):
    """Render a small Markdown table without adding a tabulate dependency."""
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |",
             "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines.extend("| " + " | ".join(str(v).replace("|", "\\|") for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def rate_table(frame):
    out = frame[["group", "rows", "fraud", "fraud_rate"]].copy()
    out["rows"] = out.rows.map(lambda x: f"{x:,}")
    out["fraud"] = out.fraud.map(lambda x: f"{x:,}")
    out["fraud_rate"] = out.fraud_rate.map(lambda x: f"{100*x:.2f}%")
    out.columns = ["Group", "Transactions", "Fraud transactions", "Fraud rate"]
    return markdown_table(out)


def run():
    train = load_training()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = train.isFraud
    tables = {"class_balance": rates(target, target)}
    for col in CATEGORY_COLUMNS:
        tables[f"category_{col}"] = rates(train[col], target).sort_values("rows", ascending=False)
    tables["amount_bins"] = rates(pd.cut(train.TransactionAmt, AMOUNT_EDGES,
        labels=AMOUNT_LABELS, right=False), target).set_index("group").reindex(AMOUNT_LABELS).reset_index()
    amount_rows = []
    for label, group in train.groupby("isFraud", observed=True):
        values = group.TransactionAmt.astype("float64")
        amount_rows.append({"isFraud": int(label), "rows": len(values), "mean": values.mean(),
                           "median": values.median(), "p95": values.quantile(.95),
                           "p99": values.quantile(.99), "max": values.max()})
    tables["amount_summary"] = pd.DataFrame(amount_rows)
    profile = column_profile(train)
    profile["missing_rows"] = profile.column.map(train.isna().sum())
    profile["present_rows"] = len(train) - profile.missing_rows
    profile["rate_gap_pp"] = 100 * (profile.fraud_rate_missing - profile.fraud_rate_present)
    tables["missingness"] = profile

    first_day = int(train.TransactionDay.min())
    week = ((train.TransactionDay - first_day) // 7).astype("int16")
    weekly = rates(week, target)
    weekly["week"] = weekly.group.astype(int)
    weekly["start_day"] = first_day + weekly.week * 7
    weekly["end_day"] = weekly.start_day + 6
    weekly["complete_interval"] = ((weekly.start_day * 86400 >= train.TransactionDT.min())
        & ((weekly.end_day + 1) * 86400 <= train.TransactionDT.max()))
    tables["weekly"] = weekly.sort_values("week")

    # Reconciliation checks catch dropped NaN categories and wrong denominators.
    for name, table in tables.items():
        if "fraud_rate" in table:
            assert int(table.rows.sum()) == len(train), name
            assert int(table.fraud.sum()) == int(target.sum()), name
            assert table.fraud_rate.between(0, 1).all(), name
    assert len(profile) == train.shape[1] and profile.column.is_unique
    assert int(profile.missing_rows.sum()) == int(train.isna().sum().sum())
    for name, table in tables.items():
        table.to_csv(OUTPUT / f"{name}.csv", index=False)

    summary = {
        "source": "data/processed/cleaned.parquet",
        "scope": "training only; rows 0 through 472431 in chronological order",
        "train_rows": len(train), "columns": train.shape[1],
        "fraud_count": int(target.sum()), "fraud_rate": float(target.mean()),
        "first_transaction_dt": int(train.TransactionDT.min()),
        "last_transaction_dt": int(train.TransactionDT.max()),
        "first_day": first_day, "last_day": int(train.TransactionDay.max()),
        "missing_cells_pct": float(100 * train.isna().to_numpy().mean()),
        "columns_with_missing": int((profile.missing_rows > 0).sum()),
        "columns_over_90pct_missing": int((profile.pct_missing > 90).sum()),
        "table_reconciliation_passed": True,
        "validation_or_test_analyzed": False,
        "fitted_transforms_or_models": False,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(tables, summary)
    print(json.dumps(summary, indent=2))
    return train, tables, summary


def write_report(tables, s):
    amount = tables["amount_summary"].set_index("isFraud")
    identity = tables["category_has_identity"].set_index("group")
    product = tables["category_ProductCD"].sort_values("fraud_rate", ascending=False)
    balance = tables["class_balance"][["group", "rows", "share"]].copy()
    balance["group"] = balance.group.map({"0": "Legitimate", "1": "Fraud"})
    balance["rows"] = balance.rows.map(lambda n: f"{n:,}")
    balance["share"] = balance.share.map(lambda x: f"{100*x:.2f}%")
    balance.columns = ["Class", "Transactions", "Share of training data"]
    complete = tables["weekly"].query("complete_interval")
    low, high = complete.loc[complete.fraud_rate.idxmin()], complete.loc[complete.fraud_rate.idxmax()]
    lines = [
        "# Training data exploratory analysis", "",
        f"The training block contains {s['train_rows']:,} transactions, including {s['fraud_count']:,} "
        f"labelled fraud ({100*s['fraud_rate']:.2f}%). Fraud rates differ by product and identity availability, "
        "and transaction amounts have a long upper tail. These observations motivate retaining missingness "
        "and evaluating models chronologically; they do not establish causes or model performance.", "",
        "## Scope and reproducibility", "",
        "Source: `data/processed/cleaned.parquet`. All statistics and plots use rows 0 through 472431 "
        "only, the first 80% in the existing time order. No validation or test rows are analyzed. "
        "The cleaned data retains every feature, including the proposed V-column drops. "
        "No values are imputed, resampled, clipped, scaled, or encoded for modelling.", "",
        f"TransactionDay spans {s['first_day']} through {s['last_day']}; the reference date is unknown, "
        "so these are relative days, not calendar dates or weekdays.", "",
        "Run `.venv/bin/python -m src.eda` for tables and this report. Run "
        "`notebooks/01_eda.ipynb` to regenerate the same tables and all figures. "
        "Every reported statistic comes from the CSV tables and `summary.json` in `reports/eda/`. "
        "These are descriptive data summaries, not experiment scores; no model run was added to `results/log.csv`.", "",
        "## Class balance", "", markdown_table(balance), "",
        "The strong imbalance supports "
        "the project's use of PR-AUC for later model evaluation; accuracy alone would obscure minority-class performance.", "",
        "![Class balance](figures/eda_class_balance.png)", "",
        "## Transaction amounts", "",
        f"Legitimate transactions have mean {amount.loc[0, 'mean']:.2f} and median {amount.loc[0, 'median']:.2f}; "
        f"fraud transactions have mean {amount.loc[1, 'mean']:.2f} and median {amount.loc[1, 'median']:.2f}. "
        f"The 99th percentiles are {amount.loc[0, 'p99']:.2f} and {amount.loc[1, 'p99']:.2f}, respectively. "
        "Amounts are reported in dataset units because currency is not established here.", "",
        "The distribution chart uses log10(amount) and normalizes each class separately; it compares "
        "distribution shapes, not the class counts. Its right panel shows the fraud rate by amount decile "
        "with 95% intervals; the table below uses fixed amount bands. No outliers were removed.", "",
        rate_table(tables["amount_bins"]), "",
        "![Amount distributions and fraud rates](figures/eda_amounts.png)", "",
        "## Missing values", "",
        f"{s['columns_with_missing']} of {s['columns']} columns have missing values, with "
        f"{s['columns_over_90pct_missing']} above 90% missing. Across all table cells, "
        f"{s['missing_cells_pct']:.2f}% are missing. Missing identity records are expected from the left join.", "",
        f"Rows with identity records have fraud rate {100*identity.loc['1', 'fraud_rate']:.2f}% "
        f"(n={int(identity.loc['1', 'rows']):,}); rows without them have fraud rate "
        f"{100*identity.loc['0', 'fraud_rate']:.2f}% (n={int(identity.loc['0', 'rows']):,}). "
        "This is an association and may reflect product or transaction mix. It supports retaining missingness "
        "indicators, not assigning fraud labels from identity availability.", "",
        "![Columns with most missing values](figures/eda_missingness.png)", "",
        "The next chart selects the ten largest absolute missing-versus-present fraud-rate gaps among "
        "columns with at least 1% of training rows on each side. It is an exploratory ranking; related "
        "features can repeat the same missingness signal. All 436 columns are in `eda/missingness.csv`.", "",
        "![Missingness associations](figures/eda_missingness_rates.png)", "",
        "## Product and categorical patterns", "", rate_table(product), "",
        f"Product {product.iloc[0]['group']} has the highest observed fraud rate "
        f"({100*product.iloc[0].fraud_rate:.2f}%, n={int(product.iloc[0].rows):,}). "
        "Category rates are unadjusted for other features and time. Small groups are unstable; "
        "use the displayed counts when interpreting differences.", "",
        "![Product card and device patterns](figures/eda_categories.png)", "",
        "Email plots show the 15 most frequent domains (missing included) with 95% intervals. "
        "Full category counts and fraud rates are saved separately. "
        "Domain names are observed feature values; these rates do not imply wrongdoing by a provider.", "",
        "![Email domain patterns](figures/eda_email_domains.png)", "",
        "## Changes over time", "",
        f"Across complete seven-day intervals, fraud rates range from {100*low.fraud_rate:.2f}% "
        f"(relative days {int(low.start_day)}–{int(low.end_day)}, n={int(low.rows):,}) to "
        f"{100*high.fraud_rate:.2f}% (days {int(high.start_day)}–{int(high.end_day)}, n={int(high.rows):,}). "
        "Edge intervals without a full observed seven-day span are marked in the plot and excluded "
        "from this range. Weekly variation may reflect changing transaction mix and motivates "
        "the existing chronological evaluation plan; it does not by itself prove concept drift.", "",
        "![Weekly volume and fraud rate](figures/eda_time.png)", "",
        "## Implications and limits", "",
        "Preserve missingness and positive extreme amounts. Fit preprocessing only on training rows "
        "within each model's training split. Keep the chronological holdout and expanding-window CV. "
        "V-column pruning remains a separate approval decision.", "",
        "These are descriptive associations on the training period. Transactions may share customers, "
        "so observations are not necessarily independent. No significance tests or causal claims are made. "
        "No model was fitted, and future-period performance remains unmeasured.", "",
    ]
    (ROOT / "reports" / "eda_findings.md").write_text("\n".join(lines))


if __name__ == "__main__":
    run()
