# Training data exploratory analysis

The training block contains 472,432 transactions, including 16,599 labelled fraud (3.51%). Fraud rates differ by product and identity availability, and transaction amounts have a long upper tail. These observations motivate retaining missingness and evaluating models chronologically; they do not establish causes or model performance.

## Scope and reproducibility

Source: `data/processed/cleaned.parquet`. All statistics and plots use rows 0 through 472431 only, the first 80% in the existing time order. No validation or test rows are analyzed. The cleaned data retains every feature, including the proposed V-column drops. No values are imputed, resampled, clipped, scaled, or encoded for modelling.

TransactionDay spans 1 through 141; the reference date is unknown, so these are relative days, not calendar dates or weekdays.

Run `.venv/bin/python -m src.eda` for tables and this report. Run `notebooks/01_eda.ipynb` to regenerate the same tables and all figures. Every reported statistic comes from the CSV tables and `summary.json` in `reports/eda/`. These are descriptive data summaries, not experiment scores; no model run was added to `results/log.csv`.

## Class balance

| Class | Transactions | Share of training data |
| --- | --- | --- |
| Legitimate | 455,833 | 96.49% |
| Fraud | 16,599 | 3.51% |

The strong imbalance supports the project's use of PR-AUC for later model evaluation; accuracy alone would obscure minority-class performance.

![Class balance](figures/eda_class_balance.png)

## Transaction amounts

Legitimate transactions have mean 133.87 and median 68.50; fraud transactions have mean 149.04 and median 76.13. The 99th percentiles are 1104.00 and 994.00, respectively. Amounts are reported in dataset units because currency is not established here.

The distribution chart uses log10(amount) and normalizes each class separately; it compares distribution shapes, not the class counts. Fixed amount bands show observed fraud rates with transaction counts. No outliers were removed.

| Group | Transactions | Fraud transactions | Fraud rate |
| --- | --- | --- | --- |
| <10 | 5,171 | 402 | 7.77% |
| 10–<25 | 29,608 | 1,961 | 6.62% |
| 25–<50 | 114,936 | 3,575 | 3.11% |
| 50–<100 | 128,794 | 3,707 | 2.88% |
| 100–<250 | 140,442 | 4,201 | 2.99% |
| 250–<500 | 34,185 | 1,868 | 5.46% |
| 500–<1000 | 13,455 | 752 | 5.59% |
| 1000+ | 5,841 | 133 | 2.28% |

![Amount distributions and fraud rates](figures/eda_amounts.png)

## Missing values

414 of 436 columns have missing values, with 12 above 90% missing. Across all table cells, 44.86% are missing. Missing identity records are expected from the left join.

Rows with identity records have fraud rate 7.55% (n=120,464); rows without them have fraud rate 2.13% (n=351,968). This is an association and may reflect product or transaction mix. It supports retaining missingness indicators, not assigning fraud labels from identity availability.

![Columns with most missing values](figures/eda_missingness.png)

The next chart selects the ten largest absolute missing-versus-present fraud-rate gaps among columns with at least 1% of training rows on each side. It is an exploratory ranking; related features can repeat the same missingness signal. All 436 columns are in `eda/missingness.csv`.

![Missingness associations](figures/eda_missingness_rates.png)

## Product and categorical patterns

| Group | Transactions | Fraud transactions | Fraud rate |
| --- | --- | --- | --- |
| C | 56,410 | 6,391 | 11.33% |
| S | 8,127 | 503 | 6.19% |
| H | 29,691 | 1,377 | 4.64% |
| R | 32,203 | 1,169 | 3.63% |
| W | 346,001 | 7,159 | 2.07% |

Product C has the highest observed fraud rate (11.33%, n=56,410). Category rates are unadjusted for other features and time. Small groups are unstable; use the displayed counts when interpreting differences.

![Product card and device patterns](figures/eda_categories.png)

Email plots show the eight largest nonmissing domains by transaction count, missing rows, and an aggregated remainder. Full category counts and fraud rates are saved separately. Domain names are observed feature values; these rates do not imply wrongdoing by a provider.

![Email domain patterns](figures/eda_email_domains.png)

## Changes over time

Across complete seven-day intervals, fraud rates range from 2.07% (relative days 22–28, n=34,906) to 5.07% (days 113–119, n=21,078). Edge intervals without a full observed seven-day span are marked in the plot and excluded from this range. Weekly variation may reflect changing transaction mix and motivates the existing chronological evaluation plan; it does not by itself prove concept drift.

![Weekly volume and fraud rate](figures/eda_time.png)

## Implications and limits

Preserve missingness and positive extreme amounts. Fit preprocessing only on training rows within each model's training split. Keep the chronological holdout and expanding-window CV. V-column pruning remains a separate approval decision.

These are descriptive associations on the training period. Transactions may share customers, so observations are not necessarily independent. No significance tests or causal claims are made. No model was fitted, and future-period performance remains unmeasured.
