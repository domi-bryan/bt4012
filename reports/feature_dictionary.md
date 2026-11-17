# Feature dictionary

Feature sets (Context.md):

- **F0 Raw:** cleaned transaction and identity columns (pipelines drop IDs, raw time and pruned V columns).
- **F1 Card:** F0 + the 11 aggregates below grouped by `card1` (prefix `card1_`).
- **F2 Customer:** F1 + the same 11 aggregates grouped by `uid` (prefix `uid_`).

Built by `build_feature_set(df, "F1" | "F2")` in `src/features.py`. uid = card1 + addr1 + floor(TransactionDay - D1); see `reports/uid_check.md`.

## How leakage is prevented

Every aggregate for a row uses only rows that come **earlier in the time-sorted table** with the same key. The row itself and everything after it are excluded, and no label (`isFraud`) is used, so the features are available at transaction time. Rows that share a TransactionDT count as earlier when they have a lower TransactionID; this affects 166 card1 rows and 76 uid rows. The aggregates are computed on the full table in time order, so a validation or test row may use history from earlier blocks, as it would in production, but never from later rows. A test recomputes every aggregate for 200 sampled rows from `df.iloc[:row]` and matches to within 1e-5.

A key's first transaction has count features 0, and mean, ratio and time-since-previous NaN. Std needs two prior rows (ddof = 1).

## Aggregate columns

Each `card1_*` column has a `uid_*` twin computed the same way over uid groups.

| Column (card1 / uid) | Definition | Why it might signal fraud |
| --- | --- | --- |
| `card1_prior_count` / `uid_prior_count` | Number of earlier transactions with the same key. | New or rarely seen cards and customers carry more risk; a sudden history after none can mark a newly compromised card. |
| `card1_prior_amt_mean` / `uid_prior_amt_mean` | Mean TransactionAmt of earlier transactions with the key; NaN if none. | Describes the key's normal spending level, as context for the current amount. |
| `card1_prior_amt_std` / `uid_prior_amt_std` | Sample std (ddof = 1) of earlier amounts; NaN with fewer than two. | Erratic spending is unusual for a genuine customer; very regular amounts can indicate scripted testing. |
| `card1_amt_to_prior_mean` / `uid_amt_to_prior_mean` | Current amount divided by `*_prior_amt_mean`. | Fraudsters often spend well above (cash-out) or below (card testing) the owner's usual amount. |
| `card1_secs_since_prev` / `uid_secs_since_prev` | Seconds since the key's previous transaction; NaN if none. | Rapid repeat purchases (velocity) are a classic sign of a stolen card being used before it is blocked. |
| `card1_count_1h` / `uid_count_1h` | Earlier transactions with the key in the trailing hour (TransactionDT > t - 3,600). | Short-window velocity: bursts of transactions within an hour. |
| `card1_count_24h` / `uid_count_24h` | Same over the trailing 24 hours. | Daily velocity relative to the key's usual pace. |
| `card1_count_7d` / `uid_count_7d` | Same over the trailing 7 days. | Weekly activity; a spike against a quiet history stands out. |
| `card1_n_P_emaildomain` / `uid_n_P_emaildomain` | Distinct non-missing purchaser email domains among earlier transactions. | One card used with many email accounts suggests shared or stolen card details. |
| `card1_n_addr1` / `uid_n_addr1` | Distinct non-missing billing regions (addr1) among earlier transactions. | Many billing addresses on one card is unusual. For uid this is always 0 or 1, because addr1 is part of the uid. |
| `card1_n_DeviceInfo` / `uid_n_DeviceInfo` | Distinct non-missing DeviceInfo values among earlier transactions. | Several devices behind one card or customer can mean account takeover. |

## Missing values on the train block

Share of training rows where the aggregate is NaN (counts are never NaN):

| Aggregate | card1 | uid |
| --- | --- | --- |
| prior_amt_mean, amt_to_prior_mean, secs_since_prev | 2.69% | 38.64% |
| prior_amt_std | 4.67% | 54.80% |

## Top features (Step 17)

The inputs with the best average permutation-importance rank across the six models on G3 (`notebooks/10_model_comparison.ipynb`, section 5; rank 1 = most important). C2 and `fe_amt_log` tie for fifth. Vesta masks the meaning of the C columns; the descriptions follow the competition's data notes.

| Input | Mean rank | Group | Plain description |
| --- | --- | --- | --- |
| `fe_D1n` | 4.8 | G1 row-level | The day the card was first used: TransactionDay minus D1 (days since the card's first transaction), so it is constant for a card across its transactions and, with card1 and addr1, identifies a customer. |
| `C11` | 9.2 | G0 raw | A masked count linked to the payment card (Vesta's C columns count things such as addresses associated with the card). |
| `C14` | 11.8 | G0 raw | Another masked count linked to the payment card. |
| `C1` | 13.5 | G0 raw | A masked count linked to the payment card; the most important input for random forest, LightGBM, XGBoost and CatBoost. |
| `C2` | 23.2 | G0 raw | Another masked count linked to the payment card. |
| `fe_amt_log` | 23.2 | G1 row-level | log(1 + TransactionAmt): the amount on a log scale, mainly useful to the linear models and the MLP. |
