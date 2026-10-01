# BT4012 Group 11: Fraud detection on IEEE-CIS (Vesta)

Benchmark of fraud detection models (logistic regression, random forest, LightGBM, MLP) on the IEEE-CIS Fraud Detection training data. Design decisions are in `Context.md`, the build order is in `BUILD_STEPS.md`, and finished work is logged in `PROGRESS.md`.

## 1. Set up the environment

Python 3.13 was used. From the repo root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run everything below with `.venv/bin/python` (or activate the venv).

## 2. Get the raw data

Download the competition files from Kaggle (needs a Kaggle account and the competition rules accepted) and unzip into `data/raw/`:

```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/raw/
```

Only two files are used:

- `data/raw/train_transaction.csv`: 590,540 rows, 20,663 fraud
- `data/raw/train_identity.csv`: 144,233 rows

The Kaggle test files are not used, because they carry no labels. `data/raw/` is gitignored and read-only: never edit or overwrite it.

## 3. Load and join the data

```bash
.venv/bin/python -m src.data
```

This runs `load_raw()` in `src/data.py`, which:

1. Reads both CSVs with explicit dtypes: string columns (ProductCD, card4, card6, P_emaildomain, R_emaildomain, M1-M9, the string columns among id_12-id_38, DeviceType, DeviceInfo) as `category`; TransactionID, isFraud and TransactionDT as integers; every other column as `float32`.
2. Left-joins identity onto transaction by TransactionID. Rows with no identity record keep NaN in the identity columns.
3. Adds `TransactionDay` (`TransactionDT // 86400`) and `has_identity` (1 if the row matched an identity record).
4. Sorts by TransactionDT, then TransactionID, and resets the index. This time order is what every split and feature relies on, so do not re-sort the table.
5. Saves `data/processed/joined.parquet` (about 80 MB, 590,540 x 436, about 1 GB of RAM; takes about 20 seconds).

In code:

```python
from src.data import load_joined
df = load_joined()   # reads data/processed/joined.parquet
```

Expected counts, which `tests/test_data.py` asserts: 590,540 rows, 20,663 with `isFraud == 1`, 144,233 with `has_identity == 1`, and `TransactionDT` non-decreasing.

## 4. Cleaning

Cleaning is deliberately light so that no fitted statistic can leak from validation or test rows:

- The loaded table keeps NaN as NaN. Missingness is itself a signal (most rows have no identity record), so nothing is dropped or filled at load time.
- Imputation, scaling and encoding are fitted transforms. They are built in `src/data.py` as `tree_pipeline` and `linear_pipeline` (build Step 5, not written yet) and are fit on the training block only. Never fit a transform on the whole table.
- V-column pruning is proposed in `config/features.yaml` under `v_columns_proposed`. It is awaiting group approval, and the pipelines will read the approved list.

To regenerate the data profile and V-column proposal (about 1 minute):

```bash
.venv/bin/python -m src.data_profile
```

This writes `reports/data_profile.md` and `config/features.yaml`, using the first 80% of rows by time only.

## 5. Run the tests

```bash
.venv/bin/python -m pytest -q tests
```

Tests that need the real table are skipped if `joined.parquet` has not been built. The V-column correlation test takes about 3 minutes.

## Rules to keep in mind

- Seeds are 0, 1, 2.
- The last 10% of rows by time (the test block) is not scored until the group declares the experiment freeze.
- Every feature uses only transactions earlier in time than the row it describes.
- Every training run appends one row to `results/log.csv`.
