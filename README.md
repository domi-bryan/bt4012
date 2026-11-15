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

```bash
.venv/bin/python -m src.clean_data
```

This validates the existing joined table and saves a separate `data/processed/cleaned.parquet`,
with an exact Parquet roundtrip check before publishing the file. The audit found no
confirmed errors requiring value changes, so all rows, columns, missing values, and
extreme amounts are preserved. Validation rejects invalid keys, labels, amounts,
time ordering, dtypes, derived fields, infinities, and blank or padded categories.
It does not automatically repair failures. The run writes `reports/cleaning_summary.json`;
use `load_cleaned()` from `src.data` to read the result. Raw files and `joined.parquet`
are preserved. V-column pruning is not applied.

- The loaded table keeps NaN as NaN. Missingness is itself a signal (most rows have no identity record), so nothing is dropped or filled at load time.
- Imputation, scaling and encoding are fitted transforms. `TreePipeline` and `LinearPipeline` in `src/data.py` fit on each fold's training rows only. Never fit a transform on the whole table.
- V-column pruning is proposed in `config/features.yaml` under `v_columns_proposed`. The pipelines currently use that list provisionally, with a warning, pending group approval.

To regenerate the data profile and V-column proposal (about 1 minute):

```bash
.venv/bin/python -m src.data_profile
```

This writes `reports/data_profile.md` and `config/features.yaml`, using the first 80% of rows by time only.

## 5. Run exploratory data analysis

After creating `cleaned.parquet`, run the EDA notebook locally (about 1 minute):

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 notebooks/01_eda.ipynb
```

`notebooks/01_eda.ipynb` is the single EDA notebook (it absorbed the former `01_profile.ipynb`).
All label-based analysis uses the first 472,432 rows by time; the validation block is used
without labels to measure shift. It regenerates `reports/eda_findings.md`, twelve supporting
CSV tables plus `summary.json` in `reports/eda/`, and seven `eda_*.png` figures in
`reports/figures/`. Topics: class balance, column groups and descriptive statistics,
missingness and the V-column blocks, fraud over time, customer keys and `uid`, categories,
amounts and D columns, past-behaviour and domain-knowledge checks, redundancy, train-validation
shift, feature stability over time, and the proposed model inputs (section 13). No model is
fitted, and it does not rewrite `config/features.yaml` (only `src.data_profile` does).

To regenerate only the tables and findings report:

```bash
.venv/bin/python -m src.eda
```

## 6. Logistic regression baseline (Step 6)

```bash
.venv/bin/python -m src.models.logreg
```

Tunes C over {0.01, 0.1, 1, 10} with time CV on the train block, refits the best C on the
whole train block and scores the validation block. Each fold and the validation result
append a row to `results/log.csv` (about 13 minutes, 1.5 GB peak memory). It refuses to run
again for the same feature set unless you pass `--force`. Then execute
`notebooks/02_baseline.ipynb`, which only reads the log and saves
`reports/figures/baseline_logreg_cv.png`.

## 7. Customer ID check (Step 7)

```bash
.venv/bin/python -m src.features
```

Builds uid = card1 + addr1 + floor(TransactionDay - D1) (`add_uid` in `src/features.py`) and
writes `reports/uid_check.md`: group counts and sizes, fraud concentration on the train block,
and how many validation/test customers already appear in earlier blocks. Takes a few seconds.

## 8. Features and tree models (Steps 8–9)

`build_feature_set(df, "F1")` adds prior card history; `"F2"` also adds prior customer
history. Definitions and tied-timestamp conventions are in `reports/feature_dictionary.md`.
The LR command above also accepts `--feature-set F1` or `--feature-set F2`.

Run the Step 9 experiment with:

```bash
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 .venv/bin/python -u -m src.train_trees
```

This uses the existing F0 LR results, tunes LR on F1/F2, and tunes RF and LightGBM
on all three feature sets using time CV with seed 0. Selected configurations are
validated with seeds 0, 1 and 2. The test block is excluded before feature construction.
Run only one training command at a time.

- LightGBM: a reproducible sample of 20 configurations over `num_leaves`,
  `learning_rate`, `min_child_samples`, and `scale_pos_weight`. The balanced
  weight is recomputed from each training part. Early stopping uses validation
  average precision with patience 50 and a cap of 1,500 trees.
- RF: four combinations of depth {12, 24} and minimum leaf size {5, 20},
  with 300 trees and `class_weight="balanced_subsample"`.
- LR: the same C grid as Step 6.

Every completed fit appends to `results/log.csv`. Repeating the command resumes
missing folds and validation seeds; it does not rerun completed fits. Optional
`--models logreg rf lgbm` and `--feature-sets F0 F1 F2` restrict the work.
`results/step9_manifest.json` fingerprints the data, feature configuration, model
code and package versions; a changed fingerprint stops resumption to avoid mixing
experiments. `config/best_params.yaml` is updated after each cell's tuning completes.
LightGBM's actual best iteration and resolved weight are stored in `params._fit`
in each log row; validation thresholds remain in `params.threshold`.

After training, regenerate the results notebook and comparison plot:

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 notebooks/03_trees.ipynb
```

The notebook checks all 27 selected validation runs and the complete CV grid, and
generates `reports/figures/trees_validation.png` from the log. The completed Step 9
results and per-seed runtimes are recorded in `PROGRESS.md`. Results remain
provisional on the V-column proposal.

## 9. MLP (Step 10)

```bash
.venv/bin/python -u -m src.train_mlp
```

PyTorch MLP on the linear pipeline output, on CPU with 4 threads: two ReLU hidden layers
(hidden, hidden/2) with dropout, Adam, batch 2,048, `BCEWithLogitsLoss` with
`pos_weight = negatives / positives` from the training part, early stopping on validation
PR-AUC (patience 5, max 40 epochs, best epoch restored). Grid: hidden {128, 256} x dropout
{0.1, 0.3} x learning rate {3e-4, 1e-3}, tuned with time CV (seed 0) on F0, F1 and F2; the
best config is validated with seeds 0, 1 and 2. About 21 minutes and 1.8 GB peak memory.
Like Step 9 it resumes instead of duplicating rows, and refuses to run if its inputs changed
since `results/step10_manifest.json` was written. Per-epoch loss and validation PR-AUC are
stored in each log row's `params._fit`; `notebooks/04_mlp.ipynb` reads them and saves
`reports/figures/mlp_training_curves.png`.

## 10. Validation grid summary (Step 11)

```bash
.venv/bin/python -m src.summarise
```

Rebuilds `reports/grid_validation.md` from the validation rows of `results/log.csv`: the
4 models x 3 feature sets table (mean ± std over seeds 0, 1, 2) for PR-AUC, AUROC and recall
in the top 1%, with the LR → LightGBM and F0 → F2 gains. It refuses to build if any cell is
missing a seed or has a repeated one.

## 11. Leakage check (Step 12)

```bash
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 .venv/bin/python -u -m src.leakage
```

Scores LR and LightGBM on F2 (Step 9 best configs) inside the train block under random
stratified 5-fold and StratifiedGroupKFold on uid (seed 0), logging each fold with
`cv_scheme` "random" or "group_uid". The time-CV rows for the same configs come from Step 9.
About 30 minutes and 2 GB peak memory; it resumes rather than duplicating rows and checks
`results/step12_manifest.json`. `--report` only rebuilds `reports/leakage_check.md` from the
log. `notebooks/05_leakage.ipynb` reads the log and saves `reports/figures/leakage_check.png`.

## 12. Run the tests

For the optional CatBoost extension, install `requirements-catboost.txt` and see
[CATBOOST.md](CATBOOST.md) for tuning, sensitivity analysis, encoding policy and
the dependency on the open shared implementation PR.

```bash
.venv/bin/python -m pytest -q tests
```

Tests that need the real table are skipped if `joined.parquet` has not been built. The V-column correlation test takes about 3 minutes.

## Rules to keep in mind

- Seeds are 0, 1, 2.
- The last 10% of rows by time (the test block) is not scored until the group declares the experiment freeze.
- Every feature uses only transactions earlier in time than the row it describes.
- Every training run appends one row to `results/log.csv`.
- Notebooks are committed with their outputs, so the results and plots can be read on GitHub
  without the data or a rerun. Run the notebook top to bottom before committing, so the outputs
  match the code. Keep outputs short: print summary tables, not whole DataFrames.
  PNGs in `reports/figures/` are gitignored and rebuilt by the notebooks.
