# BT4012 Group 11: Fraud detection on IEEE-CIS (Vesta)

Benchmark of six fraud detection models (logistic regression, random forest, LightGBM, XGBoost, CatBoost and an MLP) on four cumulative feature sets (G0–G3) of the IEEE-CIS Fraud Detection training data, plus a segment blend of the tree models. Final test results are in `reports/test_results.md`. Design decisions are in `Context.md`, the build order is in `BUILD_STEPS.md`, and finished work is logged in `PROGRESS.md`.

Notebooks, in reading order: `01_eda` (exploratory analysis) → `02_feature_engineering` (builds `features_v2`
and the G0–G3 feature groups) → `03_v_threshold` (chooses the V-column threshold, r = 0.75) → one notebook per
model (`04_logreg` to `09_mlp`) → `10_model_comparison` → `11_leakage_check` → `13_ensemble_check` (chooses the
segment blend) → `12_test` (final test results).

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
- V-column pruning: the group agreed to drop V columns correlated above r = 0.75 with a kept one (`notebooks/03_v_threshold.ipynb`), and the G0–G3 feature sets in `config/features_v2.yaml` use only that rule. The older r > 0.9 list in `config/features.yaml` was used only by the F0–F2 first pass (see below), and `src/data.py` still warns that this older list is unapproved whenever it is loaded.

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

## 6. Feature engineering and the model notebooks (G0–G3)

Build the feature table once (a few minutes):

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 notebooks/02_feature_engineering.ipynb
```

This writes `data/processed/features_v2.parquet` (gitignored) and `config/features_v2.yaml`, which defines four
cumulative feature sets: G0 raw columns (197) → G1 + row-level features (224) → G2 + card and device history (234)
→ G3 + customer history (252). `notebooks/03_v_threshold.ipynb` documents the V-column threshold (r = 0.75).

Each model has its own notebook: `04_logreg`, `05_random_forest`, `06_lightgbm`, `07_xgboost`, `08_catboost`,
`09_mlp`. They all follow the same steps and call the same code, so their results are comparable:

1. `src.feature_sets.load_features()` loads `features_v2` with the test block already removed, and
   `feature_list("G0")` … `("G3")` gives each set's columns.
2. `src.experiment.tune_and_validate()` tunes the model's settings grid (`GRID` in `src/models/<model>.py`) with
   time CV inside the training block (seed 0), then refits the best settings on the whole training block with
   seeds 0, 1 and 2 and scores the validation block. The three boosting models (LightGBM, XGBoost, CatBoost) all
   use a random 80% of rows and 80% of columns per tree (fixed, not tuned), and all three tune L2 regularisation on
   the leaf weights over {1, 3, 10}, so they differ only in their algorithms.
   Models that early-stop (the three boosting models and the MLP) hold out the last 10% (by time) of each training
   part for it, so no model is ever tuned or stopped on the rows it is scored on. In tuning that is the whole fit;
   validation (and the saved fold scores) then refit on the whole training part with the stopping point found
   (number of trees, or epochs for the MLP), so the scored model trains on every training row, as logistic
   regression and random forest do. The boosting models allow up to 3,000 trees. Search budgets: logistic
   regression 11 settings (grid), random forest 12 (grid), MLP 16 (random), boosting models 20 each (random). By default (`TUNE_ON = "G3"`) it tunes once on G3 and
   validates those settings on every set; `TUNE_ON = None` tunes each set separately.
3. Every fit appends a row to `results/log.csv`, and anything already logged is skipped, so a notebook can be
   stopped and rerun. Validation scores (and, with `SAVE_OOF`, the time-CV fold scores of the chosen settings) are
   saved to `results/preds/` (gitignored) for the comparison and the segment blend.

Each model notebook reports, per feature set, PR-AUC (main metric), AUROC, recall and precision in the top 1% of
scores, and precision and recall at the F1 threshold chosen on validation, plus precision–recall curves, known vs
unknown customers, how each tuned setting affects time-CV PR-AUC (5.5, for RQ4; read from the tuning runs in the
log, no extra training), and feature importance (section 6): permutation importance (drop in validation PR-AUC when a column
is shuffled; comparable across models) and the model's built-in importance where it has one (gain for LightGBM and
XGBoost, impurity for random forest, prediction values change for CatBoost, |coefficient| for logistic regression;
none for the MLP). Importance tables are saved to `results/importance/` (committed) and the refitted models to
`results/models/` (gitignored).
`10_model_comparison.ipynb` reads the log and shows every model × feature set (mean ± sd over seeds), what each
feature group adds, and PR-AUC for **known vs unknown customers** (`uid` seen in the training data or not;
`src/segments.py`) with 95% bootstrap intervals, on the validation block and on the time-CV fold scores (where
"known" means seen in that fold's training months). The split only uses information available when a transaction
arrives, so it can also be used to weight models in an ensemble; choose any weights on the fold scores and check
them on validation. Run the model notebooks in Jupyter or with the `nbconvert`
command above (use a long timeout, e.g. `--ExecutePreprocessor.timeout=-1`, as full tuning can take hours).

On macOS, XGBoost, PyTorch and LightGBM each ship their own OpenMP library and can crash when combined in one
process. Each model notebook imports only its own model library first, which is safe; don't train several model
libraries in one kernel. For the same reason the XGBoost test and the LightGBM final-test test run in separate processes.

Section 7 is the customer ID check, section 8 the leakage check on G3, section 9 the segment blend and section 10
the final test scoring.

## 7. Customer ID check (Step 7)

```bash
.venv/bin/python -m src.features
```

Builds uid = card1 + addr1 + floor(TransactionDay - D1) (`add_uid` in `src/features.py`) and
writes `reports/uid_check.md`: group counts and sizes, fraud concentration on the train block,
and how many validation/test customers already appear in earlier blocks. Takes a few seconds.

## 8. Leakage check on G3 (Step 18)

```bash
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 .venv/bin/python -u -m src.leakage
```

Scores LR and LightGBM on G3 inside the train block under random stratified 5-fold and
StratifiedGroupKFold on uid (seed 0), logging each fold with `cv_scheme` "random" or
"group_uid". The settings are the ones validated in `04_logreg` and `06_lightgbm`, read from
the log, and the time-CV rows are the tuning rows of those settings, so run those two
notebooks first. Every fold is fitted like a tuning fold: the pipeline is refit on the fold's
training part, and LightGBM early-stops on the last 10% (by time) of that part, never on the
rows it is scored on. It checks that no uid sits on both sides of a group fold, and it resumes
rather than duplicating rows. `--report` only rebuilds `reports/leakage_check.md` from the
log. `notebooks/11_leakage_check.ipynb` reads the log and saves `reports/figures/leakage_check.png`.

## 9. Segment blend (ensemble check)

`notebooks/13_ensemble_check.ipynb` chooses the blend from the saved time-CV fold scores and checks it on the
validation block. Scores are turned into percentile ranks within the scored block, and each transaction is
blended by whether its customer is known (`uid` seen in the train block; `src/segments.py`):

- known customers: XGBoost only;
- unknown customers: CatBoost 0.6, LightGBM 0.2, XGBoost 0.1, random forest 0.1.

The weights were picked from every weighting in steps of 0.1 (286 for the four models) by mean PR-AUC over the four
time-CV folds, never on the validation block. They are fixed in `src/ensemble.py` (`BLEND_KNOWN`,
`BLEND_UNKNOWN`), on G3 only. The blend needs no training; it logs its validation rows (and the F1 threshold) from
the saved validation scores of the four models:

```bash
.venv/bin/python -m src.ensemble --log-validation
```

## 10. Final test scoring (Step 19)

The test block (the last 10% by time, 59,054 rows) was scored once, after the group agreed the experiment freeze on
17 Nov 2026. No setting, threshold, feature set or model choice changed after the freeze.

```bash
.venv/bin/python -u -m src.final_test --model xgboost --freeze-confirmed   # one model per process
.venv/bin/python -u -m src.final_test --blend --freeze-confirmed           # after XGBoost, LightGBM, CatBoost and RF
.venv/bin/python -m src.final_test --report                                # rebuilds reports/test_results.md
```

`--model` takes `logreg`, `rf`, `lgbm`, `xgboost`, `catboost` or `mlp`. For each of the model's 12 cells (G0–G3 x
seeds 0, 1, 2) it refits the settings from the cell's validation row on the train block exactly as in validation
(same rows, seed and two-step early stopping), checks that the refit reproduces the logged validation PR-AUC to
within 1e-6 (and stops with "would not belong" otherwise), then scores the test block with the threshold stored at
validation and logs one row with `split = test`. Test scores go to `results/preds/<model>_<set>_test_seed<seed>.parquet`.
Cells already logged are skipped, so a run can be stopped and restarted. Results can differ slightly across
machines and library versions, which is why the reproduction check comes first: a test row is only logged for the
model that was validated, whichever member's machine ran it.

`--blend` combines the four input models' saved G3 test scores with the fixed weights (each input file must
reproduce its logged test PR-AUC) and applies the blend's stored validation threshold. The log then holds 75 test
rows: 6 models x 4 sets x 3 seeds, plus 3 for the blend. `notebooks/12_test.ipynb` reads them and shows the test
tables, the change from validation to test, known vs unknown customers on test, and the blend against XGBoost.

## 11. Run the tests

```bash
.venv/bin/python -m pytest -q tests
```

Tests that need the real table are skipped if `joined.parquet` has not been built. The V-column correlation test takes about 3 minutes.

## Earlier first pass (F0–F2)

In October the group ran a first pass on three feature sets, F0 (raw), F1 (+ card1 history) and F2 (+ customer
history), with logistic regression, random forest, LightGBM, the MLP and CatBoost. It was superseded by the G0–G3
build and is not used in the report. Its training scripts, settings files, manifests, summaries and notebooks were
removed before submission (they remain in the git history); its rows stay in `results/log.csv` under feature sets
F0, F1 and F2 and are left out of every G0–G3 table. `PROGRESS.md` and Part 2 of `BUILD_STEPS.md` record what it
found.

## Rules to keep in mind

- Seeds are 0, 1, 2.
- The last 10% of rows by time (the test block) was not scored until the experiment freeze (17 Nov 2026), and then only once, by `src.final_test`.
- Every feature uses only transactions earlier in time than the row it describes.
- Every training run appends one row to `results/log.csv`.
- Notebooks are committed with their outputs, so the results and plots can be read on GitHub
  without the data or a rerun. Run the notebook top to bottom before committing, so the outputs
  match the code. Keep outputs short: print summary tables, not whole DataFrames.
  PNGs in `reports/figures/` are gitignored and rebuilt by the notebooks.
