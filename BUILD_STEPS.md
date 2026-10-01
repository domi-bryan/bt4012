# Build steps

Work through these in order. Each step names the role that owns it (see `Context.md`), the files it touches, what to do, and a **Done when** list. A step is finished only when every Done-when item is checked. Then append to `PROGRESS.md` and stop.

`PROGRESS.md` entry format:

```
## Step N: <name> (YYYY-MM-DD)
- Built: <files>
- Checks: <each Done-when item and its result>
- Numbers: <key counts or scores, copied from output>
- Open: <questions for the group, or "none">
```

Prerequisite: `data/raw/train_transaction.csv` and `data/raw/train_identity.csv` from Kaggle (`kaggle competitions download -c ieee-fraud-detection`, unzip into `data/raw/`). Test files are not used.

---

## Step 0: Environment and data check

**Role:** A

**Files:** `requirements.txt`, `.gitignore`, `src/__init__.py`, `tests/`, `PROGRESS.md`, `results/`

**Do**

1. Create the folders in `Context.md` (Repo layout) plus `tests/`, `reports/` and `data/processed/`.
2. Write `requirements.txt`: pandas, pyarrow, numpy, scikit-learn (>=1.4), lightgbm, torch, shap, matplotlib, seaborn, pytest, pyyaml.
3. Add `data/raw/` and `data/processed/` to `.gitignore`.
4. Create `PROGRESS.md` with a one-line header.

**Done when**

- [ ] Every package imports in the project environment.
- [ ] Both raw CSVs exist; `train_transaction.csv` has 590,540 data rows and `train_identity.csv` has 144,233.

---

## Step 1: Load, join, save

**Role:** A

**Files:** `src/data.py`, `tests/test_data.py`

**Do**

1. `load_raw()`: read both CSVs with float columns as float32 and the string columns (ProductCD, card4, card6, P_emaildomain, R_emaildomain, M1-M9, id_12-id_38 string columns, DeviceType, DeviceInfo) as `category`.
2. Left-join identity onto transaction by TransactionID.
3. Add `TransactionDay = TransactionDT // 86400` and `has_identity` (1 if the row matched identity).
4. Sort by TransactionDT, then TransactionID, and reset the index. This order is the time order every later step relies on.
5. Save `data/processed/joined.parquet`.

**Done when**

- [ ] Joined table has 590,540 rows, 20,663 with isFraud = 1, and 144,233 with has_identity = 1.
- [ ] TransactionDT is non-decreasing down the saved file.
- [ ] Peak memory and file size are recorded in PROGRESS.md.

---

## Step 2: Data profile and V-column pruning proposal

**Role:** A

**Files:** `notebooks/01_profile.ipynb`, `reports/data_profile.md`, `config/features.yaml`

**Do**

1. Report per column: dtype, % missing, distinct count, fraud rate when missing vs present. Sort by % missing.
2. Group V1-V339 by identical missing-count pattern. Within each group, propose a subset to keep: drop one of any pair with |Pearson r| > 0.9 on training rows (first 80% by time), keeping the column with more distinct values.
3. Write the proposed keep and drop lists to `config/features.yaml` under `v_columns_proposed`.

**Done when**

- [ ] `reports/data_profile.md` covers every column.
- [ ] Every V column appears in exactly one of keep or drop.
- [ ] The proposal is summarised for the user (groups, columns kept, columns dropped). **Stop here for the group to approve the list** before Step 5 uses it.

---

## Step 3: Splits

**Role:** C

**Files:** `src/split.py`, `tests/test_split.py`

**Do**

1. `holdout_indices(df)`: on the time-sorted table, train = rows 0 to 472,431, validation = next 59,054, test = last 59,054. Return row-index arrays.
2. `time_cv_folds(df, train_idx)`: bin training rows by `TransactionDay // 30` (relative to the first day). Fold k trains on bins 0..k-1 and validates on bin k, for every k >= 1. Return a list of (train_idx, val_idx).
3. `random_cv_folds(train_idx, y, k=5, seed)`: StratifiedKFold over the train block. Leakage check only.
4. `group_cv_folds(train_idx, y, groups, k=5, seed)`: StratifiedGroupKFold over the train block. Leakage check only.

**Done when**

- [ ] Tests pass: holdout blocks are disjoint and cover every row; max TransactionDT of train <= min of validation <= min of test.
- [ ] Each time fold's training rows are all earlier than its validation rows.
- [ ] Group folds never place one group in both sides of a fold.
- [ ] Fraud rate and row count of each block and each time fold are in PROGRESS.md.

---

## Step 4: Metrics and results log

**Role:** C

**Files:** `src/metrics.py`, `src/results.py`, `tests/test_metrics.py`

**Do**

1. `evaluate(y_true, y_score, threshold=None)` returns pr_auc (average precision), auroc, precision, recall, recall_top1pct. With no threshold given, precision and recall are left empty.
2. `pick_threshold(y_true, y_score)`: the threshold that maximises F1 on the given (validation) data.
3. `recall_top1pct`: share of all fraud rows that fall in the top 1% of scores.
4. `log_run(**fields)` appends one row to `results/log.csv` with the columns listed in `Context.md`, creating the file with a header if missing.

**Done when**

- [ ] pr_auc matches `sklearn.metrics.average_precision_score` on a test case.
- [ ] A random-score test case gives pr_auc close to the positive rate.
- [ ] `log_run` called twice gives a file with one header and two rows.

---

## Step 5: Preprocessing pipelines

**Role:** A

**Files:** `src/data.py` (extend), `tests/test_pipelines.py`

**Do**

1. Read the approved V-column list from `config/features.yaml`.
2. `tree_pipeline`: keeps NaN and `category` dtypes for LightGBM. A variant for Random Forest fills NaN with -999 and encodes categoricals as integer codes.
3. `linear_pipeline` (LR and MLP): numeric columns get median imputation, a missing-indicator column, and standard scaling; categoricals with <= 20 levels get one-hot encoding, the rest get frequency encoding.
4. Both pipelines expose `fit(train_df)` and `transform(df)`, and are fit on training rows only.

**Done when**

- [ ] `linear_pipeline` output has no NaN on validation rows.
- [ ] A test confirms medians and scaler means equal those computed on training rows alone.
- [ ] Output column counts for each pipeline are in PROGRESS.md.

---

## Step 6: Logistic regression baseline on F0

**Role:** C

**Files:** `src/models/logreg.py`, `notebooks/02_baseline.ipynb`

**Do**

1. Implement `fit` and `predict_proba` per the interface in `Context.md`, with `class_weight="balanced"`.
2. Tune C over {0.01, 0.1, 1, 10} with time CV; refit the best C on the full train block; score the validation block and pick its threshold.
3. Log every CV fold and the validation result.

**Done when**

- [ ] `results/log.csv` has rows for every C and fold, plus one validation row.
- [ ] Validation PR-AUC is above 0.035; the value is in PROGRESS.md.
- [ ] No row in the log has split = test.

---

## Step 7: Customer ID (uid) and check

**Role:** B

**Files:** `src/features.py`, `reports/uid_check.md`

**Do**

1. `account_open_day = TransactionDay - D1` (floored); `uid = card1 + "_" + addr1 + "_" + account_open_day`, with missing parts written as "nan".
2. Report: number of distinct uids, group-size distribution, share of rows in single-transaction groups, share of uids with any fraud, and the fraud rate within those.
3. Report how many test-block uids also appear in the train block (returning customers).

**Done when**

- [ ] `reports/uid_check.md` contains every item above, for card1 and uid side by side.
- [ ] Rows with missing D1 or addr1 are counted and the count is in the report.

---

## Step 8: Past-only aggregates (F1, F2)

**Role:** B

**Files:** `src/features.py` (extend), `tests/test_features.py`, `reports/feature_dictionary.md`

**Do**

For key in {card1, uid}, on the time-sorted table, compute for each row using only rows **earlier in the sort order** with the same key:

1. Prior transaction count.
2. Prior TransactionAmt mean and std; current amount divided by prior mean.
3. Seconds since the key's previous transaction.
4. Prior counts in trailing 1-hour, 24-hour and 7-day windows (searchsorted on TransactionDT within each key).
5. Distinct prior P_emaildomain, addr1 and DeviceInfo values.

F1 = F0 + card1 aggregates. F2 = F1 + uid aggregates. Rows with no prior history get NaN. Use vectorised groupby operations (cumcount, cumsum minus the current row, diff); avoid per-row Python loops on 590k rows.

**Done when**

- [ ] A test samples 200 rows, recomputes every aggregate by brute force from `df.iloc[:row_position]`, and matches within 1e-5.
- [ ] Each key's first transaction has count 0 and NaN mean.
- [ ] `reports/feature_dictionary.md` lists each new column, its definition, and why it might signal fraud.

---

## Step 9: Tree models across F0 to F2

**Role:** D

**Files:** `src/models/lgbm.py`, `src/models/rf.py`, `notebooks/03_trees.ipynb`

**Do**

1. LightGBM: tune num_leaves, learning_rate, min_child_samples and scale_pos_weight with time CV and early stopping on each fold's validation part. Use a small grid or random search (about 20 configs) per feature set.
2. Random Forest: tune max_depth and min_samples_leaf with n_estimators fixed at 300 and `class_weight="balanced_subsample"`.
3. For each model and feature set, refit the best config on the train block, score the validation block, and log it. Run seeds 0, 1, 2.
4. Run LR on F1 and F2 the same way as Step 6.

**Done when**

- [ ] The log has validation rows for LR, RF and LightGBM on each of F0, F1, F2, three seeds each.
- [ ] Best configs are saved to `config/best_params.yaml`.
- [ ] Runtime per fit is in PROGRESS.md.

---

## Step 10: MLP across F0 to F2

**Role:** A

**Files:** `src/models/mlp.py`, `notebooks/04_mlp.ipynb`

**Do**

1. PyTorch MLP on `linear_pipeline` output: 2 to 3 hidden layers with ReLU and dropout, Adam, `BCEWithLogitsLoss(pos_weight=neg/pos)` from training rows, mini-batches of 2048.
2. Early stopping on validation PR-AUC (patience 5 epochs).
3. Tune hidden size, dropout and learning rate over a small grid with time CV; refit and log on validation for seeds 0, 1, 2.

**Done when**

- [ ] The log has MLP validation rows for F0, F1, F2, three seeds each.
- [ ] Training curves (loss and validation PR-AUC by epoch) are saved to `reports/figures/`.
- [ ] Best config added to `config/best_params.yaml`.

---

## Step 11: Grid summary

**Role:** C

**Files:** `src/summarise.py`, `reports/grid_validation.md`

**Do**

1. Build the 4 x 3 table of validation PR-AUC (mean ± std over seeds) from `results/log.csv`. Repeat for AUROC and recall_top1pct.
2. Add two derived rows: gain from LR to LightGBM per feature set, and gain from F0 to F2 per model.

**Done when**

- [ ] All 12 cells are filled from the log, with no manual entry.
- [ ] The table regenerates by running one command, recorded in PROGRESS.md.

---

## Step 12: Leakage check

**Role:** C

**Files:** `notebooks/05_leakage.ipynb`, `reports/leakage_check.md`

**Do**

1. Using F2 and each model's best config from Step 9, run LR and LightGBM under three schemes on the train block: random stratified 5-fold, StratifiedGroupKFold on uid (5 folds), and time CV.
2. Log every fold with cv_scheme set accordingly.
3. Report mean ± std PR-AUC per model per scheme.

**Done when**

- [ ] The report table has 6 cells, each built from the log.
- [ ] The report states which scheme scored highest and by how much, using the logged numbers.

---

## Step 13: Hyperparameter sweeps (RQ4)

**Role:** D

**Files:** `notebooks/06_sweeps.ipynb`, `reports/figures/sweep_*.png`

**Do**

For each model, start from its best F2 config and vary one parameter at a time (others fixed), scoring validation PR-AUC:

- LightGBM: num_leaves, learning_rate, min_child_samples, scale_pos_weight
- LR: C
- RF: max_depth, n_estimators
- MLP: hidden size, dropout, learning rate

Use 5 to 7 values per parameter, spanning at least one order of magnitude where the parameter allows.

**Done when**

- [ ] One plot per parameter, PR-AUC against value, with the best config marked.
- [ ] Every point on every plot has a row in the log.

---

## Step 14: Feature importance (RQ1)

**Role:** B

**Files:** `notebooks/07_importance.ipynb`, `reports/figures/`

**Do**

1. LightGBM gain importance on F2, top 30 features.
2. SHAP values on a 20,000-row sample of the validation block: summary plot and dependence plots for the top 5 features.

**Done when**

- [ ] Plots are saved and each top-5 feature has a one-line plain description in `reports/feature_dictionary.md`.

---

## Step 15: Final test scoring

**Role:** C

**Run only after the user confirms the experiment freeze.**

**Files:** `notebooks/08_test.ipynb`, `reports/test_results.md`

**Do**

1. For each of the 12 grid cells, take the best config fit on the train block (as in Steps 6, 9 and 10) and apply the threshold picked on validation. Score the test block for seeds 0, 1, 2 and log with split = test.
2. Split test PR-AUC by returning vs new customers (uid seen in the train or validation block, or not).

**Done when**

- [ ] The 4 x 3 test table and the returning vs new table are built from the log.
- [ ] The validation-to-test change for each cell is reported.
