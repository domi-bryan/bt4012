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

## How this file is organised

On 2026-11-14 the group moved from the F0–F2 feature sets and four models to the `features_v2` table, its G0–G3 feature sets, and six models. This file was rewritten for that plan on 2026-11-17.

- **Part 1 (Steps 0–5, 7):** foundation. Still in force.
- **Part 2 (Steps 6, 8–12):** the first pass on F0–F2. Finished and superseded; kept as a short record.
- **Part 3 (Steps 13–19):** the G0–G3 build. This is the current plan.

Steps 0–12 keep their numbers, so earlier `PROGRESS.md` entries, the README and the `results/step*_manifest.json` files still point at the right step. The earlier Steps 13–15 were never started and are replaced:

| Earlier step | Now |
| --- | --- |
| 13: one-at-a-time hyperparameter sweeps (RQ4) | Section 5.5 of each model notebook (Step 16), read from the tuning runs. No extra sweeps are run. |
| 14: LightGBM gain and SHAP (RQ1) | Section 6 of each model notebook (Step 16) and the cross-model table in Step 17, using permutation importance. SHAP is not used. |
| 15: final test scoring | Step 19 |

Older text that says "Step 13", "Step 14" or "Step 15" (in `PROGRESS.md` entries for Steps 4–12, `reports/feature_dictionary.md` and `reports/uid_check.md`) uses the earlier meaning.

Most G0–G3 work so far was logged in `PROGRESS.md` under descriptive titles without step numbers. Each Part 3 step lists those entries under **Earlier entries**. Check a step's Done-when list against them before starting it. New entries use the `## Step N:` heading again.

---

# Part 1: Foundation

## Step 0: Environment and data check

**Role:** A

**Files:** `requirements.txt`, `.gitignore`, `src/__init__.py`, `tests/`, `PROGRESS.md`, `results/`

**Do**

1. Create the folders in `Context.md` (Repo layout) plus `tests/`, `reports/` and `data/processed/`.
2. Write `requirements.txt`: pandas, pyarrow, numpy, scikit-learn (>=1.4), lightgbm, xgboost, catboost, torch, shap, matplotlib, seaborn, pytest, pyyaml, and nbformat, nbconvert, ipykernel to run the notebooks.
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

A later audit and cleaning pass (`src/data_audit.py`, `src/clean_data.py`) validates this table and saves an identical copy as `data/processed/cleaned.parquet`; no values were changed.

---

## Step 2: Data profile and first V-column proposal

**Role:** A

**Files:** `notebooks/01_eda.ipynb` (section 3; was `01_profile.ipynb`), `src/data_profile.py`, `reports/data_profile.md`, `config/features.yaml`

**Do**

1. Report per column: dtype, % missing, distinct count, fraud rate when missing vs present. Sort by % missing.
2. Group V1-V339 by identical missing-count pattern. Within each group, propose a subset to keep: drop one of any pair with |Pearson r| > 0.9 on training rows (first 80% by time), keeping the column with more distinct values.
3. Write the proposed keep and drop lists to `config/features.yaml` under `v_columns_proposed`.

**Done when**

- [ ] `reports/data_profile.md` covers every column.
- [ ] Every V column appears in exactly one of keep or drop.
- [ ] The proposal is summarised for the user (groups, columns kept, columns dropped).

**Outcome:** the r > 0.9 list was never approved. Steps 5–12 used it provisionally. The group's agreed rule is the same pruning at r = 0.75 (Step 13), and the G0–G3 runs use only that. `config/features.yaml` is kept for the F0–F2 scripts.

---

## Step 3: Splits

**Role:** C

**Files:** `src/split.py`, `tests/test_split.py`

**Do**

1. `holdout_indices(df)`: on the time-sorted table, train = rows 0 to 472,431, validation = next 59,054, test = last 59,054. Return row-index arrays.
2. `time_cv_folds(df, train_idx)`: bin training rows by `TransactionDay // 30` (relative to the first day). Fold k trains on bins 0..k-1 and validates on bin k, for every k >= 1. Return a list of (train_idx, val_idx). This gives 4 folds.
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
3. `recall_top1pct`: share of all fraud rows that fall in the top 1% of scores. `precision_top1pct` is derived from it (recall × fraud rows ÷ rows in the top 1%), so it needs no log column.
4. `log_run(**fields)` appends one row to `results/log.csv` with the columns listed in `Context.md`, creating the file with a header if missing. The fold number, the validation threshold and fit details (`_fit`, `es_setup`) go inside `params`.

**Done when**

- [ ] pr_auc matches `sklearn.metrics.average_precision_score` on a test case.
- [ ] A random-score test case gives pr_auc close to the positive rate.
- [ ] `log_run` called twice gives a file with one header and two rows.

---

## Step 5: Preprocessing pipelines

**Role:** A

**Files:** `src/data.py` (extend), `src/catboost_pipeline.py`, `tests/test_pipelines.py`

**Do**

1. `TreePipeline`: keeps NaN and `category` dtypes for LightGBM and XGBoost. A variant for Random Forest fills NaN with -999 and encodes categoricals as integer codes. `CatBoostPipeline` subclasses it; CatBoost then uses only its built-in one-hot encoding (no target statistics).
2. `LinearPipeline` (LR and MLP): numeric columns get median imputation, a missing-indicator column, and standard scaling; categoricals with <= 20 levels get one-hot encoding, the rest get frequency encoding.
3. Every pipeline exposes `fit(train_df)` and `transform(df)`, and is fit on training rows only. In CV it is refit on each fold's training part.
4. Every pipeline takes an optional `columns` list. The G0–G3 runs pass the feature set's columns. Without it, the pipeline falls back to the F0–F2 behaviour and reads the V list from `config/features.yaml`.

**Done when**

- [ ] `LinearPipeline` output has no NaN on validation rows.
- [ ] A test confirms medians and scaler means equal those computed on training rows alone.
- [ ] A test confirms a pipeline given `columns` uses exactly those columns.
- [ ] Output column counts for each pipeline are in PROGRESS.md.

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

The same uid definition is rebuilt in `src/domain_features.make_uid` for Step 14.

---

# Part 2: First pass on F0–F2 (finished, superseded)

These steps ran between 2026-10-18 and 2026-10-22 on the F0–F2 feature sets (F0 raw, F1 + card1 aggregates, F2 + uid aggregates) with LR, RF, LightGBM and the MLP, plus an optional CatBoost run. Their full results are in `PROGRESS.md` and their rows stay in `results/log.csv` under feature sets F0, F1 and F2. They are not part of the current plan.

| Step | What it built | Used by the G0–G3 build? |
| --- | --- | --- |
| 6: LR baseline on F0 | `src/models/logreg.py`, `src/experiment.py` (`time_cv`, `best_params`, `fit_validate`) | Yes: the model module and the experiment helpers. The C grid changed (Step 15). |
| 8: Past-only aggregates F1, F2 | `src/features.py` (`past_aggregates`, `build_feature_set`), `reports/feature_dictionary.md` | No. G2 and G3 history features come from `src/domain_features.py` (Step 14). |
| 9: Tree models on F0–F2 | `src/models/rf.py`, `src/models/lgbm.py`, `src/train_trees.py`, `config/best_params.yaml` | The model modules only. Grids, subsampling and early stopping changed (Step 15). |
| 10: MLP on F0–F2 | `src/models/mlp.py`, `src/train_mlp.py` | The model module only. Search and early stopping changed (Step 15). |
| 11: Grid summary | `src/summarise.py`, `reports/grid_validation.md` (4 x 3 table on F0–F2) | No. Replaced by `notebooks/10_model_comparison.ipynb` (Step 17). |
| 12: Leakage check on F2 | `src/leakage.py`, `reports/leakage_check.md`, `notebooks/11_leakage_check.ipynb` | The code is reused by Step 18. |
| Optional CatBoost on F0–F2 (no step number) | `src/train_catboost.py`, `src/models/catboost.py`, `config/catboost_best_params.yaml`, `reports/catboost_sensitivity_F2.*` | The model module only. |

Rules for this part:

- Do not rerun `src.train_trees`, `src.train_mlp` or `src.train_catboost`. `src.leakage` now runs the G3 check (Step 18) and no longer covers F2. The manifests of those three scripts fingerprint model code that has since changed, so they refuse to resume, and the old LightGBM and CatBoost settings can no longer be reproduced.
- Their notebooks are in `notebooks/archive/` and are not meant to be rerun.
- The report uses G0–G3 results only. `src/summarise.py` and `reports/grid_validation.md` stay as the F0–F2 record and are not regenerated.

---

# Part 3: The G0–G3 build

Feature sets (cumulative, defined in `config/features_v2.yaml`):

| Set | Adds | Inputs |
| --- | --- | --- |
| G0 | Raw columns chosen in Step 14 | 197 |
| G1 | 27 row-level features built from the transaction's own values | 224 |
| G2 | 10 card and device history features | 234 |
| G3 | 18 customer (uid) history features | 252 |

Models: logistic regression, random forest, LightGBM, XGBoost, CatBoost, MLP. The core experiment is a **6 models x 4 feature sets grid** (24 cells) on the validation block, three seeds each.

## Step 13: V-column threshold

**Role:** A

**Files:** `notebooks/03_v_threshold.ipynb`, `src/feature_selection.py` (`prune_v`, `V_THRESHOLD`), `tests/test_feature_selection.py`

**Do**

1. Use the Step 2 pruning rule: group V columns by identical missing count on training rows; inside a group, visit columns from most to fewest distinct values and drop a column if |Pearson r| with one already kept is above the threshold.
2. Sweep the threshold from 0.50 to 1.00 in steps of 0.05 (1.00 keeps every V column). For each, score logistic regression (C = 0.1, balanced class weights) on the raw columns with the 4 time-CV folds. Pruning and preprocessing are fit on each fold's training rows. Log every fold with `feature_set` `raw_vr050` to `raw_vr100`.
3. Choose the most aggressive threshold whose mean PR-AUC stays within noise of no pruning, and set `V_THRESHOLD`.

**Done when**

- [ ] The log has 4 rows for each of the 11 thresholds (44 rows).
- [ ] Section 6 of the notebook states the decision with a table read from the log. The decision is r = 0.75.
- [ ] The group's agreement is recorded in the notes of `config/features_v2.yaml`.

**Earlier entries:** none of its own. Mentioned in "EDA notebooks combined" and "Repo cleanup before the G0–G3 model update".

---

## Step 14: Feature table and the G0–G3 sets

**Role:** B

**Files:** `src/domain_features.py`, `src/feature_selection.py`, `src/feature_sets.py`, `notebooks/02_feature_engineering.ipynb`, `config/features_v2.yaml`, `data/processed/features_v2.parquet` (gitignored), `tests/test_domain_features.py`, `tests/test_v2_models.py`

**Do**

1. Take constants from the train block only: the most common `addr2` and `card3`, and the V-column groups. Build uid as in Step 7.
2. Build the `fe_` columns by family: amount, time, normalised D columns, novelty, velocity, identity, location and card, missingness, frequency. Every history feature uses only rows earlier in the sort order.
3. Keep an engineered feature only if the EDA supports it and its link to fraud holds between the first and last 30 days of the train block (`feature_selection.time_stability`). Record the features left out and why.
4. Choose the raw inputs on training rows only. Exclude TransactionID, TransactionDT, TransactionDay, uid, `id_12`, `id_27`, V columns pruned at r = 0.75 (Step 13), and raw columns whose link to fraud reversed or faded over time.
5. Define the four cumulative sets in the table above.
6. Save `features_v2.parquet` with every row in the original time order (uid and excluded raw columns included for analysis) and write the column lists, groups, exclusions and constants to `config/features_v2.yaml`.
7. `feature_sets.load_features()` returns the train and validation blocks only, so no model notebook can read the test block. `feature_list("G0")` to `("G3")` give each set's columns.

**Done when**

- [ ] The sets are cumulative with 197, 224, 234 and 252 inputs, and none contains an identifier, raw time, uid or the target (test).
- [ ] A test confirms a row's features do not change when later rows are added.
- [ ] The past-only helpers (prior count, mean, max, previous value, distinct values, trailing windows, frequencies) pass their tests against hand-worked or brute-force values.
- [ ] `load_features()` returns no test-block rows (test).
- [ ] The notebook is executed top to bottom and committed with outputs.

**Earlier entries:** none of its own. `src/feature_sets.py` is recorded in "Model notebooks on the G0–G3 feature sets".

---

## Step 15: Shared run code and the model notebook template

**Role:** C, with each model's owner for its module

**Files:** `src/experiment.py`, `src/models/*.py` (adds `xgb.py`; `GRID` in each), `src/segments.py`, `src/importance.py`, `src/metrics.py`, `notebooks/04_logreg.ipynb` to `09_mlp.ipynb` (one template), `tests/test_v2_models.py`

**Do**

1. `tune_and_validate`: tune the model's `GRID` with time CV (seed 0) on G3 only (`TUNE_ON = "G3"`), choose the settings with the highest mean PR-AUC, then refit them on the whole train block with seeds 0, 1, 2 and score the validation block on each of G0–G3. Pick the F1 threshold on validation and store it in `params.threshold`. Skip anything already logged, so a run can be stopped and resumed.
2. Search budgets:
   - LR: grid of 11 values of C from 0.001 to 100; balanced class weights.
   - RF: grid of 12, max_depth {12, 24, unlimited} x min_samples_leaf {5, 20} x max_features {sqrt, 0.3}; 300 trees; `class_weight="balanced_subsample"`.
   - LightGBM: 20 random settings over num_leaves, learning_rate, min_child_samples, reg_lambda, scale_pos_weight.
   - XGBoost: 20 random settings over max_depth, learning_rate, min_child_weight, reg_lambda, scale_pos_weight.
   - CatBoost: 20 random settings over depth, learning_rate, l2_leaf_reg, scale_pos_weight.
   - MLP: 16 random settings over hidden, dropout, lr, layers, weight_decay, pos_weight; batch 2,048, patience 5, at most 40 epochs.
3. Keep the three boosting models on equal terms: 80% of rows and 80% of columns per tree (fixed), L2 on leaf weights tuned over {1, 3, 10}, at most 3,000 trees, early-stopping patience 50.
4. Early stopping (LightGBM, XGBoost, CatBoost, MLP) never uses the rows a model is scored on. A tuning fit trains on the first 90% (by time) of its training part and stops on the last 10%. Validation, fold-score and final fits take a second fit on the whole training part with the stopping point found. Rows from this setup log `es_setup: 2`. `experiment.current_runs` leaves older early-stopping rows out of every summary, and those runs are redone on a rerun.
5. Save validation scores, and the time-CV fold scores of the chosen settings (`SAVE_OOF`), to `results/preds/` (gitignored; shared on the group drive).
6. Notebook template: 1 settings, 2 data, 3 model input, 4 tune and validate, 5.1 tuning table, 5.2 validation table and precision-recall curves, 5.3 what each feature group adds, 5.4 known vs unknown customers with 95% bootstrap intervals, 5.5 how each tuned setting affects time-CV PR-AUC (RQ4, read from the log with `setting_sensitivity`), 6 feature importance (permutation on 20,000 validation rows, plus the model's built-in measure; saved to `results/importance/<model>_<set>.csv`), 7 findings.
7. Each notebook imports only its own model library first and trains one model per kernel. On macOS, LightGBM, XGBoost and PyTorch each ship their own OpenMP library and can crash when combined.

**Done when**

- [ ] `tests/test_v2_models.py` passes, covering: resume skips logged fits; early-stopping rows are the last 10% of each training part and never overlap scored rows; the refit trains on the whole training part with the stopping point from the first fit; older-setup rows are rerun and left out of summaries; `fit_final` reproduces the logged validation PR-AUC.
- [ ] Every model runs end to end through `tune_and_validate` on a sample of the real `features_v2`, logging to a scratch file, with `results/log.csv` unchanged.

**Earlier entries:** "Model notebooks on the G0–G3 feature sets", "Known vs unknown customer segments", "Model notebook results sections and precision in the top 1%", "Same subsampling for the three boosting models", "Fair early stopping and comparable search budgets", "Precision–recall curves and feature importance", "Same L2 regularisation search for the three boosting models", "RQ4: setting sensitivity from the tuning runs", "Early stopping refit and a higher tree cap".

---

## Step 16: Run the six models on G0–G3

Each notebook is one unit of work: finish it, check its Done-when list, write its `PROGRESS.md` entry (`## Step 16: <model> on G0–G3`), then stop.

| Notebook | Model | Role | Rows to expect (CV + validation) | Earlier entries |
| --- | --- | --- | --- | --- |
| `04_logreg.ipynb` | Logistic regression | C | 44 + 12 | none |
| `05_random_forest.ipynb` | Random forest | D | 48 + 12 | "Random forest and MLP on G0–G3", "Findings for random forest and MLP" |
| `06_lightgbm.ipynb` | LightGBM | D | 80 + 12 | "LightGBM on G0–G3" |
| `07_xgboost.ipynb` | XGBoost | extra model, no role in `Context.md` | 80 + 12 | "XGBoost on G0–G3" (older setup), "XGBoost rerun with the refit and the 3,000-tree cap" |
| `08_catboost.ipynb` | CatBoost | extra model, no role in `Context.md` | 80 + 12 | "CatBoost on G0–G3" |
| `09_mlp.ipynb` | MLP | A | 64 + 12 | "Random forest and MLP on G0–G3", "Findings for random forest and MLP" |

**Do**

1. Run the notebook top to bottom with its default settings: `FEATURE_SETS` G0–G3, `TUNE_ON = "G3"`, `GRID = model.GRID`, `SEEDS = [0, 1, 2]`, `SAVE_OOF = True`, the default `LOG_PATH`. A quick trial with a smaller grid logs to a separate file.
2. Write section 7 (findings), quoting only numbers from the notebook's own tables: which set scores best and what each step G0 → G1 → G2 → G3 adds; known vs unknown customers; whether the chosen settings sit at the edge of the grid; limitations.
3. Commit the executed notebook, the new rows of `results/log.csv` and `results/importance/<model>_G3.csv`. Share the `results/preds/<model>_*` files on the group drive.

**Done when (per notebook)**

- [ ] Under the current setup the log has one CV row per setting per fold on G3 and one validation row per seed on each of G0–G3, each exactly once (counts in the table above).
- [ ] No row in the log has split = test.
- [ ] Every code cell ran without error and the outputs are committed.
- [ ] `results/importance/<model>_G3.csv` exists, and `results/preds/` holds 12 validation files and 4 fold-score files for the model.
- [ ] Section 7 is written and every number in it matches a table in the notebook.
- [ ] Boosting models only: the number of fits that stopped within 50 trees of the 3,000 cap is recorded.
- [ ] The `PROGRESS.md` entry gives the chosen settings, their mean time-CV PR-AUC, validation PR-AUC per set and the run time.

---

## Step 17: Model comparison

**Role:** C

**Files:** `notebooks/10_model_comparison.ipynb`, `reports/feature_dictionary.md`

**Do**

1. Run after all six notebooks of Step 16 are complete. The notebook reads `results/log.csv` through `experiment.current_runs` and refits nothing.
2. Build the 6 x 4 validation table (mean ± sample sd over seeds 0, 1, 2) for PR-AUC, AUROC, recall and precision in the top 1%, and precision and recall at the threshold.
3. Show what each feature group adds for every model.
4. Report PR-AUC for known vs unknown customers with 95% bootstrap intervals on the validation block, and per fold on the saved time-CV fold scores (where "known" means the uid appears in that fold's training months).
5. Plot precision-recall curves of all models on G3, and the cross-model table of permutation-importance ranks and totals by feature group.
6. Write section 6 (findings): the best model and set and its margin over the next best (RQ2); the gain between models against the gain from G0 to G3, with G2 → G3 as the customer-profile step (RQ3); how known and unknown customers differ.

**Done when**

- [ ] All 24 cells are filled from the log with no manual entry, and the "what has been run" table shows nothing missing.
- [ ] Every code cell ran without error and the outputs are committed.
- [ ] Section 6 is written and every number in it matches a table in the notebook.
- [ ] Each of the five inputs with the best average permutation rank has a one-line plain description in `reports/feature_dictionary.md` (carried over from the earlier Step 14).

**Earlier entries:** none for the run. The notebook's sections were built in the Step 15 entries.

---

## Step 18: Leakage check on G3

**Role:** C

This replaces the Step 12 check on F2 as the report's leakage table.

**Files:** `src/leakage.py`, `tests/test_leakage.py`, `reports/leakage_check.md`, `notebooks/11_leakage_check.ipynb`

**Do**

1. Using G3 and the chosen settings of LR and LightGBM from Step 16 (read from the G3 validation rows of the log), run two more schemes inside the train block with seed 0: random stratified 5-fold and StratifiedGroupKFold on uid (5 folds). Reuse the Step 16 time-CV rows of the same settings.
2. Fit as in tuning: refit the pipeline on each fold's training part, and let LightGBM early-stop on the last 10% of that training part. Step 12 stopped on the fold's validation part, so its F2 numbers are not directly comparable.
3. Log every fold with `feature_set` G3 and `cv_scheme` `random` or `group_uid`.
4. Rebuild `reports/leakage_check.md` and the notebook from the G3 rows: mean ± std PR-AUC per model per scheme. The F2 rows stay in the log.

**Done when**

- [ ] The G3 report table has 6 cells, each built from the log, with exactly one row per fold (5, 5, 4).
- [ ] The report states which scheme scored highest and by how much, using the logged numbers.
- [ ] Group folds share no uid between the two sides of a fold (checked on the real G3 folds).
- [ ] No row in the log has split = test.

---

## Step 19: Final test scoring

**Role:** C

**Run only after the user confirms the experiment freeze.** After the freeze no setting, threshold, feature set or model choice changes.

**Files:** `notebooks/12_test.ipynb`, `reports/test_results.md`, `src/feature_sets.py` (a loader that includes the test block, used by this step only)

**Do**

1. For each of the 24 grid cells and seeds 0, 1, 2: refit the chosen settings on the train block exactly as in the validation run (`experiment.fit_final`, two fits for the early-stopping models), score the test block, apply the threshold stored in that cell's validation row, and log with split = test.
2. Split test PR-AUC by known vs unknown customers with 95% bootstrap intervals (`src/segments.py`). A customer is known if the uid appears in the train block, the rows the scored model was fit on.

**Done when**

- [ ] The log has exactly 72 test rows, one per cell and seed.
- [ ] The 6 x 4 test table and the known vs unknown table are built from the log.
- [ ] The validation-to-test change for each cell is reported.
