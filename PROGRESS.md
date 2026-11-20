# Progress log: BT4012 Group 11 fraud detection (one entry per finished step)

## Optional CatBoost implementation (2026-11-05)
- Base: open team PR #1, `jarren/steps-3-11`, commit `13a1e65628c87c573c76c4e6a56be702381c5aae`. Isolated branch `damien/catboost`; original main worktree preserved.
- Built: `src/catboost_pipeline.py`, `src/models/catboost.py`, `src/train_catboost.py`, `tests/test_catboost.py`, `requirements-catboost.txt`, `CATBOOST.md`; README links the extension. Shared builders, splitters, experiment functions, metrics and logging are reused without changes. No shared experiment results or best-parameter settings were changed.
- Checks: CatBoost suite 11 passed. Full suite 81 passed, 11 skipped for absent real data, using the existing PyTorch `libomp` through process-local `DYLD_LIBRARY_PATH` to satisfy the shared LightGBM import. `pip check`, CLI help, missing-data diagnostic, fresh-process shared-model training and diff whitespace checks passed. Tests inspect the CatBoost JSON for absence of target statistics, preserve numeric card1, exclude raw uid, verify fold-local categories/weights, resume exact fits and reconstruct sensitivity plots from complete logged seeds.
- Numbers: implementation provides deterministic 20-config time-CV tuning for each of F0/F1/F2, validation seeds 0/1/2, and 21 one-parameter F2 sensitivity configurations (63 fits). No real-data CatBoost scores are available or claimed.
- Open: raw/processed data is absent, so full tuning and sensitivity experiments remain to run. Upstream V-column approval is pending. The locked four-model summary, leakage comparison and final-test scoring do not yet include CatBoost. PR stacking/retargeting instructions and the conservative internal one-hot categorical policy are documented in `CATBOOST.md`. (2026-11-14: `CATBOOST.md` was folded into README section 11 and `requirements-catboost.txt` into `requirements.txt`; the PR was merged as #4.) This entry does not mark the team's Step 13 complete.

## Step 0: Environment and data check (2026-10-01)
- Built: `requirements.txt`, `.gitignore` (added `data/raw/`, `data/processed/`, `.venv/`, caches), `src/__init__.py`, `src/models/__init__.py`, `results/.gitkeep`, folders `src/ src/models/ tests/ notebooks/ reports/ reports/figures/ config/ results/ data/processed/`, `.venv/` (Python 3.13.5).
- Checks:
  - Every package imports in `.venv`: OK (pandas 3.0.6, pyarrow 25.0.1, numpy 2.5.3, scikit-learn 1.9.1, lightgbm 4.7.0, torch 2.14.1, shap 0.52.0, matplotlib 3.11.2, seaborn 0.13.2, pytest 9.1.1, pyyaml 6.0.3).
  - Raw CSVs exist; `train_transaction.csv` has 590,540 data rows, `train_identity.csv` has 144,233 (counted with pandas). isFraud = 1 in 20,663 transaction rows.
- Numbers: 590,540 / 144,233 / 20,663 fraud.
- Open: none. Context.md was added afterwards and its repo layout matches the folders created. Requirements are unpinned; consider pinning versions.

## Step 1: Load, join, save (2026-10-01)
- Built: `src/data.py` (`load_raw`, `save_joined`, `load_joined`; run `python -m src.data` to rebuild), `tests/test_data.py`, `data/processed/joined.parquet` (gitignored).
- Checks:
  - 590,540 rows, 20,663 isFraud = 1, 144,233 has_identity = 1: pass (asserted in `test_real_joined_table`).
  - TransactionDT non-decreasing down the saved file: pass (same test).
  - Peak memory and file size recorded below: pass.
  - Extra: no object-dtype columns remain; 4 tests pass.
- Numbers: shape (590540, 436); in-memory 0.91 GB; peak RSS 1.09 GB; parquet 80 MB; build time 19 s.
- Open: the string columns among id_12-id_38 were identified by inspecting the header and sample rows (id_12, 15, 16, 23, 27-31, 33-38); the test confirms none were left as object. TransactionID is int32, isFraud int8, TransactionDT int32, TransactionDay int16, all other numerics float32.

## Step 2: Data profile and V-column pruning proposal (2026-10-01)
- Built: `src/data_profile.py` (run `python -m src.data_profile` to regenerate), `notebooks/01_profile.ipynb` (executed), `reports/data_profile.md`, `config/features.yaml`, `tests/test_data_profile.py`. Added nbformat, nbconvert, ipykernel to `requirements.txt` to run the notebook.
- Checks:
  - `reports/data_profile.md` covers every column: pass (436 columns, asserted in test).
  - Every V column is in exactly one of keep or drop: pass (229 + 110 = 339, no overlap, asserted in test).
  - Proposal summarised for the user: done in the report to the user; the approval stop is pending.
  - Extra: kept columns within each group have |r| <= 0.9 (test). 7 tests pass; the new tests take about 3.5 minutes because they recompute correlations.
- Numbers: train block 472,432 rows, fraud rate 3.51%. V1-V339 fall into 14 groups (identical missing count; the missing-row mask is also identical within every group). Keep 229, drop 110.
- Open: the group must approve `v_columns_proposed` in `config/features.yaml` before Step 5 reads it. Profile and correlations use the train block only (rows 0 to 472,431), so validation and test rows did not influence the proposal. The prune rule is greedy: columns are visited by distinct count (descending, ties in V order) and a column is dropped if |r| > 0.9 with one already kept.

## Local data-quality audit (2026-10-04)
- Scope: user-approved audit before cleaning and EDA; this does not complete build Step 3 or approve V-column pruning.
- Built: `src/data_audit.py`, `reports/data_quality_audit.json`, `reports/data_quality_columns.csv`. Reproduce with `.venv/bin/python -m src.data_audit`.
- Checks: all 14 structural checks pass, including unique/nonmissing transaction and identity keys, raw core-column agreement, chronological ordering, binary labels, derived day and identity flags, and missing identity fields for unmatched rows. `.venv/bin/python -m pytest -q tests/test_data.py`: 4 passed.
- Numbers: saved join 590,540 rows x 436 columns. Quality statistics use only the first 472,432 rows: 414 columns have missing values; 12 exceed 90% missing; no entirely missing or constant-nonmissing columns; no infinite values, blank/padded categorical cells, invalid amounts, or negative transaction times. Zero duplicate full rows; 2 repeated predictor rows beyond the first (4 rows total across repeated groups), excluding TransactionID and isFraud. Training fraud count 16,599 (3.5135%). All numbers come from the audit JSON/CSV; no model experiment ran.
- Data changes: none; raw files and the existing joined parquet were not modified. No commits made.
- Open: awaiting user approval for the next cleaning step. Recommend retaining missingness, distinct transaction IDs, and extreme amounts; fit imputation/scaling/encoding on training data only when preprocessing is implemented. V-column pruning remains unapproved.

## Local conservative cleaning (2026-10-04)
- Scope: user approved preserving the audited table and creating a separate validated cleaned dataset. This does not implement fitted preprocessing or approve V-column pruning.
- Built: `src/data.py` validation/save/load helpers, `src/clean_data.py`, `data/processed/cleaned.parquet` (gitignored), `reports/cleaning_summary.json`; extended `tests/test_data.py` and documented the command in `README.md`. Reproduce with `.venv/bin/python -m src.clean_data`.
- Checks: full-data validation and exact Parquet roundtrip passed, preserving values, missingness, dtypes, column order, and row order. `.venv/bin/python -m pytest -q tests/test_data.py`: 16 passed, including rejection of corrupt inputs before writing and protection against overwriting raw/joined data. `git diff --check` passed. SHA-256 comparisons confirm both raw CSVs, the original joined parquet, and feature configuration are unchanged.
- Numbers: 590,540 rows x 436 columns; 115,523,073 missing cells preserved; 0 rows removed, 0 columns removed, 0 values changed (from `reports/cleaning_summary.json`). No confirmed value errors required repair.
- Open: awaiting user approval to run EDA locally on the training block. No EDA or model fitting ran in this step; no commits made.

## Local training EDA (2026-10-04)
- Scope: user-approved exploratory analysis of the cleaned training block only. Validation and test rows are excluded from all summaries and plots; no preprocessing or V-column pruning decisions are applied.
- Built: `src/eda.py`, executed `notebooks/01_eda.ipynb`, `reports/eda_findings.md`, 12 CSV tables and `summary.json` in `reports/eda/`, and 7 PNG figures in `reports/figures/`; reproduction commands added to `README.md`.
- Checks: all 7 code cells executed with zero errors. Category, amount-band, class, and weekly tables reconcile to 472,432 transactions and 16,599 fraud labels; missingness totals reconcile across all 436 columns. Email display groups also preserve the population and fraud counts. Notebook schema validates; all 7 plots visually inspected and all report image links resolve. SHA-256 confirms both raw CSVs, joined and cleaned parquet files, and `config/features.yaml` are unchanged. Jupyter required permission for its local kernel connection; execution then succeeded.
- Numbers: training fraud rate 3.5135%; ProductCD C 11.33% versus W 2.07%; identity present 7.55% versus absent 2.13%; 44.86% of training cells missing; 414 columns have missingness, 12 exceed 90%. Complete weekly fraud rates range from 2.07% to 5.07%. Numbers are generated from descriptive EDA tables, not model experiment scores.
- Approval: user approved the completed EDA step on 2026-10-04. The requested local cleaning and EDA work is complete. No models fitted, no test scoring, and no commits made. V-column pruning remains unapproved.

## Step 3: Splits (2026-10-18)
- Built: `src/split.py` (`holdout_indices`, `time_cv_folds`, `random_cv_folds`, `group_cv_folds`, `split_summary`; run `.venv/bin/python -m src.split` to print the table below), `tests/test_split.py`.
- Checks:
  - Holdout blocks are disjoint and cover every row; max TransactionDT of train <= min of validation <= min of test: pass (toy data with tied timestamps, and the real table in `test_real_holdout_and_time_folds`). On the real table the boundaries are strict (12,192,842 < 12,192,900; 13,990,904 < 13,990,941).
  - Each time fold's training rows are all earlier than its validation rows: pass (strictly earlier by TransactionDT). Tests also confirm the window expands (fold k+1 train = fold k train + val) and that fold 4's train + val is the whole train block.
  - Group folds never place one group on both sides of a fold: pass on toy groups, and on the real train block using card1 as the group key, since uid comes in Step 7.
  - Row count and fraud rate per block and per fold: below.
  - Extra: `random_cv_folds` is reproducible for a fixed seed and covers the train block once. `y` and `groups` must be aligned with `train_idx`; misaligned input raises an error. Full suite: 27 passed (7 min, mostly the Step 2 correlation tests).
- Numbers (from `python -m src.split`):

  | Block | Rows | Fraud | Fraud rate | Days |
  | --- | --- | --- | --- | --- |
  | train | 472,432 | 16,599 | 3.5135% | 1-141 |
  | validation | 59,054 | 1,851 | 3.1344% | 141-161 |
  | test | 59,054 | 2,213 | 3.7474% | 161-182 |
  | fold1 train / val | 134,339 / 89,399 | 3,401 / 3,577 | 2.5317% / 4.0012% | 1-30 / 31-60 |
  | fold2 train / val | 223,738 / 92,189 | 6,978 / 3,724 | 3.1188% / 4.0395% | 1-60 / 61-90 |
  | fold3 train / val | 315,927 / 98,615 | 10,702 / 3,898 | 3.3875% / 3.9527% | 1-90 / 91-120 |
  | fold4 train / val | 414,542 / 57,890 | 14,600 / 1,999 | 3.5220% / 3.4531% | 1-120 / 121-141 |

- Open: 5 bins give 4 time folds, matching Context.md's "about 4 folds". Fold 4 validates on a partial 21-day bin (days 121-141). Calendar day 141 is split between train and validation, and day 161 between validation and test; timestamps never overlap, but per-day features would span two blocks. The seed for random and group folds is a required keyword argument. Test-block fraud counts are only row-count summaries; no model has been scored on it.

## Step 4: Metrics and results log (2026-10-18)
- Built: `src/metrics.py` (`evaluate`, `pick_threshold`, `recall_top1pct`), `src/results.py` (`log_run`, `COLUMNS`, `LOG_PATH`), `tests/test_metrics.py`.
- Checks:
  - pr_auc matches `sklearn.metrics.average_precision_score` on a test case: pass (identical to 1e-12; AUROC is checked against `roc_auc_score` the same way).
  - A random-score test case gives pr_auc close to the positive rate: pass (200,000 rows, positive rate 0.034855, pr_auc 0.034520; tolerance 0.003).
  - `log_run` called twice gives a file with one header and two rows: pass (header equals the Context.md columns; params round-trip as JSON).
  - Extra: `pick_threshold` beats a 200-point threshold grid on F1. `recall_top1pct` is exact on a hand case. With tied scores, the cutoff group shares the remaining top-1% slots evenly, so the result does not depend on row order. `log_run` rejects unknown fields, a split outside cv/val/test, missing model/feature_set/cv_scheme/split/pr_auc, and a file with a different header; a rejected row writes nothing. 12 new tests pass; split + metrics tests: 20 passed.
- Numbers: test-case values only (pr_auc 0.618557 for both ours and sklearn's; random case above). No training run happened and `results/log.csv` has not been created.
- Open:
  - **Fold column:** the Context.md log columns have no fold column, but Step 6 logs one row per C per fold. For now `log_run` expects the fold inside `params` (e.g. `{"C": 0.1, "fold": 2}`). Adding a `fold` column would make Step 11/12 summaries simpler; that is a Context.md change for the group to agree.
  - **Conventions:** threshold rule is `score >= threshold`. The top 1% is ceil(1% of rows) (591 rows on a 59,054-row block). `author` defaults to `git config user.name` and `date` to today. Precision/recall are left empty when no threshold is passed. Nothing blocks `split="test"` in code; the test-block freeze still relies on the guardrail.

## Step 5: Preprocessing pipelines (2026-10-18) — PROVISIONAL
- Status: user chose to build on the **proposed** V-column list (keep 229, drop 110) before group approval. `config/features.yaml` still says "proposed, awaiting group approval" and was not edited; `load_v_columns()` warns until its status starts with "approved". If the group changes the list, re-run the counts below; no code change needed.
- Built: `src/data.py` extended with `load_v_columns`, `feature_columns`, `TreePipeline` (kind "lgbm" or "rf"), `LinearPipeline`; `tests/test_pipelines.py`.
- Checks:
  - `linear_pipeline` output has no NaN on validation rows: pass (real table: fit on the train block, transform validation, 0 NaN and all finite; also on toy data).
  - Medians and scaler means equal those computed on training rows alone: pass (they match `np.nanmedian`, sklearn `SimpleImputer(median)` and `StandardScaler` fitted on training rows only; a toy validation block with extreme amounts would have moved the mean more than 10x if it had leaked in).
  - Output column counts recorded below: pass.
  - Extra: category levels come from training rows only. The saved table's category dtypes include levels first seen in validation/test (id_30 72 vs 75, id_31 110 vs 130, id_33 212 vs 260, DeviceInfo 1,639 vs 1,786), and those become missing / -999 / all-zero one-hot. LightGBM output has identical category dtypes for train and validation. Tests: 8 new; fast suites 44 passed (`test_data_profile.py` not re-run, its code is unchanged).
- Numbers (fit on the full train block, 472,432 rows):

  | Pipeline | Output columns | Fit + transform train | Train output size |
  | --- | --- | --- | --- |
  | TreePipeline("lgbm") | 322 (NaN and `category` kept) | 1.0 s | 0.53 GB |
  | TreePipeline("rf") | 322 (NaN -> -999, categoricals -> codes) | 0.9 s | 0.57 GB |
  | LinearPipeline | 658 = 291 scaled numeric + 274 missing flags + 87 one-hot (25 categoricals, levels + missing) + 6 frequency-encoded | 8.3 s | 1.16 GB |

  322 F0 inputs = 436 columns - TransactionID, isFraud, TransactionDT, TransactionDay - 110 pruned V. Peak RSS for the whole measurement: 1.53 GB. The frequency-encoded columns are P_emaildomain, R_emaildomain, id_30, id_31, id_33, DeviceInfo.
- Open:
  - **Group approval of the V list** is still needed; until then every downstream result is provisional on it.
  - **Choices for the group to confirm:** TransactionDT and TransactionDay are excluded as features, because time is the split axis and its values in validation/test lie outside the training range. Missing flags are added only for columns with a missing training value (274 of 291). Missing categoricals get their own one-hot column. Unseen levels get all zeros (one-hot), frequency 0, or -999 (RF). Frequency features are not scaled. RF uses -999 for both missing and unseen levels. card1 and other ID-like numerics are scaled like any numeric (no target encoding, per Context.md).
  - The pipelines pick up any new columns automatically, so F1/F2 aggregates from Step 8 flow through unchanged. For CV, refit the pipeline on each fold's training part.

## Step 6: Logistic regression baseline on F0 (2026-10-18) — PROVISIONAL (V list)
- Built: `src/models/logreg.py` (`fit`, `predict_proba`, `Model`; run `.venv/bin/python -m src.models.logreg`), `src/experiment.py` (shared `time_cv`, `best_params`, `fit_validate`, `set_seeds`, `already_logged`, reusable for Steps 9-10), `tests/test_experiment.py`, executed `notebooks/02_baseline.ipynb`, `reports/figures/baseline_logreg_cv.png`, `results/log.csv` (17 rows), README section 6.
- Checks:
  - `results/log.csv` has rows for every C and fold, plus one validation row: pass (16 cv rows = 4 C x 4 folds, fold stored in params; 1 val row).
  - Validation PR-AUC is above 0.035: pass, 0.343849.
  - No row in the log has split = test: pass (splits are cv 16, val 1; the notebook asserts it).
  - Extra: the chosen model (C = 0.1, full train block) converged in 553 of 1,000 lbfgs iterations; an unlogged refit reproduced validation PR-AUC 0.343849 exactly. A synthetic end-to-end test checks the row counts, fold numbering, threshold in params and no test rows. Fast suites: 46 passed.
- Numbers (from `results/log.csv`):

  | C | Mean time-CV PR-AUC | Std | Folds 1-4 | Mean fit+predict |
  | --- | --- | --- | --- | --- |
  | 0.01 | 0.4227 | 0.0120 | 0.4081 / 0.4227 / 0.4227 / 0.4374 | 18.7 s |
  | **0.1** | **0.4269** | 0.0145 | 0.4099 / 0.4234 / 0.4292 / 0.4450 | 35.2 s |
  | 1 | 0.4264 | 0.0164 | 0.4097 / 0.4175 / 0.4315 / 0.4470 | 56.0 s |
  | 10 | 0.4248 | 0.0183 | 0.4110 / 0.4087 / 0.4322 / 0.4472 | 56.0 s |

  Validation (C = 0.1, seed 0): PR-AUC 0.3438, AUROC 0.8348, precision 0.4463 and recall 0.3144 at the F1 threshold 0.8996 (picked on validation), recall_top1pct 0.2015, fit 65 s. Whole run 13 min wall time, peak RSS 1.53 GB.
- Open:
  - **Provisional** on the unapproved V-column list (Step 5).
  - **C barely matters:** mean CV PR-AUC spans 0.4227-0.4269, well inside one fold-to-fold std. C = 0.1 won by 0.0005 over C = 1.
  - **Validation is lower than CV** (0.3438 vs 0.4269). The validation fraud rate is lower (3.13% vs 3.45-4.04% in the CV validation folds), which pulls PR-AUC down, but the gap is larger than that alone suggests; drift over time is the likely cause. Worth watching whether trees and the F1/F2 aggregates narrow it.
  - **Convergence:** lbfgs hit max_iter = 1000 on at least fold 1 at C = 10. Python shows each warning once per location, so other high-C CV fits may also have stopped early; their logged scores are slightly approximate. This does not affect the chosen C = 0.1 model.
  - **Conventions set here:** cv rows use cv_scheme "time" with the fold in params; the val row uses cv_scheme "holdout" and stores the threshold in params for Step 15. runtime_s counts model fit + predict only (not the pipeline). LR is deterministic, so Step 6 ran seed 0 only; Step 9 asks for three seeds per cell, which for LR will give identical rows.
  - The precision/recall threshold is chosen and evaluated on the same validation block, as the build step specifies, so those two numbers are optimistic; test scoring in Step 15 reuses the threshold.

## Step 7: Customer ID (uid) and check (2026-10-18)
- Built: `src/features.py` (`account_open_day`, `add_uid`, `group_stats`, `returning`, `uid_check`, `write_uid_report`; run `.venv/bin/python -m src.features`), `reports/uid_check.md`, `tests/test_features.py`, README section 7.
- Checks:
  - `reports/uid_check.md` contains every item for card1 and uid side by side: pass (distinct groups, group-size distribution, share of rows in single-transaction groups, share of groups with any fraud, fraud rate within those, returning test-block customers).
  - Rows with missing D1 or addr1 counted and in the report: pass (66,794 of 590,540 rows; 54,560 in the train block). The 66,794 rows with a "nan" uid part match exactly, since card1 is never missing.
  - Extra: 4 hand-checked tests (uid strings with missing parts, flooring, group statistics, returning counts). Fast suites: 50 passed.
- Numbers (train block, from `reports/uid_check.md`):

  | | card1 | uid |
  | --- | --- | --- |
  | Distinct groups | 12,730 | 182,535 |
  | Median / max group size | 4 / 12,205 | 1 / 361 |
  | Rows in single-transaction groups | 0.72% | 22.47% |
  | Groups with any fraud | 12.03% | 3.62% |
  | Fraud rate within those groups | 4.67% | 46.17% |
  | Test rows whose key is in train | 98.78% | 38.43% |
  | Test rows whose key is in train + validation | 98.98% | 46.76% |

- Open:
  - **uid behaves like a customer:** fraud is concentrated by uid (46.17% fraud within uids that have any, and 61.94% of those uids are all fraud) but barely by card1 (4.67%, close to the 3.51% base rate). This supports Context.md's leakage concern and the uid-grouped CV in Step 12.
  - **Returning vs new for Step 15:** 46.76% of test rows (27,616) belong to a uid seen in train or validation, and 53.24% are new, so both subsets are large enough to score.
  - **Choices:** fraud statistics use train-block labels only; returning counts use keys only, so no validation or test labels were read. account_open_day uses the integer TransactionDay, as the build step specifies. Rows with a missing part are kept with "nan" in the uid rather than given a unique ID, so 66,794 rows fall into coarser groups. 22.47% of training rows are a uid's only transaction, so F2 uid aggregates will be NaN or zero for many rows.

## Step 8: Past-only aggregates F1, F2 (2026-10-18)
- Built: `src/features.py` extended (`make_uid`, `past_aggregates`, `agg_columns`, `build_feature_set`), `tests/test_features.py` extended, `reports/feature_dictionary.md`.
- Checks:
  - A test samples 200 rows, recomputes every aggregate by brute force from `df.iloc[:row_position]`, and matches within 1e-5: pass for card1 and uid on the real table (200 random rows per key, seed 0; worst relative difference 4.9e-15). The sample includes history: 195 card1 rows and 136 uid rows have at least one prior transaction; 90 and 41 have one in the last hour. All 400 rows of a toy table with tied timestamps, repeated amounts and missing values also match.
  - Each key's first transaction has count 0 and NaN mean: pass (every first card1 and uid row on the real table; also window counts 0 on toy data). Every later row has a non-NaN mean.
  - `reports/feature_dictionary.md` lists each new column, its definition and why it might signal fraud: pass (a test checks all 22 column names appear).
  - Extra: a row never counts itself (prior count equals groupby cumcount); constant amounts give std exactly 0. Vectorised: argsort by key, per-group cumsums and diffs, a single searchsorted for the windows; no per-row loops. Fast suites: 59 passed.
- Numbers: 11 aggregates per key: prior count; prior amount mean and std; amount / prior mean; seconds since previous; counts in trailing 1 h, 24 h, 7 d; distinct prior P_emaildomain, addr1, DeviceInfo. Full-table F2 build: 1.5 s, table (590,540, 458), peak RSS 1.04 GB. NaN share on the train block: card1 mean/ratio/gap 2.69%, std 4.67%; uid 38.64% and 54.80%. Pipeline output columns (fit on train, provisional V list): F1 tree 333, linear 673; F2 tree 344, linear 688; 0 NaN in linear validation output.
- Open:
  - **Tied timestamps:** "prior" means earlier in the sort order, as the build step specifies, so a same-second transaction with a lower TransactionID counts as history. This affects 166 card1 rows and 76 uid rows. To exclude same-second rows instead (strictly earlier time), say so; it's a small change.
  - **Choices:** std uses ddof = 1, so it is NaN until two prior rows exist. Count features are 0 (not NaN) with no history. Distinct counts ignore missing values. Aggregates are computed on the full table in time order, so validation/test rows use history from earlier blocks; no labels are used.
  - **Degenerate column:** `uid_n_addr1` is always 0 or 1, because addr1 is part of the uid. It was kept for symmetry with card1; drop it if the group prefers.
  - The F1/F2 tables are built on the fly (1.5 s) rather than saved to parquet.

## Step 9: Tree models across F0 to F2 (2026-10-20) — COMPLETE, PROVISIONAL (V list)
- Built: `src/models/rf.py`, `src/models/lgbm.py`, `src/train_trees.py`, `tests/test_tree_models.py`, executed `notebooks/03_trees.ipynb`, `reports/figures/trees_validation.png`, `config/best_params.yaml`, `results/step9_manifest.json`; extended `src/experiment.py`, `tests/test_experiment.py`, and README section 8.
- Corrected the standalone LR command so `--feature-set F1/F2` builds the requested aggregates before fitting. Existing Step 6 F0 results were unaffected.
- Search: time CV with seed 0; 20 reproducibly sampled LightGBM configurations per feature set, four RF configurations (depth 12/24 × minimum leaf size 5/20; 300 trees, balanced_subsample), and the Step 6 LR C grid. LightGBM stops on sklearn average precision with patience 50 and a 1,500-tree cap; balanced weights use training labels only. Selected configurations are refit on train and scored on validation for seeds 0/1/2.
- Done-when checks:
  - All 27 validation rows (3 models × 3 feature sets × 3 seeds): pass. Log totals: 363 rows = 336 CV + 27 validation; the 17 existing F0 baseline rows were reused and 346 completed fits were appended. No test rows were scored.
  - Best configurations saved to `config/best_params.yaml`: pass. Each matches the highest mean CV PR-AUC among its fully evaluated candidates.
  - Runtime per validation fit: recorded by seed in the table below. Every CV runtime is also in the log.
- Additional verification: all logged metrics are finite; all candidate/fold and selected-configuration/seed combinations are present exactly once. A real-data resume with every model fit replaced by a failing guard performed zero fits and left `results/log.csv` byte-for-byte unchanged. Data, feature configuration, model-code hashes and package versions match the manifest. `git diff --check` passed.
- Tests: 63 fast-suite tests passed, followed by an 8-test targeted rerun including two additional regression tests (65 distinct passing tests). The unchanged V-column correlation suite was excluded. Tests cover model interfaces, training-only weights, early stopping, interrupted-run recovery, changed-input rejection and correct F1/F2 use in the LR command.
- Notebook: all 5 code cells executed without errors; notebook schema validated and the saved comparison plot was visually inspected.

Validation results below are generated from `results/log.csv`. Standard deviations are sample standard deviations over the three seeds. Times cover model fit + predict, excluding feature construction and preprocessing; timings varied during this local run.

| Model | Features | PR-AUC mean ± std | AUROC mean | Recall top 1% mean | Fit + predict seconds: seeds 0 / 1 / 2 |
| --- | --- | --- | --- | --- | --- |
| LR | F0 | 0.343849 ± 0.000000 | 0.834750 | 0.201513 | 65.17 / 87.22 / 142.14 |
| LR | F1 | 0.345514 ± 0.000000 | 0.838332 | 0.202593 | 131.43 / 97.71 / 83.20 |
| LR | F2 | 0.347050 ± 0.000000 | 0.840709 | 0.202593 | 55.00 / 56.05 / 56.21 |
| RF | F0 | 0.452248 ± 0.001699 | 0.890024 | 0.258239 | 130.70 / 99.04 / 98.47 |
| RF | F1 | 0.465774 ± 0.002072 | 0.899685 | 0.258059 | 157.12 / 158.25 / 259.48 |
| RF | F2 | 0.464473 ± 0.000569 | 0.898213 | 0.259679 | 110.73 / 113.28 / 144.92 |
| LightGBM | F0 | 0.621505 ± 0.003524 | 0.924754 | 0.293715 | 156.99 / 110.64 / 123.84 |
| LightGBM | F1 | 0.645580 ± 0.002861 | 0.923572 | 0.300918 | 157.81 / 176.49 / 145.77 |
| LightGBM | F2 | 0.654117 ± 0.002064 | 0.929875 | 0.300378 | 151.32 / 216.67 / 226.72 |

Selected configurations:

- LR: C=0.1 on F0, F1 and F2.
- RF: max_depth=24 and min_samples_leaf=20 on F0, F1 and F2; n_estimators=300.
- LightGBM F0: num_leaves=127, learning_rate=0.03, min_child_samples=100, scale_pos_weight=1.0.
- LightGBM F1: num_leaves=127, learning_rate=0.1, min_child_samples=100, scale_pos_weight=balanced.
- LightGBM F2: num_leaves=127, learning_rate=0.1, min_child_samples=20, scale_pos_weight=1.0.

Reproduce or resume:

```bash
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 VECLIB_MAXIMUM_THREADS=6 .venv/bin/python -u -m src.train_trees
.venv/bin/python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 notebooks/03_trees.ipynb
```

- Findings: LightGBM has the highest validation PR-AUC for every feature set. History features improve LR modestly; RF improves from F0 to F1 but its F2 mean is slightly lower than F1. LightGBM improves from F0 through F2. These are validation findings for the three completed model families; the MLP and full 4×3 summary remain for Steps 10–11.
- Open/limitations:
  - The V-column proposal remains unapproved; all downstream results are provisional.
  - Some high-C LR fits hit the existing 1,000-iteration cap (observed for F1 C=10 on fold 1, as previously for F0). Per-fit convergence status is not stored; scores reflect the fitted models under that cap.
  - LightGBM best iteration and resolved class weight are stored in `params._fit`. The search has a fixed 1,500-tree cap; some fits select iterations near that cap.
  - Validation is used for LightGBM early stopping and threshold choice, so these validation scores are not independent final-test estimates. The test block remains reserved until the user confirms experiment freeze.
  - Next: Step 10, MLP across F0/F1/F2. No Step 10 training or commits were made.

## Step 10: MLP across F0 to F2 (2026-10-21) — PROVISIONAL (V list)
- Built: `src/models/mlp.py`, `src/train_mlp.py` (run `.venv/bin/python -u -m src.train_mlp`), `tests/test_mlp.py`, executed `notebooks/04_mlp.ipynb`, `reports/figures/mlp_training_curves.png`, `results/step10_manifest.json`, `mlp` entries in `config/best_params.yaml`, README section 9. `src/train_trees.py`: `check_manifest` now takes the manifest path, file list, models and packages as arguments; its defaults are unchanged and the Step 9 manifest still matches.
- Method: CPU PyTorch (4 threads) on `LinearPipeline` output refit per training part. Two ReLU hidden layers (hidden, hidden/2) with dropout, Adam, batch 2,048, `BCEWithLogitsLoss(pos_weight = neg/pos)` from training rows only. Early stopping on validation PR-AUC (fold validation part in CV, validation block in the refit), patience 5, max 40 epochs, best epoch's weights restored. Grid: hidden {128, 256} x dropout {0.1, 0.3} x lr {3e-4, 1e-3}, time CV seed 0; the best config by mean CV PR-AUC is refit and validated with seeds 0, 1, 2 via Step 9's `run_cell`.
- Checks:
  - The log has MLP validation rows for F0, F1, F2, three seeds each: pass (9 val rows plus 96 cv rows = 8 configs x 4 folds x 3 feature sets). The log went from 363 to 468 rows; the 363 earlier rows are byte-identical to a pre-run backup. No test rows.
  - Training curves (loss and validation PR-AUC by epoch) saved to `reports/figures/`: pass (`mlp_training_curves.png`, one column per feature set, one line per seed, best epoch marked; inspected). Every fit's curves are stored in its log row (`params._fit`).
  - Best config added to `config/best_params.yaml`: pass (`models.mlp.F0/F1/F2` plus an `mlp_training` block).
  - Extra: 5 tests cover the interface, pos_weight from training labels, best-epoch restore (rescoring reproduces the best epoch's PR-AUC), patience, same-seed reproducibility, required validation data, and run_cell logging/resume. Fast suites: 70 passed.
- Numbers (from `results/log.csv`; validation mean ± sample std over seeds 0/1/2):

  | Features | Selected config | Val PR-AUC | AUROC | Recall top 1% | Seeds 0 / 1 / 2 | Fit + predict s |
  | --- | --- | --- | --- | --- | --- | --- |
  | F0 | hidden 256, dropout 0.3, lr 1e-3 | 0.445295 ± 0.010321 | 0.842915 | 0.256798 | 0.455863 / 0.444784 / 0.435239 | 37.9 / 34.7 / 26.4 |
  | F1 | same | 0.452102 ± 0.002586 | 0.846853 | 0.256258 | 0.449940 / 0.451398 / 0.454966 | 28.8 / 38.6 / 38.7 |
  | F2 | same | 0.445618 ± 0.012238 | 0.858406 | 0.245273 | 0.443449 / 0.458795 / 0.434610 | 31.3 / 52.9 / 24.8 |

  Mean time-CV PR-AUC of the selected config: F0 0.4746, F1 0.4800, F2 0.4763; all 8 configs fall within 0.4671-0.4800. CV fits ran 7-29 epochs (best epoch 2-24, median 9); validation fits stopped at best epochs 13-32. No fit reached the 40-epoch cap. Whole run 21 min wall time, peak RSS 1.83 GB.
- Open:
  - **Grid corner:** the same config (largest hidden, dropout and learning rate in the grid) won on all three feature sets. CV differences between configs are small (0.013 range), so a wider grid would likely help only a little, but the optimum may lie outside it. The RQ4 sweeps in Step 13 vary these one at a time.
  - **Noisy stopping:** validation PR-AUC moves about ±0.01-0.02 between epochs, so patience 5 stops at a partly random epoch. That explains most of the seed spread (std 0.010-0.012 on F0 and F2). Patience 10 or a learning-rate schedule would steady it; this was not changed, since the build step fixes patience at 5.
  - **No clear F0 to F2 gain for the MLP:** PR-AUC differences are within seed noise, though AUROC rises (0.843 to 0.858). The MLP sits between RF (0.45-0.47) and LightGBM (0.62-0.65) on validation PR-AUC.
  - As with LightGBM, the validation block drives early stopping and threshold choice, so MLP validation scores are not independent estimates; the test block remains unscored.
  - All 12 grid cells now have validation rows, so Step 11 (grid summary) is next.

## Step 11: Grid summary (2026-10-21) — PROVISIONAL (V list)
- Built: `src/summarise.py` (`validation_grid`, `metric_table`, `render`), `reports/grid_validation.md`, `tests/test_summarise.py`, README section 10.
- Checks:
  - All 12 cells are filled from the log, with no manual entry: pass. The report is rendered entirely from the 36 validation rows of `results/log.csv`. `validation_grid` raises if any (model, feature set) cell lacks one of seeds 0, 1, 2 or repeats one, so reruns or partial results cannot be averaged in silently (tested).
  - The table regenerates with one command: `.venv/bin/python -m src.summarise`.
  - Extra: 4 tests on a synthetic log (means, sample std, both gains, CV rows ignored, incomplete and duplicated cells rejected). Cell values match the Step 9 and Step 10 entries. Fast suites: 74 passed.
- Layout: one table per metric (PR-AUC, AUROC, recall in top 1%), rows = models, columns = F0/F1/F2. The **LR → LightGBM** gain is a derived row. The **F0 → F2** gain is a derived last column rather than a row, because it is one value per model. Both show the absolute and relative change.
- Numbers (validation PR-AUC, mean over seeds, from `reports/grid_validation.md`):

  | Model | F0 | F1 | F2 | F0 → F2 |
  | --- | --- | --- | --- | --- |
  | LR | 0.3438 | 0.3455 | 0.3471 | +0.0032 |
  | RF | 0.4522 | 0.4658 | 0.4645 | +0.0122 |
  | LightGBM | 0.6215 | 0.6456 | 0.6541 | +0.0326 |
  | MLP | 0.4453 | 0.4521 | 0.4456 | +0.0003 |
  | LR → LightGBM | +0.2777 | +0.3001 | +0.3071 | |

- Open:
  - **RQ3 reading (validation, provisional):** the model gain (LR → LightGBM, +0.28 to +0.31 PR-AUC) is roughly ten times the largest feature-set gain (F0 → F2, +0.033 for LightGBM). Customer history helps LightGBM most, helps RF and LR a little, and is within seed noise for the MLP. Final claims should wait for the leakage check (Step 12) and test scoring (Step 15).
  - Results stay provisional on the unapproved V-column list. LightGBM and the MLP early-stop on the validation block, which slightly favours them in this table.

## Step 12: Leakage check (2026-10-22) — PROVISIONAL (V list)
- Built: `src/leakage.py` (run `.venv/bin/python -u -m src.leakage`; `--report` rebuilds the report only), `reports/leakage_check.md`, executed `notebooks/05_leakage.ipynb`, `reports/figures/leakage_check.png`, `results/step12_manifest.json`, `tests/test_leakage.py`, `tests/conftest.py`, README section 11.
- Method: LR (C = 0.1) and LightGBM (num_leaves 127, learning_rate 0.1, min_child_samples 20, scale_pos_weight 1.0) on F2 with their Step 9 best configs, inside the train block only. New: random stratified 5-fold (`cv_scheme` "random") and StratifiedGroupKFold on uid (`cv_scheme` "group_uid"), seed 0, 20 fits. Reused: the 4 Step 9 time-CV rows per model with the same config, as requested. Every fold refits its pipeline on its training part; LightGBM early-stops on the fold's validation part, as in time CV.
- Checks:
  - The report table has 6 cells, each built from the log: pass. `leakage_table` reads only `results/log.csv` and raises unless each cell has exactly one row per fold (5, 5, 4).
  - The report states which scheme scored highest and by how much, using the logged numbers: pass (random, for both models; margins below).
  - Extra: the log grew from 468 to 488 rows; the earlier 468 are byte-identical to a pre-run backup; no test rows. Group folds share no uid between sides (tested on toy data; the split function was already tested on card1). Fast suites: 78 passed.
- Numbers (PR-AUC, mean ± sample std over folds, from `reports/leakage_check.md`):

  | Model | Random 5-fold | Group on uid 5-fold | Time CV (4) |
  | --- | --- | --- | --- |
  | LR | 0.4546 ± 0.0085 | 0.4310 ± 0.0206 | 0.4257 ± 0.0163 |
  | LightGBM | **0.8756** ± 0.0052 | 0.5909 ± 0.0148 | 0.6416 ± 0.0536 |

  Random scores highest for both. LR: +0.0236 over group, +0.0289 over time. LightGBM: +0.2846 over group (+48.2%), +0.2339 over time (+36.5%). Run time 29 min, peak RSS 2.0 GB.
- Open:
  - **The leakage is model-dependent.** LightGBM gains 0.23-0.28 PR-AUC from random folds; LR gains only 0.02-0.03. A flexible model can memorise customer-specific patterns, which a linear model cannot. This is the core finding for the report's "why random K-fold misleads" section.
  - **Group < time for LightGBM** (0.591 vs 0.642). Group folds hold out whole customers, so every validation row is a customer never seen in training. In time CV many validation customers already have history from earlier months (Step 7: about 47% of later rows are returning uids), and F2 encodes that history. Group folds are therefore a "new customers only" estimate, stricter than deployment. Time CV has the larger spread (fold 4 = 0.711 vs fold 3 = 0.596).
  - **Random-fold LightGBM fits ran to the tree cap** (best iteration 1,319-1,500 of 1,500), so its random score is if anything understated. Group-fold fits stopped at 146-612.
  - **Training-size caveat:** random and group folds train on 80% of the block; time folds on 28-88%. That favours the 5-fold schemes, but cannot explain LightGBM's random-vs-group gap, since those two use the same size.
  - **Two bugs fixed before any fit was logged:** row matching broke on pandas 3 Arrow string columns read from CSV, then on an empty selection. The aborted runs logged nothing; their manifest was deleted and rewritten by the successful run. A CSV-roundtrip test now covers both.
  - **Test crash fixed:** importing LightGBM before training a torch model segfaults on macOS (two OpenMP runtimes). `tests/conftest.py` imports torch first. The training commands were not affected (`train_mlp` imports torch first), but any future script that trains the MLP should do the same.
  - `git diff --check` flags `results/log.csv` because `log_run` writes CRLF line endings (Python csv default); this was true of the committed rows already and is harmless.

## EDA notebooks combined (2026-11-14)
- `notebooks/01_profile.ipynb`, `notebooks/01_eda.ipynb` and `notebooks/eda_features.ipynb` are now one notebook, `notebooks/01_eda.ipynb` (executed, outputs committed). Earlier entries that name `01_profile.ipynb` refer to what is now its section 3 and section 13.
- Kept from `01_profile`: the missing-vs-present comparison (section 3) and V blocks with a same-rows check (all 14 blocks are missing on identical rows). The per-column profile table stays in `reports/data_profile.md`.
- The V threshold is no longer argued in the EDA: `notebooks/logreg_v_threshold.ipynb` gained a result section (section 6) with the sweep table and the decision r = 0.75, and EDA section 13 only references it (and uses the shared `prune_v` from `src/feature_selection.py`). The first proposal, r = 0.90 in `config/features.yaml`, is described there as replaced.
- Kept from Jarren's `01_eda`: `src/eda.run()` (regenerates `reports/eda/` and `reports/eda_findings.md`; CSV tables unchanged), class balance, weekly volume and fraud rate with incomplete weeks marked, the absolute missing-vs-present gap chart, the amount summary and fixed bands, and the seven report figures. Duplicate charts (categories, emails, amount histogram, missingness) use the 95%-interval versions, saved under the same figure names; two sentences in `src/eda.py` describing those charts were updated.
- Section numbers of the former `eda_features` are unchanged, so "EDA x.y" references in `notebooks/feature_engineering.ipynb` and `config/features_v2.yaml` still hold. The training block now comes from `src/eda.load_training` (cleaned.parquet) and the validation block from `src/split.holdout_indices`; an assert checks the rows match.
- Corrected: weekly fraud rate over complete weeks is 2.1% to 5.1% (was stated as 1.9% to 5.1% without a computing cell); 30-day blocks 2.5%, 4.0%, 4.0%, 4.0%, 3.5%.
- References updated: `src/feature_selection.py`, `src/data_profile.py` and `reports/data_profile.md`, `src/eda.py`, `notebooks/feature_engineering.ipynb`, `README.md` (section 5), `BUILD_STEPS.md` (Step 2 files).

## Repo cleanup before the G0–G3 model update (2026-11-14)
- Results: `results/jarren.csv` (Jarren's 488 rows plus Damien's 312 CatBoost rows) and `results/mabel.csv` (306) combined into the single `results/log.csv` the code and docs already use (group decision): 1,106 rows, no duplicates. `reports/leakage_check.md` rebuilds identically; `reports/grid_validation.md` has identical tables, but its header row count would now include every validation row in the log (126, not 36), so it was left as committed until the G0–G3 rerun.
- CatBoost: `catboost` added to `requirements.txt`; `requirements-catboost.txt` removed; `CATBOOST.md` folded into README section 11 (its notes on the unmerged PR and missing data were out of date). CatBoost tests: 11 passed.
- Notebooks: renamed `feature_engineering` → `02_feature_engineering`, `logreg_v_threshold` → `03_v_threshold`, `05_leakage` → `11_leakage_check`, leaving 04–10 for one notebook per model plus a comparison. Archived (moved, not deleted) to `notebooks/archive/` with a README: `02_baseline`, `03_trees`, `04_mlp` (F0–F2 results viewers) and `xgboost`, `xgboost_v_threshold` (`features_v1`). References updated in README, BUILD_STEPS, `config/features_v2.yaml`, the EDA and FE notebooks, `src/domain_features.py`, `src/feature_selection.py` and a test comment. Earlier PROGRESS entries keep the old names as history.

## Model notebooks on the G0–G3 feature sets (2026-11-14)
- Built: `src/feature_sets.py` (G0–G3 column lists from `config/features_v2.yaml`; `load_features()` returns features_v2 with the test block removed), `src/models/xgb.py` (shared model contract; native categories; fold-local "balanced" weight; early stopping on sklearn average precision, patience 50, best round kept), `tune_and_validate`, `oof_predictions`, `save_predictions` and `run_summary` in `src/experiment.py`, an optional `columns` argument for `TreePipeline`, `LinearPipeline` and `CatBoostPipeline` (unchanged behaviour without it, so the F0–F2 scripts still work), notebooks `04_logreg`, `05_random_forest`, `06_lightgbm`, `07_xgboost`, `08_catboost`, `09_mlp` (one template) and `10_model_comparison`, `tests/test_v2_models.py`. `results/preds/` is gitignored.
- Checks: full suite 108 passed (5 new tests). Every model ran end to end through `tune_and_validate` on the real features_v2 (every 20th row, one small setting, G0 and G3, logging to a scratch file): tuning, validation, saved validation and fold scores, and a rerun that skipped everything. Test copies of all seven notebooks executed top to bottom the same way. `results/log.csv` was not modified; no G0–G3 results exist yet.
- Notes: on macOS, LightGBM crashes when trained with several threads after PyTorch is loaded, and XGBoost crashes once PyTorch is loaded first (each ships its own OpenMP). The notebooks import their model library first and run one model per kernel, which is safe; the XGBoost test runs in a subprocess because `tests/conftest.py` loads PyTorch first for LightGBM's sake.
- Open: the full runs (all settings, G0–G3, seeds 0–2). Settings grids differ in size by model (logreg 4, RF 4, MLP 8, LightGBM/XGBoost/CatBoost 20); the default tunes once on G3 and validates those settings on every set. LightGBM and LR give identical scores across seeds (no random sampling in their defaults). The F0–F2 scripts, `config/features.yaml`, `features_v1.yaml` and the old best-parameter files are still in place until the group has the G0–G3 results.

## Known vs unknown customer segments (2026-11-14)
- Built: `src/segments.py` (`known_uid`, `fold_reference_uids`, `bootstrap_pr_auc`, `segment_pr_auc`); `10_model_comparison` sections 3–4 now report PR-AUC for known vs unknown `uid` with 95% bootstrap intervals (mean over seeds 0–2) on the validation block, and per-fold PR-AUC on the saved time-CV fold scores, where "known" means seen in that fold's training months. Group decision: segment by known and unknown uid only.
- Validation block: 48.6% of rows have a known uid (28,685 rows, 640 fraud) and 51.4% an unknown one (30,369 rows, 1,211 fraud), computed from features_v2 with no model.
- Checks: 2 new tests (fold references use only that fold's training months; intervals contain the point and are reproducible); full suite 110 passed. The comparison notebook ran end to end on the scratch test scores; no real G0–G3 results exist yet.

## Model notebook results sections and precision in the top 1% (2026-11-14)
- Each model notebook (04–09) now reports, from the log: tuning means over the 4 folds for PR-AUC (± sd), AUROC, recall and precision in the top 1%, with the chosen settings marked (5.1); validation mean ± sd over seeds for PR-AUC, AUROC, recall and precision in the top 1%, precision and recall at the threshold, and the mean threshold (5.2); the change in PR-AUC from G0 to G3 step by step (5.3); and known vs unknown PR-AUC with 95% bootstrap intervals (5.4). `10_model_comparison` shows the same six metrics.
- Precision in the top 1% (`src/metrics.py`: `precision_top1pct`, `top_rows`) is derived from the logged recall in the top 1% (precision = recall × fraud rows ÷ ceil(1% of rows)), so the log format is unchanged and past runs get it too; it ranks models exactly as recall in the top 1% does. On the validation block the top 1% is 591 rows and there are 1,851 fraud rows, so recall in the top 1% is at most 0.319; the notebooks print this ceiling. `experiment.block_sizes` gives the row and fraud counts per fold and for validation.
- Checks: 1 new test (precision in the top 1% equals a direct count and recall × fraud ÷ rows); full suite 111 passed. Test copies of all seven notebooks ran end to end again, reusing the scratch log (every fit skipped). The comparison notebook assumes validation rows come from the full validation block, which is how the model notebooks run.

## Same subsampling for the three boosting models (2026-11-14)
- LightGBM, XGBoost and CatBoost now all use a random 80% of rows and 80% of columns per tree, fixed and not tuned: LightGBM `subsample=0.8, subsample_freq=1, colsample_bytree=0.8`; XGBoost `subsample=0.8, colsample_bytree=0.8` (column sampling removed from its grid, now 20 of 48 combinations); CatBoost `bootstrap_type="Bernoulli", subsample=0.8, rsm=0.8`. Before this, LightGBM used no subsampling and CatBoost its default MVS row sampling with all columns.
- The earlier F0–F2 LightGBM and CatBoost runs in `results/log.csv` used those old defaults, so they can't be reproduced with the current code (and `src/train_trees.py` / `src/train_catboost.py` will refuse to resume, as their manifests fingerprint the model code). The G0–G3 runs all use the new settings.
- With row sampling, LightGBM's validation scores now vary by seed (the same seed still gives identical scores; checked for LightGBM and CatBoost); logistic regression remains deterministic. Tests: 111 passed; LightGBM, XGBoost and CatBoost ran end to end on a sample of the real features_v2 with a scratch log.

## Fair early stopping and comparable search budgets (2026-11-14)
- Early stopping: in the G0–G3 runs (`tune_and_validate`, `oof_predictions`), LightGBM, XGBoost, CatBoost and the MLP (`EARLY_STOPPING = True` in their modules) now train on the first 90% (by time) of each training part and early-stop on the last 10%, instead of on the fold or validation rows they are then scored on. Logistic regression and random forest still train on the whole part. The number of held-out rows is logged in `params._fit.early_stopping_rows`. `time_cv`, `fit_validate` and `_fit_score` take this as an option (`es_fraction`, default None), so the F0–F2 scripts and their tests keep the old behaviour.
- Search budgets: logistic regression C now 11 values from 0.001 to 100 (was 4); random forest 12 settings, depth {12, 24, unlimited} × leaf {5, 20} × `max_features` {sqrt, 0.3} (was 4); MLP random search, 16 of 32 combinations adding depth {2, 3} and weight decay {0, 1e-5} (was an 8-setting grid; the 2-layer network is unchanged). Boosting models stay at 20. The F0–F2 runs in the log used the old grids.
- Checks: 1 new test (early-stopping rows are the last 10% of each training part and never overlap the scored rows; LR trains on all rows); full suite 112 passed. All six models ran end to end on a sample of the real features_v2 (early-stopping rows logged for the four that use it), and test copies of all seven notebooks ran top to bottom with a scratch log; `results/log.csv` unchanged.

## Precision–recall curves and feature importance (2026-11-14)
- Each model notebook now plots precision–recall curves per feature set (5.2) and has a feature importance section (6): the chosen settings are refitted on the training block exactly as in the seed-0 validation run (`experiment.fit_final`, cached in `results/models/`, gitignored; checked to give the same validation PR-AUC), then permutation importance (drop in validation PR-AUC when a raw column and all pipeline columns it produces are shuffled, on 20,000 validation rows) and the model's built-in importance (`feature_importance` in each model module: |coefficient|, impurity, total gain, prediction values change; none for the MLP), each feature labelled with the G group that adds it, plus totals by group. Tables are saved to `results/importance/<model>_<set>.csv`.
- `10_model_comparison` adds precision–recall curves of all models on the largest feature set and a cross-model table of permutation-importance ranks and totals by feature group.
- Checks: 2 new tests (refit matches the logged validation run and is cached; pipeline columns map back to raw columns, the informative column ranks first, category dtypes survive shuffling); full suite 114 passed. The refit-matches-validation check passed for all six models on synthetic data, and test copies of all seven notebooks ran end to end on a sample of the real data with scratch paths.

## Same L2 regularisation search for the three boosting models (2026-11-14)
- LightGBM and XGBoost now tune `reg_lambda` (L2 on the leaf weights) over {1, 3, 10}, the values CatBoost already used for `l2_leaf_reg`. Before, LightGBM used none (default 0) and XGBoost a fixed 1. Each grid is still a seeded random sample of 20 combinations (LightGBM now from 108, XGBoost from 144); each L2 value appears 5–8 times in each.
- Regularisation by model: logistic regression L2 via C (tuned); random forest depth, leaf size and `max_features` (tuned) plus bagging; MLP dropout and weight decay (tuned) plus early stopping; the boosting models tree size, minimum leaf size or child weight, learning rate and L2 (tuned), fixed 80% row and column subsampling, and early stopping on held-out rows.
- Checks: the requested `reg_lambda` reaches LightGBM and XGBoost; full suite 114 passed.

## RQ4: setting sensitivity from the tuning runs (2026-11-14)
- `experiment.setting_sensitivity` reads a model's time-CV rows from the log, averages each tried combination over its folds (only complete combinations), and for each tuned setting and value gives the mean, best and worst PR-AUC and the number of combinations. Each model notebook shows it as section 5.5 (table plus one panel per setting, chosen value starred), with no extra training.
- Caveat stated in each notebook: LR and RF use full grids, so values are compared on equal terms; LightGBM, XGBoost, CatBoost and the MLP use random search, so each value is paired with a different mix of other settings and the means are indicative rather than a controlled one-at-a-time sweep (as `Context.md`'s RQ4 plan describes). The tree count and epochs are set by early stopping, not tuned.
- Checks: 3 new tests (fold averaging and incomplete combinations; unlimited depth kept as "none"; integer values kept as integers); full suite 117 passed. Test copies of the six model notebooks ran end to end with multi-setting grids on a data sample.


## XGBoost on G0–G3 (2026-11-15)
- Ran `07_xgboost.ipynb` in full: 20 settings tuned on G3 with time CV (80 rows in `results/log.csv`), then validated on G0–G3 with seeds 0–2 (12 rows). Chosen: learning rate 0.03, depth 10, min child weight 1, `reg_lambda` 1, no class weighting (CV PR-AUC 0.5788).
- Validation PR-AUC: G0 0.5289, G1 0.5636, G2 0.5629, G3 0.5615. Known customers 0.6893 vs unknown 0.5081 on G3. Findings written in section 7 of the notebook.
- Permutation importance saved to `results/importance/xgboost_G3.csv`. Validation and OOF scores in `results/preds/xgboost_*` (gitignored; to be shared on the group drive).

## Early stopping refit and a higher tree cap (2026-11-16)
- Problem found after the XGBoost run: validation models of the early-stopping models (LightGBM, XGBoost, CatBoost, MLP) trained on only the first 90% of the training block, because the last 10% was held out for early stopping. Those are the rows closest in time to the validation block, so the model missed many recent customers, and logistic regression and random forest (which train on every row) were compared on a different footing. Check on XGBoost G3 seed 0 with the chosen settings and 1,040 trees: trained on the first 90%, validation PR-AUC 0.5600 (known customers 0.6869, unknown 0.5081); trained on the whole block, 0.6251 (0.8079, 0.5371). Also, XGBoost hit the 1,500-tree cap in 14 of its 80 tuning fits (13 of them at learning rate 0.03) and 5 of its 12 validation fits.
- Fix 1 (`src/experiment.py`, `_fit_on_part`): validation, OOF and `fit_final` fits of the early-stopping models now take two fits. The first trains on the first 90% and finds the stopping point on the last 10%; the second refits the pipeline and model on the whole training part with that stopping point (`refit_params` in each model module: trees, or epochs for the MLP) and no early stopping. The validation block is used in neither fit. Tuning (time CV) is unchanged. The MLP can now train for a fixed number of epochs without validation rows.
- Fix 2: tree cap raised from 1,500 to 3,000 for LightGBM (`n_estimators`), XGBoost (`n_estimators`) and CatBoost (`iterations`).
- MLP class weighting: `pos_weight` {1 (unweighted), "balanced"} added to the MLP random search (still 16 combinations, now sampled from 64; 7 unweighted, 9 balanced), the same choice the boosting models tune as `scale_pos_weight`. Logistic regression and random forest keep balanced weights (to be stated as a limitation; for XGBoost the best weighted and unweighted settings differed by 0.0001 CV PR-AUC).
- Runs under the new setup log `es_setup: 2` in params. Early-stopping rows without it (the first XGBoost run: 80 tuning and 12 validation rows) stay in `results/log.csv` but are left out of every summary (`experiment.current_runs`, also used by `10_model_comparison`), and they don't count as already run, so rerunning `07_xgboost` redoes them. Saved OOF scores and cached final models from the older setup are recomputed. Logistic regression and random forest are unaffected.
- Notebooks: model input and tune-and-validate text in 06–09 describe the two-step fit and the new cap; 07's findings reset to the template until it is rerun; `02_feature_engineering`'s stale note about early stopping on the validation block corrected.
- Checks: 4 new tests (MLP class weight is a setting; refit trains on the whole block with the stopping point from step 1 and logs it; refit settings fix the tree count for LightGBM, CatBoost and XGBoost and the MLP's epochs; older-setup rows are rerun, OOF recomputed and left out of the summaries while logistic regression rows stay); 2 tests updated; full suite 121 passed. On every 20th row of the real features, a 2-setting XGBoost run logged `es_setup` 2, refit with the step-1 tree count, and `fit_final` reproduced its validation PR-AUC exactly; `results/log.csv` unchanged.

## XGBoost rerun with the refit and the 3,000-tree cap (2026-11-15)
- Reran `07_xgboost.ipynb` in full under `es_setup` 2: 80 tuning rows and 12 validation rows added to `results/log.csv` (the 92 older-setup rows stay but are left out of the summaries). Chosen: learning rate 0.03, depth 10, min child weight 1, `reg_lambda` 10, balanced class weight (CV PR-AUC 0.5819).
- Validation PR-AUC: G0 0.6057, G1 0.6495, G2 0.6449, G3 0.6434 (G3 was 0.5615 under the old setup). G0 → G1 adds 0.0437; G2 and G3 add nothing measurable. Known customers 0.8414 vs unknown 0.5369 on G3. Findings rewritten in section 7.
- Tree cap: no tuning fit reached 3,000 (most 2,607), but 6 of the 12 validation fits stopped within 50 trees of it (2,977 to 2,998). Kept at 3,000 for all boosting models and stated as a limitation in the findings.
- Permutation importance updated in `results/importance/xgboost_G3.csv`. New validation and OOF scores in `results/preds/xgboost_*` (gitignored; replace the old ones on the group drive).

## Random forest and MLP on G0–G3 (2026-11-15)
- Ran `05_random_forest.ipynb` in full: 12 settings tuned on G3 with time CV (48 rows in `results/log.csv`), then validated on G0–G3 with seeds 0–2 (12 rows). Chosen: unlimited depth, minimum leaf 5, `max_features` 0.3 (CV PR-AUC 0.5847). Validation PR-AUC: G0 0.5411, G1 0.5607, G2 0.5667, G3 0.5620. Unaffected by the early-stopping refit (random forest trains on the whole block).
- Ran `09_mlp.ipynb` in full on the refit setup (`es_setup` 2, `pos_weight` in the search): 16 settings tuned on G3 (64 rows), validated on G0–G3 with seeds 0–2 (12 rows). Chosen: 2 layers, hidden 128, dropout 0.3, learning rate 0.0003, weight decay 1e-5, unweighted loss (CV PR-AUC 0.4492). Validation PR-AUC: G0 0.4237, G1 0.4404, G2 0.4360, G3 0.4379.
- An earlier MLP run on the older setup (64 tuning and 12 validation rows, before `pos_weight` and the refit) stays in the log and is left out of the summaries by `experiment.current_runs`.
- Permutation and built-in importance saved to `results/importance/rf_G3.csv` and `mlp_G3.csv`. Validation and OOF scores in `results/preds/rf_*` and `mlp_*` (gitignored; to be shared on the group drive). Findings cells (section 7) not yet written.

## Findings for random forest and MLP (2026-11-16)
- Wrote section 7 of `05_random_forest.ipynb` and `09_mlp.ipynb` from the tables in each notebook; no reruns, `results/log.csv` unchanged. Every figure was checked against the notebook outputs and `results/importance/*.csv`.
- Random forest: G2 scores best (0.5667); G0 → G1 adds 0.0196, G1 → G2 0.0060, G2 → G3 −0.0046. All of the gain goes to known customers (0.6611 on G0 to 0.7288 on G2; unknown 0.5266 to 0.5209). Chosen settings are all at the edge of the grid; class weights were not tuned (limitation).
- MLP: G1 scores best (0.4404); G1, G2 and G3 are tied within the seed spread. Known customers score lower than unknown ones on every set (0.3940 vs 0.4661 on G3). Unweighted loss and dropout 0.3 each add about 0.01 mean CV PR-AUC.

## CatBoost on G0–G3 (2026-11-15)
- Ran `08_catboost.ipynb` in full under `es_setup` 2 (about 5 h 20 min on an M5 Pro): 20 settings tuned on G3 with time CV (80 rows in `results/log.csv`), then validated on G0–G3 with seeds 0–2 (12 rows). Chosen: depth 8, `l2_leaf_reg` 3, learning rate 0.03, no class weighting (CV PR-AUC 0.5605).
- Validation PR-AUC: G0 0.5498, G1 0.5625, G2 0.5504, G3 0.5485. G0 → G1 adds 0.0128; G2 and G3 add nothing. Known customers 0.5912 vs unknown 0.5397 on G3: level with XGBoost on unknown customers, far below it on known ones (likely symmetric trees and no target statistics; untested). Findings written in section 7.
- Tree cap: 2 of 80 tuning fits and 7 of 12 validation fits stopped within 50 trees of 3,000; stated as a limitation.
- Permutation importance saved to `results/importance/catboost_G3.csv`. Validation and OOF scores in `results/preds/catboost_*` (gitignored; to be shared on the group drive).

## LightGBM on G0–G3 (2026-11-16)
- Ran `notebooks/06_lightgbm.ipynb` in full on branch `results-lightgbm`, based on `62087b8` (the merged early-stopping/refit update). Settings were unchanged: 20 configurations tuned on G3 with four time-CV folds, then validation on G0–G3 with seeds 0, 1 and 2. The run uses `es_setup=2`, a 3,000-tree cap and the full-training refit after the stopping point is chosen inside training.
- Checks: all 20 configurations have folds 1–4 exactly once (80 CV rows); every validation set has seeds 0–2 exactly once (12 validation rows); all 13 notebook code cells executed with no saved errors. All 16 prediction files exist: 12 validation files with 59,054 rows each and four OOF files with 338,093 rows each. `results/importance/lgbm_G3.csv` contains all 252 G3 inputs. No test rows have been logged.
- Selected settings: learning rate 0.03, 127 leaves, minimum child size 300, `reg_lambda=10.0`, `scale_pos_weight=1.0`; mean time-CV PR-AUC 0.5870. Validation PR-AUC (mean ± sample standard deviation): G0 0.6076 ± 0.0037; G1 0.6221 ± 0.0042; G2 0.6238 ± 0.0009; G3 0.6240 ± 0.0058.
- Findings are written in notebook section 7. Most mean validation improvement is G0 → G1 (+0.0144); the total G0 → G3 gain is +0.0164, while the G2 → G3 gain (+0.0002) is small relative to seed variation. G3 known-customer PR-AUC is 0.7992 versus 0.5466 for unknown customers. Raw C1, amount and card1 lead permutation importance, with engineered card frequency and normalised D1 next. RQ4 summaries are marginal comparisons from random search, not controlled one-at-a-time effects.
- Saved the executed notebook, the 92 new rows in `results/log.csv`, and `results/importance/lgbm_G3.csv`. Validation/OOF predictions are in ignored `results/preds/lgbm_*`; they still need sharing through the group Drive. Findings-cell edits preserve all executed code cells and outputs. The feature-building notebook was also executed locally to create `features_v2.parquet`; its generated output changes are separate from the model results.
- Local verification before the full run: six focused G0–G3 tests passed, and a fresh notebook-kernel smoke test confirmed the 3,000-tree default, `ES_SETUP=2`, full-training pipeline refit and finite predictions using the existing PyTorch OpenMP library. These checks did not alter the shared log.

## Step 17: Model comparison (2026-11-17)
- Ran `10_model_comparison.ipynb` on `main` after all six G0–G3 runs were merged; it reads `results/log.csv` through `experiment.current_runs` and the saved scores, refits nothing (about 1 minute). All 24 cells are filled; no row has split = test.
- Before running, checked all 96 files in `results/preds/` (16 per model, the logreg, rf and mlp ones from the group drive): every validation file reproduces its logged PR-AUC exactly and carries the chosen settings, the fold-score files cover all 338,093 time-CV rows, and the four early-stopping models' fold scores are `es_setup` 2.
- Results: best is XGBoost on G1 (0.6495), 0.0255 above the best other model (LightGBM, 0.6240 on G3). On G3: XGBoost 0.6434, LightGBM 0.6240, random forest 0.5620, CatBoost 0.5485, MLP 0.4379, logistic regression 0.3525. The spread between models (0.2909 on G3) is far larger than the largest feature gain G0 → G3 (0.0377); G2 → G3 is −0.0046 to +0.0020 for every model. Tree models score known customers far above unknown ones; logistic regression and the MLP the reverse. No model is clearly best per segment on validation (overlapping intervals); on the fold scores XGBoost leads on known and CatBoost on unknown customers.
- Section 6 (findings) written. `reports/feature_dictionary.md`: the Step 14 placeholder replaced by plain descriptions of the six inputs with the best average rank (fe_D1n, C11, C14, C1, and C2 and fe_amt_log tied for fifth).

## Step 18: Leakage check on G3 (2026-11-17)
- Built: `src/leakage.py` rewritten for the G0–G3 setup (run `.venv/bin/python -u -m src.leakage`; `--report` rebuilds the report only), `tests/test_leakage.py`, `reports/leakage_check.md` (now G3 only), executed `notebooks/11_leakage_check.ipynb`, README section 13. The F2 check from Step 12 stays in the log and is no longer in the report.
- Method: LR (C = 0.03) and LightGBM (learning rate 0.03, 127 leaves, minimum child size 300, `reg_lambda` 10, no class weight) on G3, the settings validated in Step 16, read from the G3 validation rows of the log. New: random stratified 5-fold (`cv_scheme` "random") and StratifiedGroupKFold on uid (`cv_scheme` "group_uid"), seed 0, 20 fits. Reused: the 4 Step 16 time-CV tuning rows per model with the same settings. Every fold is fitted like a tuning fold: the pipeline is refit on the fold's training part, and LightGBM trains on the first 90% of that part by time and early-stops on the last 10% (`es_setup` 2), never on the scored rows. Step 12 stopped on the scored rows, so its F2 numbers are not comparable with these.
- Checks:
  - The G3 report table has 6 cells, each built from the log with one row per fold (5, 5, 4): pass. `leakage_table` raises otherwise.
  - The report states which scheme scored highest and by how much, from the logged numbers: pass (random, for both models; margins below).
  - Group folds share no uid between the two sides of a fold: pass on the real G3 folds (0 shared uids in each of the 5 folds; the random folds share about 40,500 each). The run stops if this fails.
  - No row in the log has split = test: pass.
  - Extra: the log grew from 1,742 to 1,762 rows and the earlier 1,742 are byte-identical to a pre-run backup. LightGBM rows from the older early-stopping setup never match. 7 leakage tests (3 new: fold construction and the uid check, early stopping on held-out training rows, chosen settings read from the current validation rows); fast suite 121 passed (V-column correlation tests not run).
- Numbers (PR-AUC, mean ± sample std over folds, from `reports/leakage_check.md`):

  | Model | Random 5-fold | Group on uid 5-fold | Time CV (4) |
  | --- | --- | --- | --- |
  | LR | 0.4417 ± 0.0075 | 0.4201 ± 0.0117 | 0.4110 ± 0.0145 |
  | LightGBM | **0.8408** ± 0.0103 | 0.6018 ± 0.0131 | 0.5870 ± 0.0256 |

  Random scores highest for both. LR: +0.0216 over group (+5.1%), +0.0307 over time (+7.5%). LightGBM: +0.2390 over group (+39.7%), +0.2538 over time (+43.2%). Fit time 29.5 min for the 20 fits (LR 23-28 s per fold, LightGBM 87-261 s).
- Open:
  - **The leakage is model-dependent, as on F2:** random folds add 0.24-0.25 PR-AUC for LightGBM and 0.02-0.03 for LR.
  - **Group and time are close for LightGBM** (0.6018 vs 0.5870, a gap smaller than the time-CV fold spread of 0.0256). Group and random folds train on 80% of the block and time folds on 28-88%, which favours the 5-fold schemes, so the two are not separated by this check.
  - **Tree cap:** LightGBM stopped at 941 to 2,218 trees in the random folds and 931 to 2,089 in the group folds, so no fit reached the 3,000 cap.
  - The saved scores are not needed for this step; nothing was written to `results/preds/`.

## Ensemble check on the saved scores (2026-11-17)
- Added `src/ensemble.py` (rank scores within fold or block, weight grid summing to 1, weight search on the time-CV fold scores, segment blend for known vs unknown customers, loaders that check the models' score files cover the same rows) and `notebooks/13_ensemble_check.ipynb`. Read-only: nothing is trained and nothing is written to `results/log.csv`; all weights are chosen on the fold scores and the validation block is used once to check them. Not part of the 6 × 4 grid; adding a blend before the freeze is a group decision.
- Results (G3; validation PR-AUC mean over seeds 0–2): one set of weights does not beat XGBoost alone (best fold weights 0.6409 vs 0.6434). A segment blend (known customers: XGBoost; unknown: CatBoost 0.6, LightGBM 0.2, XGBoost 0.1, random forest 0.1) scores 0.6551, 0.0056 above the best single cell (XGBoost on G1, 0.6495); the gain is all on unknown customers (0.5554 vs 0.5427), with overlapping intervals. The top ten unknown weightings span 0.6344–0.6353 on the fold scores, all with CatBoost 0.4–0.7.
- Per-fold check (section 5.1): the segment blend beats XGBoost alone in all four time-CV folds (by 0.0201 to 0.0473), and in all four when each fold's weights are chosen on the other three folds only (mean gain 0.0314); those searches pick the same weights each time.
- Checks: 6 new tests (`tests/test_ensemble.py`: grid size and sums, ranking within fold, the search finds the model that works on each segment, segment routing, fold mean, row and label checks on load).

## Step 19 code: final test scoring, built and not yet run (2026-11-17)
- Built (the test block has not been scored): `src/final_test.py`, `load_features_with_test()` in `src/feature_sets.py`, `tests/test_final_test.py`, `notebooks/12_test.ipynb`.
- `src/final_test.py`, one model per process (`.venv/bin/python -u -m src.final_test --model <name> --freeze-confirmed`): for each of the 12 cells (G0–G3 x seeds 0, 1, 2) it reads the settings and threshold from that cell's current validation row, refits on the train block with `experiment.fit_final` (same rows, seed and two-step early stopping; reuses a cached refit in `results/models/` when one exists, writes no new ones), checks the refit reproduces the logged validation PR-AUC (to 1e-6, else it stops and logs nothing), scores the test block with the stored threshold, logs one `split = "test"` row and saves the scores to `results/preds/<model>_<set>_test_seed<seed>.parquet`. Cells already logged are skipped. Without `--freeze-confirmed` it refuses to run. `--report` rebuilds `reports/test_results.md` from the log.
- `notebooks/12_test.ipynb` trains nothing: status of the 24 cells, test tables (mean ± sd over seeds), validation-to-test change and ranks, known vs unknown on test with bootstrap intervals. Until test rows exist it reads nothing from the test block.
- Checks: 7 new tests (freeze guard leaves the log unchanged; settings, threshold and `es_setup` copied from the validation rows; one test row per cell and saved scores match it; rerun adds nothing; a refit that does not reproduce validation stops the run; a missing validation row stops the run; report tables; loader keeps the train and validation positions; LightGBM with a reused cached refit runs in a subprocess, as on macOS it can crash next to PyTorch in one process). Full suite passed. The notebook ran in its empty state, and its post-freeze cells ran on fake test rows and fake scores in a scratch folder (deleted afterwards).
- Segment blend adopted as an extra row beside the 6 x 4 grid (G3 only), fixed before the freeze in `src/ensemble.py`: known customers (uid in the train block) → XGBoost; unknown → CatBoost 0.6, LightGBM 0.2, XGBoost 0.1, random forest 0.1; scores ranked within the scored block, seed with seed; weights chosen on the time-CV fold scores in `13_ensemble_check`. No training: it combines the four models' saved G3 scores, after checking each file reproduces its logged PR-AUC.
- Blend validation rows logged (`.venv/bin/python -m src.ensemble --log-validation`, 3 rows, `model` segment_blend, threshold picked on validation as for every model): PR-AUC 0.6552, 0.6542, 0.6560 (mean 0.6551, as in notebook 13); `results/log.csv` 1,763 → 1,766 lines. Its test rows come from `python -m src.final_test --blend --freeze-confirmed` once the four input models' G3 test rows exist, with the stored threshold. Expected test rows: 75 (72 + 3). `12_test` shows the blend (G3 only) and a blend vs XGBoost G3/G1 table.
- 4 more tests for the blend (fixed weights; validation rows and resume; inputs must reproduce the log; test rows need the freeze and a validation row and use its threshold).
- Fix: `--model` and `--blend` runs no longer write `reports/test_results.md`; only `--report` does, once all test rows are merged. Otherwise each person's run would write a report with only their models and the PRs would conflict on it. A model run now changes only `results/log.csv` in git (scores go to the gitignored `results/preds/`). 1 new test (runs write the report only with --report).
- Slip while building: an unquoted shell heredoc ran `python -m src.final_test --blend --freeze-confirmed` once by command substitution. It stopped at its first check (no XGBoost test row), before reading any scores or labels; nothing was logged or saved (checked: 0 test rows, no test files, no report).
- While building, the notebook's first draft printed the test block's fraud count and known-customer share once (no model was scored); it now reads nothing from the test block before the freeze.

## Step 19: Final test scoring (2026-11-18)
- Ran after the experiment freeze. Each owner ran `src.final_test --model <name> --freeze-confirmed` for their own model and merged the test rows (logreg, CatBoost and the blend: Dominic Bryan; LightGBM: Damien Koh; XGBoost: Mabel-You; random forest and MLP: jarrenoh). `notebooks/12_test.ipynb` was executed top to bottom (728d057). `reports/test_results.md` was built from the log with `.venv/bin/python -m src.final_test --report`.
- Checks:
  - The log has exactly 72 test rows, one per cell and seed, plus 3 for the segment blend on G3 (75 expected, 75 present after `experiment.current_runs`): pass.
  - The settings, `es_setup` and threshold in every test row match the cell's validation row (75 of 75): pass. `final_test` also stops unless the refit reproduces the logged validation PR-AUC.
  - The 6 x 4 test table, the known vs unknown table and the validation-to-test change are built from the log in `12_test` (all 6 code cells ran, no errors) and in `reports/test_results.md`: pass.
- Test PR-AUC (mean over seeds 0–2, from `reports/test_results.md`): the segment blend scores highest, 0.6100 on G3, 0.0293 above the best single cell, LightGBM on G1 (0.5807). XGBoost, best on validation (G1 0.6495), scores 0.5711 on G1 and ranks 5th–7th on G1–G3. Then come CatBoost (0.5359–0.5459), random forest (0.5295–0.5348), MLP (0.4554–0.4718) and logistic regression (0.1572–0.1681).
- Validation to test: every model loses PR-AUC except the MLP (+0.0231 to +0.0317). The drops are: logistic regression −0.18 to −0.19, XGBoost −0.07 to −0.08, LightGBM −0.04 to −0.05, random forest −0.01 to −0.03, CatBoost −0.00 to −0.02, and the blend −0.0451. Logistic regression's seeds score the same (sd 0.0000), so its drop is not seed noise. G2 → G3 adds nothing on test for any model (−0.0058 to +0.0026).
- Known vs unknown on test (G3; 22,695 known rows with 385 fraud, 36,359 unknown rows with 1,828 fraud):
  - The tree boosters score known customers well above unknown ones: XGBoost 0.7973 vs 0.5288, LightGBM 0.7461 vs 0.5535.
  - CatBoost, random forest, the MLP and logistic regression score unknown customers higher. Logistic regression falls to 0.0380 on known customers.
  - The blend uses XGBoost's scores on known customers (0.7973) and has the best unknown-customer score (0.5742 [0.551, 0.597]). That is above CatBoost's 0.5629, but the intervals overlap.
- Open: Context.md does not record the freeze date. The 3 blend test rows came in with CRLF line endings: `src/results.py` writes with `csv.DictWriter`'s default `\r\n`, while the rest of the log is LF.

## Step 19 follow-up: README and 12_test findings (2026-11-20)
- `notebooks/12_test.ipynb` section 6: wrote the findings in place of the placeholder (best on test, models vs feature sets, validation to test, known vs unknown customers, precision and recall at the stored threshold, the blend vs XGBoost). Only that markdown cell changed; the outputs are those of the run in 728d057. Every number is quoted from the notebook's tables, plus the validation known vs unknown table in `10_model_comparison` section 3.
- `README.md`: removed stale text (XGBoost "being added", the V-column list "pending group approval", "a possible ensemble", the test block "not scored until the freeze"); added `13_ensemble_check` and `12_test` to the reading order, and new sections 14 (segment blend) and 15 (final test scoring: `src.final_test --model / --blend / --report`, the reproduction check, 75 test rows). The tests section is now 16.
- `Context.md` (needs the group's agreement): removed the Roles section; added a Segment blend section, the freeze and test-scoring dates, the leakage-check result, the new files in the repo layout and the actual timeline; the open questions (freeze date, roles, ensemble) moved to settled.
- `BUILD_STEPS.md` (needs the group's agreement): each step names its stage (EDA, feature engineering, modelling, evaluation and so on) instead of a role; Part 3 is marked finished; Step 19 now covers the segment blend, `src.final_test`, the reproduction check and 75 test rows.
- `src/`: removed the "Role A/B/C:" prefix from 12 module docstrings; no code changed.
- Checks: full test suite passes (143 passed); no log or notebook output changed.

## Cleanup before submission: remove the F0–F2 first-pass files (2026-11-20)
- Removed: `src/train_trees.py`, `src/train_mlp.py`, `src/train_catboost.py`, `src/summarise.py`; `results/step9_manifest.json`, `step10_manifest.json`, `step12_manifest.json`, `catboost_manifest.json`; `config/best_params.yaml`, `catboost_best_params.yaml`, `features_v1.yaml`; `reports/grid_validation.md`, `catboost_sensitivity_F2.csv`, `catboost_sensitivity_F2.md`; `notebooks/archive/` (5 notebooks and its README); `tests/test_summarise.py`, and the 7 tests in `test_catboost.py`, `test_mlp.py` and `test_tree_models.py` that exercised the removed scripts (plus imports they alone used).
- Kept: the F0–F2 rows in `results/log.csv` (left out of every G0–G3 table), `config/features.yaml` (still read by `src/data.py`, logistic regression and current tests), `src/features.py` (uid check and the F1/F2 builders still used by tests) and these PROGRESS entries. Everything removed stays in the git history.
- Docs: README sections 7 and 9–12 (the F0–F2 commands) replaced by a short "Earlier first pass" section; later sections renumbered (customer ID check 7, leakage check 8, segment blend 9, final test 10, tests 11). `BUILD_STEPS.md` Part 2 marks the removed files; `CLAUDE.md` no longer says they stay.
- Checks: 132 tests pass (143 before; the 11 removed tested only deleted code). No other file references a removed file except the F0–F2 log rows' params.
- Fix found while checking: `notebooks/11_leakage_check.ipynb` asserted that the whole log has no test rows (a pre-freeze guard from Step 18), so it failed on every rerun once the Step 19 test rows were merged. The guard now checks only the rows the notebook reads (train-block CV rows). Rerun top to bottom: no errors, and every table and printout matches the committed outputs. Also checked on a clean clone: all 25 `src` modules import, 132 tests pass, `src.final_test --report` and `src.leakage --report` rebuild their reports unchanged, and `10_model_comparison`, `12_test` and `13_ensemble_check` rerun without errors.
