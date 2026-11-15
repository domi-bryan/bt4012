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

