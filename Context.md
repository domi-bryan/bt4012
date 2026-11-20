# Context: BT4012 Group 11 Fraud Detection Project

Read this before working on any part of the project. It records what the group decided and why, so code and report text stay consistent with the approved proposal and the TA's feedback.

Updated 2026-11-17 for the `features_v2` table, its G0–G3 feature sets and six models, and on 2026-11-20 for the experiment freeze, the segment blend and the final test scoring. The earlier plan (feature sets F0–F2, four models) was run as a first pass in October; it is recorded in `PROGRESS.md` and Part 2 of `BUILD_STEPS.md`.

## Project

- **Course:** BT4012 Fraud Analytics (NUS).
- **Team:** Damien Koh, Jarren Oh, Mabel You, Dominic Bryan.
- **Title:** Benchmarking Various Fraudulent e-Commerce Transaction Detection Methods.
- **Proposal:** https://docs.google.com/document/d/1tngV5MgwvL5uSp1BHHHMvFMtiYPxHm8grSQRcmBYr9c
- **Plan (Google Doc):** https://docs.google.com/document/d/1BnIaWixwG8-rUpJF8Vu6WKsmWJPDfN8-MOt4_7Yct54
- **Deliverables:** final report and a GitHub repo with code, seeds and a reproduction README.
- **Report due:** 20 Nov 2026.

## Research questions

1. **RQ1:** How do fraudulent transactions differ from legitimate ones?
2. **RQ2:** Which algorithm detects fraud best?
3. **RQ3:** Do more complex models beat simpler ones, or is a user profile enough? Why?
4. **RQ4:** How do hyperparameter changes affect prediction quality?

## Dataset

IEEE-CIS Fraud Detection (Vesta Corporation, Kaggle 2019). Use the **training files only**, since only they carry labels.

| Table | Rows | Contents |
| --- | --- | --- |
| train_transaction | 590,540 (20,663 fraud, 3.5%) | TransactionDT, TransactionAmt, ProductCD, card1-6, addr1-2, dist1-2, P/R email domains, C1-C14 (counts), D1-D15 (time deltas), M1-M9 (match flags), V1-V339 (Vesta-engineered) |
| train_identity | 144,233 | id_01-id_38, DeviceType, DeviceInfo |

Facts that shape the work:

- Left-join identity onto transaction by TransactionID. Most rows have no identity record, so missingness is itself a signal.
- TransactionDT is seconds from an unknown reference and spans about 6 months. It is the time axis for every split.
- card1 has about 13,553 distinct values and acts as a coarse card fingerprint.
- D1 is days since a card-related event. Transaction day minus D1 gives an approximate account-open day that is stable within a customer.
- Vesta labels later transactions linked to a charged-back card's account, email or address as fraud too, so fraud clusters by customer. This is the root of the **leakage** risk below.
- The joined table is 590,540 x 436. Numbers are stored as float32 and text as `category`.

## Locked decisions

The TA approved the topic, dataset and methodology, then asked for three fixes (split, leakage, models). The group has since settled the feature sets, the tuning rules and the items marked "confirmed 2026-11-17". Change any of these only with the group's agreement.

### Split

Chronological 80/10/10, sorted by TransactionDT:

- **Train:** first 80% (472,432 rows). Hyperparameter tuning uses expanding-window CV inside it (30-day bins, train on bins 1..k, validate on bin k+1, 4 folds).
- **Validation:** next 10% (59,054 rows). Used for threshold choice and for comparing models and feature sets.
- **Test:** last 10% (59,054 rows). Scored once, on 17–18 Nov 2026, after the group agreed the experiment freeze on 17 Nov. Nothing was tuned, re-thresholded or chosen on it.

Random K-fold is a comparison in the leakage check only, never the main scheme.

### Leakage

Every feature and every fold must use only information available at transaction time.

- History features are computed from **prior** transactions only.
- No target encoding of card1 or uid. uid is a grouping key and is never a model input.
- Every fitted transform (imputer, scaler, encoder, V-column pruning) is fit on training rows only, and refit on each fold's training part.
- Resampling and class weighting apply inside a fold's training part only. Validation and test keep the true 3.5% fraud rate.
- Early stopping uses rows held out of the training part, never the rows a model is scored on (see Tuning and fitting).
- **Leakage check (report table):** run LR and LightGBM on G3 with their chosen settings under random stratified K-fold, StratifiedGroupKFold on uid, and time-ordered CV. Report PR-AUC for each. Random K-fold scores highest because the model recognises customers it has already seen (LightGBM on G3: random 0.8408, grouped by uid 0.6018, time CV 0.5870; `reports/leakage_check.md`). (On G3: confirmed 2026-11-17. The first-pass check on F2 is not used in the report.)
- **Known vs unknown customers:** report PR-AUC separately for known customers (uid seen in the train block) and unknown ones, on the validation block and on the test block. (Definition for the test block confirmed 2026-11-17.)

### Models

| Model | Kind | Input prep |
| --- | --- | --- |
| Logistic regression | Baseline | Median imputation, missing flags, scaling, one-hot or frequency-encoded categoricals; balanced class weights |
| Random Forest | Bagged trees | Missing values as -999, categoricals as integer codes; balanced class weights |
| LightGBM | Boosted trees | Native NaN and categoricals |
| XGBoost | Boosted trees | Native NaN and categoricals |
| CatBoost | Boosted trees | Native NaN; built-in one-hot encoding only, no target statistics |
| MLP | Deep learning (Lecture 4) | Same prep as LR; dropout and weight decay |

All six are full members of the benchmark. Every model gets the same input columns for a given feature set; only the representation differs.

### Segment blend

A blend of the tree models on G3, adopted before the freeze (`notebooks/13_ensemble_check.ipynb`, fixed in `src/ensemble.py`). Each model's scores are turned into percentile ranks within the scored block, and each transaction is scored by whether its customer is known (uid seen in the train block), which needs no label and is available when the transaction arrives:

- **Known customers:** XGBoost only.
- **Unknown customers:** CatBoost 0.6, LightGBM 0.2, XGBoost 0.1, random forest 0.1.

The weights were chosen from every weighting in steps of 0.1 by mean PR-AUC on the time-CV fold scores, never on the validation block, and checked on validation. The blend needs no training. Its F1 threshold is picked on validation and reused on test, like every model's. It is reported beside the 24 cells, not as a seventh model across all sets.

### Feature sets

Four cumulative sets from `data/processed/features_v2.parquet`, defined in `config/features_v2.yaml`:

| Set | Adds | Inputs | Question it answers |
| --- | --- | --- | --- |
| **G0 Raw** | Raw transaction and identity columns | 197 | Starting point |
| **G1 Row-level** | 27 features built from the transaction's own values | 224 | Does engineering a single transaction help? |
| **G2 Card and device history** | 10 past-only counts over the card, device and other values | 234 | Does knowing how common a value is help? |
| **G3 Customer history** | 18 past-only features over uid | 252 | Does a customer profile help? (RQ3) |

- **uid** = card1 + addr1 + floor(transaction day − D1). It is approximate; `reports/uid_check.md` validates it.
- **Raw inputs:** identifiers, raw time, uid, `id_12` and `id_27` are excluded, along with raw columns whose link to fraud reversed or faded over time.
- **V columns:** within each group of V columns that go missing together, drop a column if |Pearson r| > 0.75 with one already kept. The threshold was chosen in `notebooks/03_v_threshold.ipynb` and replaces the first proposal of 0.9.
- **Engineered features** are kept only if the EDA supports them and their link to fraud holds between the first and last 30 days of the train block.

The core experiment is a **6 models x 4 feature sets grid** (24 cells) on the validation block, scored once on the test block after the freeze, plus the segment blend on G3. RQ3 is answered by comparing the gain between models against the gain from G0 to G3, with G2 → G3 as the customer-profile step.

The report uses G0–G3 results only (confirmed 2026-11-17).

### Tuning and fitting

- Each model's settings are tuned once, on G3, with time CV and seed 0. The settings with the highest mean PR-AUC are then held fixed and validated on every set, so the sets are compared with the model unchanged.
- Search budgets: LR 11 values of C (grid), RF 12 settings (grid), MLP 16 (random search), LightGBM, XGBoost and CatBoost 20 each (random search).
- The three boosting models are kept on equal terms: 80% of rows and 80% of columns per tree, L2 on leaf weights tuned over {1, 3, 10}, at most 3,000 trees, early-stopping patience 50.
- Class weighting is tuned (unweighted or balanced) for the boosting models and the MLP. LR and RF always use balanced weights; state this as a limitation.
- Early stopping (boosting models and MLP): a tuning fit trains on the first 90% (by time) of its training part and stops on the last 10%. Validation and test fits then refit on the whole training part with the stopping point found.
- Validated models are fit on the train block with seeds 0, 1, 2. The decision threshold (best F1) is picked on the validation block and reused on the test block.

### Metrics

- **Primary:** PR-AUC (no-skill baseline = 0.035).
- **Also:** AUROC, precision and recall at the validation-chosen threshold, recall and precision in the top 1% of scores.
- Known vs unknown customer PR-AUC comes with 95% bootstrap intervals.
- Report mean and sample standard deviation over seeds 0, 1, 2.

### RQ4: effect of the settings

Read from the tuning runs, with no extra training: for each tuned setting and value, the mean, best and worst time-CV PR-AUC (section 5.5 of each model notebook). There are no separate one-at-a-time sweeps (confirmed 2026-11-17).

State the caveat in the report: LR and RF use full grids, so their values are compared on equal terms. The other four models use random search, so each value is paired with a different mix of other settings and the comparison is indicative.

### RQ1: feature importance

Permutation importance (drop in validation PR-AUC when one input column is shuffled), which means the same thing for every model, plus each model's built-in measure where it has one. SHAP is not used (confirmed 2026-11-17).

## Repo layout and interfaces

```
data/                    raw CSVs (gitignored), processed parquet
config/features_v2.yaml  G0-G3 column lists, exclusions, train-block constants
src/data.py              load, join, clean, tree-ready and linear-ready pipelines
src/domain_features.py   uid and the engineered fe_ columns
src/feature_selection.py V-column pruning, time-stability check
src/feature_sets.py      load_features() (test block removed), feature_list("G0".."G3"),
                         load_features_with_test() (final test scoring only)
src/split.py             holdout indices, CV fold generators
src/metrics.py           evaluate(y_true, y_score) -> dict
src/experiment.py        tune_and_validate, fit_final, current_runs, setting_sensitivity
src/segments.py          known vs unknown customers, bootstrap intervals
src/importance.py        permutation and built-in importance
src/ensemble.py          the segment blend (weights fixed before the freeze)
src/final_test.py        final test scoring: refit, reproduction check, test rows, report
src/models/              one file per model
notebooks/               01 EDA, 02 features, 03 V threshold, 04-09 one per model,
                         10 comparison, 11 leakage check, 12 test results,
                         13 ensemble check; shared logic lives in src/
results/log.csv          one row appended per run
reports/test_results.md  final test tables, built from the log
```

- `split.py` returns row indices, so every model trains and scores on identical rows.
- Every model module exposes `NAME`, `DEFAULTS`, `GRID`, `fit(X, y, X_val, y_val, params, seed)`, `predict_proba(model, X)` and `feature_importance(model, columns)`. Models that early-stop also set `EARLY_STOPPING` and expose `refit_params(model)`.
- `evaluate()` returns the metrics listed above.
- `results/log.csv` columns: date, author, model, feature_set, cv_scheme, split (cv/val/test), seed, params (JSON), pr_auc, auroc, precision, recall, recall_top1pct, runtime_s. The fold, the threshold and fit details sit inside `params`.
- Summaries read the log through `experiment.current_runs`, which leaves out early-stopping runs from before the refit rule.

## Timeline

| Dates | Milestone |
| --- | --- |
| Oct 1-4 | Repo, data load and join, data profile, audit, cleaning, EDA |
| Oct 18-22 | First pass on F0–F2: splits, metrics, pipelines, four models, leakage check |
| Nov 14 | Move to `features_v2`, G0–G3 and six models; shared run code and model notebooks |
| Nov 15-16 | Model runs on G0–G3 |
| Nov 17 | Model comparison, leakage check on G3, ensemble check and segment blend; experiment freeze agreed |
| Nov 17-18 | Final test scoring of all six models and the blend (75 test rows) |
| Nov 18-20 | Test findings, README and documentation; write and review the report |
| Nov 20 | Report due |

## Report requirements

The TA asked for detail and justification in every section. Each section states what was done and why:

- Dataset: why Vesta, size, variables, table structure, partial identity coverage.
- Preprocessing: each step and its reason, per model family.
- Features: what each engineered feature captures and how leakage is prevented.
- Experimental setup: the chronological split, why random K-fold misleads here, why PR-AUC leads.
- Models: why each suits imbalanced tabular fraud data.

Discussion of RQ2 and RQ3, limitations and conclusion are written together.

Fix in proposal wording: "train/validation/split" should read "train/validation/test".

## Settled questions

- Report deadline: 20 Nov 2026.
- Experiment freeze: agreed 17 Nov 2026; the test block was scored on 17–18 Nov.
- Ensemble: the segment blend above, chosen on the time-CV fold scores before the freeze.
- RQ3's "user profile" means the customer history features that G3 adds to G2.
- The results log is `results/log.csv` in the repo, shared by all members.
- Compute runs on members' own machines.
