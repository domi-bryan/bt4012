# Context: BT4012 Group 11 Fraud Detection Project

Read this before working on any part of the project. It records what the group decided and why, so code and report text stay consistent with the approved proposal and the TA's feedback.

## Project

- **Course:** BT4012 Fraud Analytics (NUS).
- **Team:** Damien Koh, Jarren Oh, Mabel You, Dominic Bryan. Roles A to D are not yet assigned to names.
- **Title:** Benchmarking Various Fraudulent e-Commerce Transaction Detection Methods.
- **Proposal:** https://docs.google.com/document/d/1tngV5MgwvL5uSp1BHHHMvFMtiYPxHm8grSQRcmBYr9c
- **Plan (Google Doc):** https://docs.google.com/document/d/1BnIaWixwG8-rUpJF8Vu6WKsmWJPDfN8-MOt4_7Yct54
- **Deliverables:** final report and a GitHub repo with code, seeds and a reproduction README.

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
- Full table is roughly 590k x 430 columns. Downcast to float32 to fit in Colab or Kaggle RAM.

## Locked decisions (from TA feedback)

The TA approved the topic, dataset and methodology, then asked for three fixes. These are settled; change them only with the group's agreement.

### Split

Chronological 80/10/10, sorted by TransactionDT:

- **Train:** first 80%. Hyperparameter tuning uses expanding-window CV inside it (monthly chunks, train on months 1..k, validate on k+1, about 4 folds).
- **Validation:** next 10%. Used for early stopping, threshold choice and model selection.
- **Test:** last 10%. Scored once, after the experiment freeze.

Random K-fold is a comparison in the leakage check only, never the main scheme.

### Leakage

Every feature and every fold must use only information available at transaction time.

- Aggregates are computed from **prior** transactions only.
- No target encoding of card1 or uid.
- Resampling and class weighting apply inside a fold's training part only. Validation and test keep the true 3.5% fraud rate.
- **Leakage check (report table):** run LR and LightGBM on the same features under random stratified K-fold, StratifiedGroupKFold on uid, and time-ordered CV. Report PR-AUC for each. Random K-fold is expected to score highest because the model recognises customers it has already seen.
- On the test block, report PR-AUC separately for returning customers (uid seen in train) and new customers.

### Models

| Model | Role | Input prep |
| --- | --- | --- |
| Logistic regression | Baseline | Imputation, missing flags, scaling, encoded categoricals, class weights |
| Random Forest | Bagged trees | Imputation or sentinel values, class weights |
| LightGBM | Boosted trees, expected strongest | Native NaN and categoricals; scale_pos_weight |
| MLP | Deep learning (Lecture 4) | Same prep as LR; dropout, weighted BCE loss |

XGBoost or CatBoost are optional extras.

### Feature sets

- **F0 Raw:** cleaned transaction and identity columns.
- **F1 Card:** F0 + aggregates grouped by card1.
- **F2 Customer:** F1 + the same aggregates grouped by uid.

**uid** = card1 + addr1 + floor(transaction day − D1). It is approximate; validate it (group count, group size distribution, fraud rate per group) before relying on it.

Aggregates: transaction count, amount mean and std, amount relative to group mean, seconds since previous transaction, counts in trailing windows, distinct devices, emails and addresses.

The core experiment is a **4 models x 3 feature sets grid** (12 cells). RQ3 is answered by comparing the gain from LR to LightGBM against the gain from F0 to F2.

### Metrics

- **Primary:** PR-AUC (no-skill baseline = 0.035).
- **Also:** AUROC, precision and recall at the validation-chosen threshold, recall in the top 1% of scores.
- Stochastic models run on seeds 0, 1, 2; report mean and std.

### RQ4 sweeps

One parameter at a time around each tuned model, PR-AUC on validation:

- LightGBM: num_leaves, learning_rate, min_child_samples, scale_pos_weight
- LR: C
- RF: max_depth, n_estimators
- MLP: hidden size, dropout, learning rate

## Repo layout and interfaces

```
data/            raw CSVs (gitignored), processed parquet
src/data.py      Role A: load, join, clean, tree-ready and linear-ready pipelines
src/features.py  Role B: uid, F0/F1/F2 builders
src/split.py     Role C: holdout indices, CV fold generators
src/metrics.py   Role C: evaluate(y_true, y_score) -> dict
src/models/      one file per model
notebooks/       EDA and plots only; shared logic lives in src/
results/log.csv  one row appended per run
```

- `split.py` returns row indices, so every model trains and scores on identical rows.
- Every model module exposes `fit(X, y, X_val, y_val, params)` and `predict_proba(X)`.
- `evaluate()` returns the metrics listed above.
- `results/log.csv` columns: date, author, model, feature_set, cv_scheme, split (cv/val/test), seed, params (JSON), pr_auc, auroc, precision, recall, recall_top1pct, runtime_s.

## Roles

| Role | Owns | Report sections |
| --- | --- | --- |
| A: Data pipeline + MLP | Join, cleaning, missingness, V-column pruning, both pipelines, MLP, README | Dataset, Preprocessing, MLP; final edit |
| B: EDA + features | RQ1 EDA, uid, F1/F2 aggregates, feature dictionary, SHAP | EDA, Feature engineering, RQ1 |
| C: Validation + baseline | split.py, metrics.py, results log, LR, leakage check, returning vs new analysis, results tables | Experimental setup, Results |
| D: Tree models + tuning | RF, LightGBM, tuning, RQ4 sweeps | Models, RQ4 |

Discussion of RQ2 and RQ3, limitations and conclusion are written together.

## Timeline

Assumes the report is due about 13 Nov 2026; confirm the real deadline.

| Dates | Milestone |
| --- | --- |
| Oct 1-7 | Kickoff, repo, interfaces agreed |
| Oct 10 | split.py, metrics.py, LR on F0 ready (C) |
| Oct 14 | Clean data and pipelines ready (A); EDA and uid done (B) |
| Oct 21 | F1/F2 feature tables ready (B); RF and LightGBM on F0 done (D) |
| Oct 22 - Nov 4 | Full grid, leakage check, tuning, RQ4 sweeps, MLP, SHAP |
| Nov 4 | Experiment freeze; test block scored |
| Nov 5-11 | Write and review |
| ~Nov 13 | Report due |

## Report requirements

The TA asked for detail and justification in every section. Each section states what was done and why:

- Dataset: why Vesta, size, variables, table structure, partial identity coverage.
- Preprocessing: each step and its reason, per model family.
- Features: what each aggregate captures and how leakage is prevented.
- Experimental setup: the chronological split, why random K-fold misleads here, why PR-AUC leads.
- Models: why each suits imbalanced tabular fraud data.

Fix in proposal wording: "train/validation/split" should read "train/validation/test".

## Open questions

- Actual report deadline.
- Who takes Roles A to D.
- Where compute runs (Colab, Kaggle, or a 16 GB+ machine).
- Whether RQ3's "user profile" means the uid aggregates in F2 (current assumption: yes).
- Weekly check-in time and where the results log lives.
