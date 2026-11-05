# CatBoost extension

This contribution starts from PR [#1](https://github.com/domi-bryan/bt4012/pull/1),
`jarren/steps-3-11`, head `13a1e65628c87c573c76c4e6a56be702381c5aae`
(5 November 2026). At inspection the PR was open, not merged, mergeable, and its
merge state was `blocked`. Its description covers the audit, cleaning and Steps
3–12; its 63 changed files supply the shared interfaces this extension imports.
The latest commit is “Keep figures and notebook outputs out of git.”

## Run

From this branch's worktree:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-catboost.txt
.venv/bin/python -m src.train_catboost --mode tune
.venv/bin/python -m src.train_catboost --mode sensitivity
.venv/bin/python -m src.train_catboost --mode report
.venv/bin/python -m pytest -q tests/test_catboost.py
```

The default input is `data/processed/cleaned.parquet`. If it is absent, obtain the
two Kaggle training CSVs as described in README, then run `python -m src.data` and
`python -m src.clean_data`. Alternatively use `--data /path/to/cleaned.parquet`.
An existing processed file can be read from another worktree without copying or
overwriting it. No real training data was available for this contribution's checks.

Tuning defaults to F0, F1 and F2. Sensitivity and report default to F2; use
`--feature-sets F0 F1 F2` to choose other sets. Tuning evaluates a deterministic
sample of 20 configurations with search seed 0, selecting highest mean time-CV
average precision. It searches depth {4, 6, 8}, learning rate {0.03, 0.1}, L2
regularization {1, 3, 10}, and positive class weight {1, balanced}. Every fold
refits preprocessing on its training portion. The selected configuration is
refitted on the 80% train block for validation seeds 0, 1, 2.

Defaults are 1,500 trees, depth 6, learning rate 0.05, L2 regularization 3, six CPU
threads and early-stopping patience 50. Positive class weighting resolves to
negatives/positives from each training part. Early stopping uses a custom
unweighted `sklearn.metrics.average_precision_score`, matching `src.metrics`
rather than CatBoost's differently integrated built-in PRAUC. The best iteration
is restored. Validation also chooses the F1 threshold through the shared runner.

Sensitivity varies one parameter from the selected configuration, keeping the
others fixed, and refits all three seeds on train, scoring validation:

| Parameter | Values |
| --- | --- |
| depth | 3, 4, 6, 8, 10 |
| learning_rate | 0.01, 0.03, 0.05, 0.1, 0.3 |
| l2_leaf_reg | 0.1, 1, 3, 10, 30 |
| scale_pos_weight | 1, 3, 10, 30, 100, balanced |

The report command reconstructs every plotted point from logged validation runs,
rejecting missing or duplicated seeds. It writes
`reports/catboost_sensitivity_F2.csv`, `reports/catboost_sensitivity_F2.md` and
four `reports/figures/sweep_catboost_F2_*.png` plots (mean ± sample standard
deviation, with the tuned baseline marked). The balanced point means the actual
train-block class ratio, recorded in `params._fit`. Figures remain gitignored.

## Shared interfaces and encoding policy

- `src.features.build_feature_set` defines F0/F1/F2 without modification. The
  test block is removed before calling it. F1 adds the same card aggregates, and
  F2 adds the same uid aggregates, including the team's tied-timestamp convention.
- `holdout_indices` and `time_cv_folds` define the identical chronological blocks
  and expanding 30-day training folds. There is no random CV or test-scoring mode.
- `CatBoostPipeline` subclasses the shared `TreePipeline(kind="lgbm")`: same
  feature order, `feature_columns`, provisional V-column pruning, numeric NaN,
  excluded IDs/time/label and category vocabularies fitted on training rows.
- CatBoost receives strings for category columns, a collision-safe missing
  marker, and the shared treatment of unseen levels as missing. Integer-like
  numeric columns, including card1, stay numeric. Raw uid is rejected; only the
  shared uid aggregates appear in F2. Feature sets are never augmented with
  raw uid or account-open day by this extension.
- All string categories use CatBoost's **internal one-hot encoding**. The
  `one_hot_max_size` equals at least the maximum training category cardinality,
  so no categorical target statistics or category combinations are calculated.
  This conservative policy makes the group's prohibition on target encoding of
  card1/uid unambiguous, at the cost of CatBoost's usual target-statistic benefit.
  It is a CatBoost implementation difference from LightGBM's native categorical
  splits, not a change to F0/F1/F2. CPU training uses `has_time=True`, preserves
  input order and rejects policy overrides. The 65,535-category CPU limit fails
  explicitly if exceeded. See the [official one-hot and time-order parameters](https://catboost.ai/docs/en/references/training-parameters/common).
- `src.experiment.time_cv`, `best_params`, `fit_validate` and `logged_run` provide
  tuning, scoring, thresholds and resumability. `src.results.log_run` writes the
  existing ledger schema, with `model=catboost`.

## Outputs and resumption

`config/catboost_best_params.yaml` records selected configurations separately
from the team's existing model settings. `results/catboost_manifest.json` hashes
processed data, shared and new source files, the V-column configuration, search
policy and relevant installed versions. It refuses to resume on changed inputs
or to adopt existing CatBoost results without a manifest. Use all three flags
`--log`, `--best-params` and `--manifest` with fresh paths for a new experiment.

By default every fit appends to `results/log.csv`; use one writer at a time.
Regular CV rows record `params.fold`. Validation records `params.threshold`.
`params._fit` records best iteration, resolved class weight, categorical names,
one-hot threshold and early-stopping metric. Sensitivity rows also record
`params._sweep` with the parameter name and complete tuned baseline. This lets
the shared lookup distinguish sweeps from selected validation runs and resume
only missing fits. No real-data rows or fabricated scores were added by this
implementation; test fixtures use temporary logs.

## Remaining integration gaps

1. The real raw/processed data is absent locally. Full 20-config CV on all three
   feature sets, nine selected validation fits, and the 63 F2 sensitivity fits
   have not been run. There are no CatBoost benchmark scores to report yet.
2. The group's V-column proposal is still unapproved upstream. This extension
   uses exactly that provisional list with the shared warning. Approval or a
   changed list requires a fresh experiment if its file fingerprint changes.
3. The shared `src.summarise` intentionally contains the locked four-model grid;
   it excludes CatBoost rows. Extending that core table, the LR/LightGBM leakage
   comparison, SHAP analysis or final-test scoring is separate integration work.
   Sensitivity reporting is provided here. Do not score the reserved test block
   until the group explicitly confirms experiment freeze.
4. CatBoost uses validation for early stopping, sweeps and threshold selection;
   its validation scores are not independent final-test estimates. Timing has
   not been measured on the real dataset, and one-hot high-cardinality handling
   may differ substantially in accuracy and cost from default CatBoost CTRs.
5. Upstream tests import torch before LightGBM to avoid their macOS OpenMP
   conflict. This runner does not import LightGBM or `src.train_trees`. The
   optional dependency file installs the shared test dependencies as well.

## Verification on this machine

The focused suite passed: `python -m pytest -q tests/test_catboost.py` → **11 passed**.
The full suite passed: **81 passed, 11 skipped**. The skipped tests require real
joined/cleaned data, which is absent. This macOS environment lacks Homebrew's
`libomp`, so full-suite collection initially failed in the shared LightGBM import.
The installed PyTorch runtime resolves it for the test process without changing
system files:

```bash
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib" .venv/bin/python -m pytest -q tests
```

Checked with Python 3.12, CatBoost 1.2.10, pandas 3.1.1, NumPy 2.5.4,
scikit-learn 1.9.1, torch 2.15.0 and LightGBM 4.7.0. `pip check` passed. A fresh
Python process also trained through `src.experiment._fit_score` successfully,
and CLI help and the missing-data diagnostic were checked. CatBoost warns that
the optional numba metric optimization is unavailable; exact AP still runs.

Tests cover shared F0/F1/F2 columns and pruning, unseen/missing categories without
future vocabulary leakage, card1/uid restrictions, JSON model inspection for
absence of CTRs, deterministic scores, exact unweighted AP, training-only class
weights, time-CV and validation logging, interrupted-fold/seed recovery,
sensitivity provenance and reporting, manifest mismatch rejection and removing
test rows before shared feature construction. Tests leave the tracked results
ledger and existing model settings unchanged.

## Branch and eventual PR base

The local branch is `damien/catboost`, created in the separate worktree
`/private/tmp/bt4012-catboost`. The original worktree stays on `main`. Neither
the teammate's branch nor any remote branch was changed.

While PR #1 is open, the eventual CatBoost PR should target
**`jarren/steps-3-11`**, with an explicit “Depends on #1” note. Comparing against
that branch displays only this contribution; targeting main now would include
the entire shared implementation. No remote branch or PR has been created here.

After #1 merges, fetch origin. If `13a1e656` remains an ancestor of main (normal
merge), retarget the CatBoost PR to main and verify the Files changed view. If
the team squash-merges or rebases #1, first replay only the CatBoost commits:

```bash
git fetch origin
git rebase --onto origin/main 13a1e65628c87c573c76c4e6a56be702381c5aae damien/catboost
git diff --stat origin/main...damien/catboost
```

Do that rebase with a clean CatBoost worktree, saving any uncommitted work first.
If published, coordinate a `--force-with-lease` push of your CatBoost branch,
then retarget its PR to main. Never force-push the teammate's branch. If its
open branch advances first, rebase the CatBoost commits on its new head, review
shared interface changes and rerun these tests before continuing experiments.
After that rebase, substitute the newer team head for `13a1e656...` in the
`--onto` command above; the boundary must be the shared commit your work builds on.
