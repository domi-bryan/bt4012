# BT4012 Group 11: Fraud Detection on IEEE-CIS (Vesta)

Group coursework benchmarking six fraud detection models on four cumulative feature sets, G0–G3. Four students share this repo; you help one of them at a time.

## Read first

- `Context.md`: the group's locked decisions (split, leakage rules, models, feature sets, tuning rules, metrics, roles). Treat it as the spec. A change to it needs the group's agreement, so propose it to the user instead of editing it.
- `BUILD_STEPS.md`: the ordered build. Part 3 (Steps 13–19) is the current G0–G3 plan. Work one step at a time, in order, unless the user names a different step. In Step 16 each model notebook is one unit of work.
- `PROGRESS.md`: what has been done so far. Most G0–G3 work was logged without step numbers, so find the next step by checking each Part 3 step's "Done when" list against the entries named under its "Earlier entries".

## How to work

- Finish a step, check its "Done when" list item by item, append an entry to `PROGRESS.md` (`## Step N: <name>`), then stop and report to the user. Start the next step only when the user says so.
- Shared logic lives in `src/` and is imported by notebooks. Notebooks `04`–`09` each run one model through `src.experiment.tune_and_validate`; the others hold EDA, feature building and comparisons.
- Model work loads data with `src.feature_sets.load_features()`, which returns `features_v2` with the test block already removed, and takes columns from `feature_list("G0")` to `("G3")`.
- Every training run appends one row to `results/log.csv` through `src/results.py`. A trial run (smaller grid or a data sample) logs to a scratch file.
- Seeds are 0, 1, 2. Set them for numpy, sklearn, lightgbm, xgboost, catboost and torch.
- Load data as float32 and categoricals as `category`; the joined table is 590k x 436.
- Train one model library per Python process and import it before anything else. On macOS, LightGBM, XGBoost and PyTorch each ship their own OpenMP runtime and crash when mixed.
- The F0–F2 first pass (`BUILD_STEPS.md` Part 2) is history. Its log rows stay as they are; its scripts, settings, manifests, summaries and notebooks were removed before submission (they remain in the git history).

## Guardrails

- **Test block:** the last 10% by time stays unscored until the user says the experiment freeze has happened. Use train-block CV and the validation block for every decision before then.
- **Leakage:** every feature uses only transactions earlier in time than the row it describes. Every fitted transform (imputer, scaler, encoder) is fit on training rows only. Early stopping uses rows held out of the training part, never the rows being scored.
- **Reported numbers:** report text and tables quote only G0–G3 values present in `results/log.csv`, read through `experiment.current_runs` so superseded runs are left out. When a result is missing, say so.
- Raw data in `data/raw/` is read-only and gitignored.
