# BT4012 Group 11: Fraud Detection on IEEE-CIS (Vesta)

Group coursework benchmarking fraud detection models. Four students share this repo; you help one of them at a time.

## Read first

- `Context.md`: the group's locked decisions (split, leakage rules, models, feature sets, metrics, roles). Treat it as the spec. A change to it needs the group's agreement, so propose it to the user instead of editing it.
- `BUILD_STEPS.md`: the ordered build. Work one step at a time, in order, unless the user names a different step.
- `PROGRESS.md`: what has been done so far. Read it to find the next step.

## How to work

- Finish a step, check its "Done when" list item by item, append an entry to `PROGRESS.md`, then stop and report to the user. Start the next step only when the user says so.
- Shared logic lives in `src/` and is imported by notebooks. Notebooks hold EDA and plots.
- Every training run appends one row to `results/log.csv` through `src/results.py`.
- Seeds are 0, 1, 2. Set them for numpy, sklearn, lightgbm and torch.
- Load data as float32 and categoricals as `category`; the joined table is about 590k x 430.

## Guardrails

- **Test block:** the last 10% by time stays unscored until the user says the experiment freeze has happened. Use train-block CV and the validation block for every decision before then.
- **Leakage:** every feature uses only transactions earlier in time than the row it describes. Every fitted transform (imputer, scaler, encoder) is fit on training rows only.
- **Reported numbers:** report text and tables quote only values present in `results/log.csv`. When a result is missing, say so.
- Raw data in `data/raw/` is read-only and gitignored.
