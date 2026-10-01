# Progress log: BT4012 Group 11 fraud detection (one entry per finished step)

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
