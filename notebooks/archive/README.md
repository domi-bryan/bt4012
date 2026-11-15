# Archived notebooks

Kept for reference only. They were replaced as the project moved to the `features_v2` feature set and
its G0–G3 feature groups, and they are not meant to be rerun: their path setup assumes they sit directly in
`notebooks/`, and some read files that are being retired (`config/features.yaml`,
`config/features_v1.yaml`, `data/processed/features_v1.parquet`). Their results are still in
`results/log.csv`.

| Notebook | What it was | Replaced by |
| --- | --- | --- |
| `02_baseline.ipynb` | Results of the logistic regression baseline on F0 (Step 6), read from the log | the per-model notebooks on G0–G3 |
| `03_trees.ipynb` | Results of LR, random forest and LightGBM on F0–F2 (Step 9), read from the log | the per-model notebooks on G0–G3 |
| `04_mlp.ipynb` | Results and training curves of the MLP on F0–F2 (Step 10), read from the log | the per-model notebooks on G0–G3 |
| `xgboost.ipynb` | XGBoost trained in the notebook on `features_v1`: feature-set comparison, ablations, tuning | the per-model XGBoost notebook on G0–G3 |
| `xgboost_v_threshold.ipynb` | XGBoost sweep of the V-column correlation threshold | `notebooks/03_v_threshold.ipynb` (logistic regression, r = 0.75) |

The F0–F2 validation results are summarised in `reports/grid_validation.md`.
