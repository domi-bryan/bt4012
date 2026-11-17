"""V-column correlation threshold check for the linear models (LR and MLP).

The proposed V list drops one of any pair of V columns with |r| > 0.90. XGBoost was shown not
to care where that threshold sits between 0.75 and 1.00; this module asks the same question of
the two models that are most sensitive to correlated inputs.

For each threshold the V list is rebuilt with the Step 2 rule (`src.data_profile.prune_group`)
on the train block only, then LR and the MLP are scored on F2 with their tuned configs from
`config/best_params.yaml` under the usual time CV. LR is deterministic (seed 0); the MLP runs
seeds 0, 1, 2. Every fit appends one row to results/log.csv with feature_set `F2_vr<threshold>`.
The validation and test blocks are never loaded into a model.

Run `.venv/bin/python -u -m src.v_threshold_linear` (resumable), then `--report` to rebuild
reports/v_threshold_linear.md from the log alone.
"""
import argparse
import gc
import json

import numpy as np
import pandas as pd
import torch  # imported before sklearn-heavy modules: two OpenMP runtimes can crash on macOS
import yaml

from src import data, experiment
from src.data_profile import prune_group, v_groups
from src.features import build_feature_set
from src.models import logreg, mlp
from src.results import LOG_PATH
from src.split import holdout_indices, time_cv_folds

THRESHOLDS = (0.75, 0.80, 0.85, 0.90)
BASELINE = 0.90
BASE_SET = "F2"
MODELS = {"logreg": logreg, "mlp": mlp}
NAMES = {"logreg": "Logistic regression", "mlp": "MLP"}
SEEDS = {"logreg": (0,), "mlp": (0, 1, 2)}
THREADS = 4
BEST_PATH = data.ROOT / "config" / "best_params.yaml"
REPORT_PATH = data.ROOT / "reports" / "v_threshold_linear.md"


def label(threshold):
    return f"{BASE_SET}_vr{round(threshold * 100):03d}"


def v_lists(train, thresholds=THRESHOLDS):
    """{threshold: (keep, drop)} from the Step 2 rule applied to `train` (train-block rows)."""
    groups = v_groups(train)
    out = {}
    for t in thresholds:
        keep, drop = [], []
        for cols in groups.values():
            k, d, _ = prune_group(train, cols, threshold=t)
            keep += k
            drop += d
        out[t] = (keep, drop)
    return out


def tuned_configs(path=BEST_PATH):
    models = yaml.safe_load(path.read_text())["models"]
    return {name: models[name][BASE_SET]["params"] for name in MODELS}


def run(thresholds=THRESHOLDS, log_path=LOG_PATH):
    torch.set_num_threads(THREADS)
    raw = data.load_cleaned()
    train_idx, _, _ = holdout_indices(raw)
    raw = raw.iloc[:len(train_idx)]          # time CV needs the train block only
    lists = v_lists(raw, thresholds)
    if BASELINE in lists and set(lists[BASELINE][1]) != set(data.load_v_columns()[1]):
        raise ValueError(f"Threshold {BASELINE} does not reproduce config/features.yaml; "
                         "the pruning rule or the data changed.")
    for t in thresholds:
        print(f"r={t:.2f}: keep {len(lists[t][0])} V columns, drop {len(lists[t][1])}", flush=True)
    configs = tuned_configs()
    df = build_feature_set(raw, BASE_SET)
    del raw
    for t in thresholds:
        keep, drop = lists[t]
        make_pipeline = lambda drop=drop: data.LinearPipeline(v_drop=drop)
        for name, model in MODELS.items():
            params = {**configs[name], "v_threshold": t, "n_v_kept": len(keep)}
            for seed in SEEDS[name]:
                print(f"\n{name} r={t:.2f} seed {seed}", flush=True)
                experiment.time_cv(model, make_pipeline, df, train_idx, [params], label(t),
                                   seed=seed, log_path=log_path, resume=True)
                gc.collect()


def threshold_rows(log, model_name):
    """One row per logged fit of this experiment for one model: threshold, n_v_kept, fold, seed, metrics."""
    rows = log[(log.model == model_name) & (log.split == "cv") & (log.cv_scheme == "time")
               & log.feature_set.astype(str).str.startswith(f"{BASE_SET}_vr")]
    params = rows["params"].map(json.loads)
    return rows.assign(v_threshold=params.map(lambda p: p["v_threshold"]),
                       n_v_kept=params.map(lambda p: p["n_v_kept"]),
                       fold=params.map(lambda p: p["fold"]))


def threshold_table(log, model_name, n_folds):
    """Mean PR-AUC per threshold and the paired change against the baseline threshold.

    Pairs are matched on (fold, seed). Raises unless every threshold has exactly one row per
    fold and seed, so partial or repeated runs are never averaged in silently.
    """
    rows = threshold_rows(log, model_name)
    expected = n_folds * len(SEEDS[model_name])
    base = rows[np.isclose(rows.v_threshold, BASELINE)].set_index(["fold", "seed"])["pr_auc"]
    out = []
    for t, part in rows.groupby("v_threshold"):
        if len(part) != expected or part.duplicated(["fold", "seed"]).any():
            raise ValueError(f"{model_name} r={t}: expected {expected} runs (one per fold and "
                             f"seed), found {len(part)}")
        delta = (part.set_index(["fold", "seed"])["pr_auc"] - base).dropna()
        is_base = np.isclose(t, BASELINE)
        out.append({
            "v_threshold": t, "n_v_kept": int(part.n_v_kept.iloc[0]), "runs": len(part),
            "pr_auc_mean": part.pr_auc.mean(), "pr_auc_std": part.pr_auc.std(ddof=1),
            "auroc_mean": part.auroc.mean(),
            "delta": np.nan if is_base else delta.mean(),
            "delta_se": np.nan if is_base else delta.std(ddof=1) / np.sqrt(len(delta)),
            "pairs_better": None if is_base else f"{int((delta > 0).sum())}/{len(delta)}",
        })
    if base.empty:
        raise ValueError(f"{model_name}: no runs at the baseline threshold {BASELINE}")
    return pd.DataFrame(out).sort_values("v_threshold").reset_index(drop=True)


def render(tables, per_fold):
    lines = [
        "# V-column threshold check for LR and the MLP",
        "",
        "Generated by `.venv/bin/python -m src.v_threshold_linear --report` from `results/log.csv`. "
        f"Each threshold rebuilds the V list with the Step 2 rule on the train block; models are "
        f"scored on {BASE_SET} with their tuned configs under time CV (4 expanding folds inside the "
        "train block). LR runs seed 0 (it is deterministic); the MLP runs seeds 0, 1, 2. The "
        "validation and test blocks are not used.",
        "",
        f"**Δ vs r = {BASELINE:.2f}** is the mean of (this threshold − r = {BASELINE:.2f}) over "
        "matching fold and seed pairs, with the standard error of that mean. A Δ within about two "
        "standard errors of zero means the data cannot tell the two thresholds apart.",
    ]
    for name, table in tables.items():
        lines += ["", f"## {NAMES[name]}", "",
                  f"| r | V kept | Runs | Mean PR-AUC | Std | Mean AUROC | Δ vs r = {BASELINE:.2f} | SE of Δ | Pairs better |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in table.itertuples():
            base = pd.isna(r.delta)
            lines.append(
                f"| {r.v_threshold:.2f} | {r.n_v_kept} | {r.runs} | {r.pr_auc_mean:.4f} | "
                f"{r.pr_auc_std:.4f} | {r.auroc_mean:.4f} | "
                + ("baseline | | |" if base else
                   f"{r.delta:+.4f} | {r.delta_se:.4f} | {r.pairs_better} |"))
        lines += ["", "Mean PR-AUC per fold (averaged over seeds):", "",
                  "| r | " + " | ".join(f"Fold {k}" for k in per_fold[name].columns) + " |",
                  "| --- |" + " ---: |" * per_fold[name].shape[1]]
        for t, row in per_fold[name].iterrows():
            lines.append(f"| {t:.2f} | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    return "\n".join(lines) + "\n"


def write_report(log_path=LOG_PATH, path=REPORT_PATH, n_folds=None):
    log = pd.read_csv(log_path)
    if n_folds is None:
        cols = ["TransactionDT", "TransactionDay", "isFraud"]
        frame = pd.read_parquet(data.CLEANED_PATH, columns=cols)
        n_folds = len(time_cv_folds(frame, holdout_indices(frame)[0]))
    tables = {name: threshold_table(log, name, n_folds) for name in MODELS}
    per_fold = {name: threshold_rows(log, name).pivot_table(index="v_threshold", columns="fold",
                                                           values="pr_auc", aggfunc="mean")
                for name in MODELS}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(tables, per_fold))
    return tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="rebuild the report from the log only")
    args = parser.parse_args()
    if not args.report:
        run()
    for name, table in write_report().items():
        print(f"\n{NAMES[name]}\n{table.to_string(index=False, float_format='%.4f')}")
