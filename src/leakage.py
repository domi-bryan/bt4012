"""Role C: leakage check (Step 12).

LR and LightGBM on F2 with their Step 9 best configs, scored on the train block under three
CV schemes. Time CV rows already exist in the log from Step 9 and are reused as they are;
this module adds random stratified 5-fold (cv_scheme "random") and StratifiedGroupKFold on
uid (cv_scheme "group_uid"), seed 0. Pipelines are refit on each fold's training part and
LightGBM early-stops on the fold's validation part, exactly as in time CV.

Run `.venv/bin/python -u -m src.leakage` (resumable), then `--report` to rebuild
reports/leakage_check.md from the log alone.
"""
import argparse
import gc
import json

import numpy as np
import pandas as pd
import yaml

from src import data, experiment
from src.features import build_feature_set, make_uid
from src.metrics import evaluate
from src.models import lgbm, logreg
from src.results import LOG_PATH, log_run
from src.split import group_cv_folds, holdout_indices, random_cv_folds
from src.train_trees import BEST_PATH, check_manifest

MODELS = {"logreg": logreg, "lgbm": lgbm}
NAMES = {"logreg": "LR", "lgbm": "LightGBM"}
FEATURE_SET = "F2"
SCHEMES = {"random": "Random stratified 5-fold", "group_uid": "StratifiedGroupKFold on uid (5)",
           "time": "Time CV (expanding, 4 folds)"}
N_FOLDS = {"random": 5, "group_uid": 5, "time": 4}
SEED = 0
REPORT_PATH = data.ROOT / "reports" / "leakage_check.md"
MANIFEST_PATH = data.ROOT / "results" / "step12_manifest.json"
MANIFEST_FILES = ["data/processed/cleaned.parquet", "config/features.yaml",
                  "config/best_params.yaml", "src/data.py", "src/features.py", "src/split.py",
                  "src/metrics.py", "src/experiment.py", "src/leakage.py",
                  "src/models/logreg.py", "src/models/lgbm.py"]


def best_configs(path=BEST_PATH):
    models = yaml.safe_load(path.read_text())["models"]
    return {name: models[name][FEATURE_SET]["params"] for name in MODELS}


def _inputs(params_json):
    """Recorded params without outcomes and fold number."""
    return {k: v for k, v in json.loads(params_json).items() if k not in ("fold", "_fit", "threshold")}


def scheme_rows(log, model_name, scheme, params):
    """Log rows for one model and scheme on F2 cv with exactly these params (fold numbers kept)."""
    rows = log[(log.model == model_name) & (log.feature_set == FEATURE_SET) & (log.split == "cv")
               & (log.cv_scheme == scheme) & (log.seed == SEED)]
    return rows[np.array([_inputs(p) == params for p in rows["params"]], dtype=bool)]


def run_folds(model, make_pipeline, df, folds, params, scheme, log_path=LOG_PATH):
    """Fit and score each fold, logging one cv row per fold; skips folds already logged."""
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=["model"])
    done = set()
    if len(log):
        done = {json.loads(p)["fold"] for p in scheme_rows(log, model.NAME, scheme, params)["params"]}
    for k, (a, b) in enumerate(folds, start=1):
        if k in done:
            print(f"{model.NAME} {scheme} fold {k}: already logged", flush=True)
            continue
        pipe = make_pipeline().fit(df.iloc[a])
        Xa, ya = experiment._xy(pipe, df, a)
        Xb, yb = experiment._xy(pipe, df, b)
        fitted, score, runtime = experiment._fit_score(model, Xa, ya, Xb, yb, params, SEED)
        metrics = evaluate(yb, score)
        logged = {**params, "fold": k}
        if hasattr(fitted, "run_metadata"):
            logged["_fit"] = fitted.run_metadata
        log_run(log_path, model=model.NAME, feature_set=FEATURE_SET, cv_scheme=scheme, split="cv",
                seed=SEED, params=logged, runtime_s=round(runtime, 2), **metrics)
        print(f"{model.NAME} {scheme} fold {k}: pr_auc={metrics['pr_auc']:.4f} ({runtime:.0f}s)",
              flush=True)
        del Xa, Xb, pipe, fitted
        gc.collect()


def main():
    check_manifest(MANIFEST_PATH, MANIFEST_FILES, MODELS)
    best = best_configs()
    raw = data.load_cleaned()
    train_idx, val_idx, _ = holdout_indices(raw)
    # Features need earlier rows only; the test block is dropped before they are built.
    raw = raw.iloc[:val_idx[-1] + 1]
    _, v_drop = data.load_v_columns()
    df = build_feature_set(raw, FEATURE_SET)
    y = df["isFraud"].to_numpy()[train_idx]
    uid = make_uid(df).to_numpy()[train_idx]
    folds = {"random": random_cv_folds(train_idx, y, k=5, seed=SEED),
             "group_uid": group_cv_folds(train_idx, y, uid, k=5, seed=SEED)}
    pipelines = {"logreg": lambda: data.LinearPipeline(v_drop=v_drop),
                 "lgbm": lambda: data.TreePipeline("lgbm", v_drop=v_drop)}
    for name, model in MODELS.items():
        for scheme, scheme_folds in folds.items():
            print(f"\n{name}/{FEATURE_SET} {scheme}: {best[name]}", flush=True)
            run_folds(model, pipelines[name], df, scheme_folds, best[name], scheme)


# ---------------------------------------------------------------------------
# Report: built from the log only.

def leakage_table(log, best):
    """Mean, sample std and fold count of PR-AUC per (model, scheme); raises if a cell is incomplete."""
    rows, problems = [], []
    for name in MODELS:
        for scheme in SCHEMES:
            r = scheme_rows(log, name, scheme, best[name])
            r = r.assign(fold=r["params"].map(lambda p: json.loads(p)["fold"])).sort_values("fold")
            if r["fold"].tolist() != list(range(1, N_FOLDS[scheme] + 1)):
                problems.append(f"{name}/{scheme}: folds {r['fold'].tolist()}")
            rows.append({"model": name, "scheme": scheme, "mean": r["pr_auc"].mean(),
                         "std": r["pr_auc"].std(), "n": len(r), "folds": r["pr_auc"].tolist()})
    if problems:
        raise ValueError("Each cell needs exactly one row per fold: " + "; ".join(problems))
    return pd.DataFrame(rows)


def render(log, best):
    t = leakage_table(log, best)
    cell = t.set_index(["model", "scheme"])
    lines = [
        "# Leakage check (Step 12)",
        "",
        "Generated by `.venv/bin/python -m src.leakage --report` from `results/log.csv`. "
        f"Models use F2 and their Step 9 best configs ({', '.join(f'{NAMES[m]} {best[m]}' for m in MODELS)}). "
        "All folds are inside the train block (first 472,432 rows); the validation and test blocks "
        "are not used. Each fold refits the preprocessing pipeline on its training part, and "
        "LightGBM early-stops on the fold's validation part. Random and group folds use seed 0; "
        "time-CV rows are the Step 9 runs of the same configs.",
        "",
        "Results are provisional until the group approves the V-column list.",
        "",
        "## PR-AUC by CV scheme (mean ± sample std over folds)",
        "",
        "| Model | " + " | ".join(SCHEMES.values()) + " |",
        "| --- | " + " | ".join("---" for _ in SCHEMES) + " |",
    ]
    for name in MODELS:
        cells = [f"{cell.loc[(name, s), 'mean']:.4f} ± {cell.loc[(name, s), 'std']:.4f}" for s in SCHEMES]
        lines.append(f"| {NAMES[name]} | " + " | ".join(cells) + " |")
    lines += ["", "Per-fold PR-AUC:", ""]
    for name in MODELS:
        for s in SCHEMES:
            folds = ", ".join(f"{v:.4f}" for v in cell.loc[(name, s), "folds"])
            lines.append(f"- {NAMES[name]}, {s}: {folds}")
    lines += ["", "## Which scheme scores highest (computed)", ""]
    for name in MODELS:
        means = {s: cell.loc[(name, s), "mean"] for s in SCHEMES}
        top = max(means, key=means.get)
        others = ", ".join(f"{means[top] - means[s]:+.4f} over {s} ({100 * (means[top] - means[s]) / means[s]:+.1f}%)"
                           for s in SCHEMES if s != top)
        lines.append(f"- {NAMES[name]}: **{top}** scores highest ({means[top]:.4f}); {others}.")
    lines += ["",
              "Random folds put rows of the same customer, and rows from later months, on both "
              "sides of a split. Group folds keep each uid on one side but still mix time. Time CV "
              "trains only on earlier months, which is how the model is used. The schemes also differ "
              "in training size (random and group folds train on 80% of the block; time folds on "
              "28-88%), which by itself favours the 5-fold schemes.",
              ""]
    return "\n".join(lines)


def write_report(log_path=LOG_PATH, report_path=REPORT_PATH, best_path=BEST_PATH):
    text = render(pd.read_csv(log_path), best_configs(best_path))
    report_path.write_text(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="only rebuild the report from the log")
    args = parser.parse_args()
    if not args.report:
        main()
    print(write_report())
