"""Role C: leakage check on G3 (Step 18).

LR and LightGBM on G3 with the settings chosen in Step 16 (notebooks 04 and 06), scored inside the
train block under three CV schemes. The time-CV rows are the Step 16 tuning rows of those settings and
are reused as they are; this module adds random stratified 5-fold (cv_scheme "random") and
StratifiedGroupKFold on uid (cv_scheme "group_uid"), seed 0.

Every fold is fitted exactly as a tuning fold: the pipeline is refit on the fold's training part, and
LightGBM trains on the first 90% (by time) of that part and early-stops on the last 10%, never on the
rows it is scored on. Only the way the scored rows are chosen differs between the schemes.

Run `.venv/bin/python -u -m src.leakage` (resumable), then `--report` to rebuild
reports/leakage_check.md from the log alone. The earlier check on F2 (Step 12) stays in the log.
"""
import argparse
import gc
import json

import numpy as np
import pandas as pd

from src import data, experiment, feature_sets
from src.metrics import evaluate
from src.models import lgbm, logreg
from src.results import LOG_PATH, log_run
from src.split import group_cv_folds, random_cv_folds

MODELS = {"logreg": logreg, "lgbm": lgbm}
NAMES = {"logreg": "LR", "lgbm": "LightGBM"}
PIPELINES = {"logreg": lambda cols: data.LinearPipeline(columns=cols),
             "lgbm": lambda cols: data.TreePipeline("lgbm", columns=cols)}
FEATURE_SET = "G3"
SCHEMES = {"random": "Random stratified 5-fold", "group_uid": "StratifiedGroupKFold on uid (5)",
           "time": "Time CV (expanding, 4 folds)"}
N_FOLDS = {"random": 5, "group_uid": 5, "time": 4}
SEED = 0
ES_FRACTION = experiment.ES_FRACTION
REPORT_PATH = data.ROOT / "reports" / "leakage_check.md"
NOT_INPUTS = ("fold", "_fit", "threshold")   # logged keys that are outcomes or the fold number


def _settings(params_json):
    """Model settings of a logged run, without fold, outcomes or the early-stopping setup."""
    return {k: v for k, v in json.loads(params_json).items() if k not in experiment.NOT_SETTINGS}


def chosen_settings(log):
    """{model: settings} that Step 16 validated on G3, read from the validation rows of the log.

    Rows from an older early-stopping setup are left out (`experiment.current_runs`). Raises unless
    each model has exactly one set of settings there.
    """
    log = experiment.current_runs(log)
    chosen = {}
    for name in MODELS:
        rows = log[(log.model == name) & (log.feature_set == FEATURE_SET) & (log.split == "val")]
        found = sorted({json.dumps(_settings(p), sort_keys=True) for p in rows["params"]})
        if len(found) != 1:
            raise ValueError(f"{name}/{FEATURE_SET}: expected one set of validated settings in the log, "
                             f"found {len(found)}: {found}")
        chosen[name] = json.loads(found[0])
    return chosen


def scheme_rows(log, model_name, scheme, params):
    """Log rows for one model and scheme on G3 cv with exactly these settings (fold numbers kept).

    LightGBM rows must also carry the current early-stopping setup, so runs from an older one never match.
    """
    wanted = experiment._run_params(MODELS[model_name], params, ES_FRACTION)
    rows = log[(log.model == model_name) & (log.feature_set == FEATURE_SET) & (log.split == "cv")
               & (log.cv_scheme == scheme) & (log.seed == SEED)]
    inputs = [{k: v for k, v in json.loads(p).items() if k not in NOT_INPUTS} for p in rows["params"]]
    return rows[np.array([i == wanted for i in inputs], dtype=bool)]


def run_folds(model, make_pipeline, df, folds, params, scheme, log_path=LOG_PATH, es_fraction=ES_FRACTION):
    """Fit and score each fold as a tuning fold, logging one cv row per fold; skips folds already logged."""
    log = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame(columns=["model"])
    done = set()
    if len(log):
        done = {json.loads(p)["fold"] for p in scheme_rows(log, model.NAME, scheme, params)["params"]}
    for k, (a, b) in enumerate(folds, start=1):
        if k in done:
            print(f"{model.NAME} {scheme} fold {k}: already logged", flush=True)
            continue
        # training rows in time order, so the early-stopping rows are the latest ones
        pipe, Xa, ya, Xe, ye, Xb, yb, n_es = experiment._prepare(model, make_pipeline, df, np.sort(a), b,
                                                                 es_fraction)
        fitted, score, runtime = experiment._fit_score(model, Xa, ya, Xe, ye, params, SEED, X_score=Xb)
        metrics = evaluate(yb, score)
        logged = {**experiment._run_params(model, params, es_fraction), "fold": k}
        meta = dict(getattr(fitted, "run_metadata", {}))
        if n_es:
            meta["early_stopping_rows"] = n_es
        if meta:
            logged["_fit"] = meta
        log_run(log_path, model=model.NAME, feature_set=FEATURE_SET, cv_scheme=scheme, split="cv",
                seed=SEED, params=logged, runtime_s=round(runtime, 2), **metrics)
        print(f"{model.NAME} {scheme} fold {k}: pr_auc={metrics['pr_auc']:.4f} ({runtime:.0f}s)",
              flush=True)
        del Xa, Xb, Xe, pipe, fitted
        gc.collect()


def check_group_folds(folds, uid):
    """Raise if any fold has a uid on both sides. `uid` is indexed by row position in the table."""
    for k, (a, b) in enumerate(folds, start=1):
        shared = set(uid[a]) & set(uid[b])
        if shared:
            raise ValueError(f"group fold {k}: {len(shared)} uids appear in both training and scored rows")


def make_folds(df, train_idx):
    """{scheme: [(train rows, scored rows)]} for the two extra schemes, inside the train block."""
    y = df["isFraud"].to_numpy()[train_idx]
    uid = df["uid"].astype(str).to_numpy()
    folds = {"random": random_cv_folds(train_idx, y, k=N_FOLDS["random"], seed=SEED),
             "group_uid": group_cv_folds(train_idx, y, uid[train_idx], k=N_FOLDS["group_uid"], seed=SEED)}
    check_group_folds(folds["group_uid"], uid)
    return folds


def main(log_path=LOG_PATH):
    chosen = chosen_settings(pd.read_csv(log_path))
    # load_features drops the test block; the validation block is loaded but never used here.
    df, train_idx, _ = feature_sets.load_features()
    columns = feature_sets.feature_list(FEATURE_SET)
    folds = make_folds(df, train_idx)
    for name, model in MODELS.items():
        for scheme, scheme_folds in folds.items():
            print(f"\n{name}/{FEATURE_SET} {scheme}: {chosen[name]}", flush=True)
            run_folds(model, lambda name=name: PIPELINES[name](columns), df, scheme_folds, chosen[name],
                      scheme, log_path)


# ---------------------------------------------------------------------------
# Report: built from the log only.

def leakage_table(log, chosen):
    """Mean, sample std and fold count of PR-AUC per (model, scheme); raises if a cell is incomplete."""
    rows, problems = [], []
    for name in MODELS:
        for scheme in SCHEMES:
            r = scheme_rows(log, name, scheme, chosen[name])
            r = r.assign(fold=r["params"].map(lambda p: json.loads(p)["fold"])).sort_values("fold")
            if r["fold"].tolist() != list(range(1, N_FOLDS[scheme] + 1)):
                problems.append(f"{name}/{scheme}: folds {r['fold'].tolist()}")
            rows.append({"model": name, "scheme": scheme, "mean": r["pr_auc"].mean(),
                         "std": r["pr_auc"].std(), "n": len(r), "folds": r["pr_auc"].tolist()})
    if problems:
        raise ValueError("Each cell needs exactly one row per fold: " + "; ".join(problems))
    return pd.DataFrame(rows)


def render(log, chosen):
    t = leakage_table(log, chosen)
    cell = t.set_index(["model", "scheme"])
    lines = [
        "# Leakage check on G3 (Step 18)",
        "",
        "Generated by `.venv/bin/python -m src.leakage --report` from `results/log.csv`. "
        f"Models use G3 and the settings chosen in Step 16 ({', '.join(f'{NAMES[m]} {chosen[m]}' for m in MODELS)}). "
        "All folds are inside the train block (first 472,432 rows); the validation and test blocks "
        "are not used. Every fold is fitted the same way: the preprocessing pipeline is refit on the "
        "fold's training part, and LightGBM trains on the first 90% of that part by time and early-stops "
        "on the last 10%, never on the rows it is scored on. Random and group folds use seed 0; "
        "time-CV rows are the Step 16 tuning rows of the same settings.",
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


def write_report(log_path=LOG_PATH, report_path=REPORT_PATH):
    log = pd.read_csv(log_path)
    text = render(log, chosen_settings(log))
    report_path.write_text(text)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="only rebuild the report from the log")
    args = parser.parse_args()
    if not args.report:
        main()
    print(write_report())
