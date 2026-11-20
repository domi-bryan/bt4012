"""Final test scoring (Step 19). Run only after the experiment freeze.

For one model, each of the 12 cells (G0-G3 x seeds 0, 1, 2) is refitted on the train block exactly as in
its validation run (`experiment.fit_final`: same settings, seed and rows, and the same two-step early
stopping), then the test block is scored with the threshold stored in that cell's validation row. Nothing
is tuned or re-thresholded here: every setting comes from the validation rows of `results/log.csv`.

Before scoring the test block, each refit rescores the validation block and must reproduce the logged
validation PR-AUC, so the test score belongs to the model that was validated.

Each cell logs one row with split = "test" and saves its test scores to
`results/preds/<model>_<set>_test_seed<seed>.parquet`. Cells already logged are skipped, so a run can be
stopped and restarted. One model per process (macOS: the model libraries' OpenMP runtimes clash):

    .venv/bin/python -u -m src.final_test --model catboost --freeze-confirmed
    .venv/bin/python -u -m src.final_test --blend --freeze-confirmed   # after the four G3 input models
    .venv/bin/python -m src.final_test --report        # once all test rows are merged: reports/test_results.md

The segment blend (`src.ensemble`, fixed before the freeze) needs no training: `--blend` combines the saved G3
test scores of XGBoost, LightGBM, CatBoost and random forest and applies the threshold from its validation row.
"""
import argparse
import gc
import importlib
import json
import time

import numpy as np
import pandas as pd

from src import experiment, feature_sets
from src.data import ROOT
from src.metrics import evaluate
from src.results import LOG_PATH, log_run

MODULES = {"logreg": "src.models.logreg", "rf": "src.models.rf", "lgbm": "src.models.lgbm",
           "xgboost": "src.models.xgb", "catboost": "src.models.catboost", "mlp": "src.models.mlp"}
NAMES = {"logreg": "Logistic regression", "rf": "Random forest", "lgbm": "LightGBM",
         "xgboost": "XGBoost", "catboost": "CatBoost", "mlp": "MLP", "segment_blend": "Segment blend (G3)"}
TABLE_MODELS = list(MODULES) + ["segment_blend"]      # the blend has G3 only
EXPECTED_TEST_ROWS = len(MODULES) * 4 * 3 + 3       # 6 models x 4 sets x 3 seeds, plus the blend's 3
FEATURE_SETS = feature_sets.FEATURE_SETS
SEEDS = experiment.VAL_SEEDS
REPRO_TOL = 1e-6          # refit must reproduce the logged validation PR-AUC to within this
REPORT_PATH = ROOT / "reports" / "test_results.md"
FREEZE_MESSAGE = ("The test block is scored only after the group confirms the experiment freeze "
                  "(CLAUDE.md, BUILD_STEPS.md Step 19). Pass freeze_confirmed=True / --freeze-confirmed "
                  "once that has happened.")


def make_pipeline(model_name, columns):
    """The same input pipeline as the model's notebook (04-09)."""
    from src import data
    if model_name in ("logreg", "mlp"):
        return data.LinearPipeline(columns=columns)
    if model_name == "rf":
        return data.TreePipeline("rf", columns=columns)
    if model_name in ("lgbm", "xgboost"):
        return data.TreePipeline("lgbm", columns=columns)
    if model_name == "catboost":
        from src.catboost_pipeline import CatBoostPipeline
        return CatBoostPipeline(columns=columns)
    raise ValueError(f"Unknown model {model_name!r}; expected one of {sorted(MODULES)}")


def settings_of(params):
    """Model settings of a logged run: params without the fold, threshold, fit metadata or es_setup."""
    p = json.loads(params) if isinstance(params, str) else params
    return {k: v for k, v in p.items() if k not in experiment.NOT_SETTINGS and not k.startswith("_")}


def validation_rows(log, model_name, sets=FEATURE_SETS, seeds=SEEDS):
    """{(set, seed): validation row} under the current setup; raises unless each cell has exactly one."""
    log = experiment.current_runs(log)
    val = log[(log.model == model_name) & (log.split == "val") & (log.cv_scheme == "holdout")]
    rows = {}
    for g in sets:
        for s in seeds:
            r = val[(val.feature_set == g) & (val.seed == s)]
            if len(r) != 1:
                raise ValueError(f"{model_name}/{g} seed {s}: expected 1 current validation row, found {len(r)}")
            rows[(g, s)] = r.iloc[0]
    return rows


def logged_test_row(log, model_name, feature_set, seed):
    """The logged test row of a cell, or None."""
    r = log[(log.model == model_name) & (log.feature_set == feature_set) & (log.split == "test")
            & (log.seed == seed)]
    if len(r) > 1:
        raise ValueError(f"{model_name}/{feature_set} seed {seed}: {len(r)} test rows in the log")
    return None if r.empty else r.iloc[0]


def preds_path(preds_dir, model_name, feature_set, seed):
    return preds_dir / f"{model_name}_{feature_set}_test_seed{seed}.parquet"


def score_cell(model, df, train_idx, val_idx, test_idx, feature_set, seed, val_row, columns,
               log_path=LOG_PATH, preds_dir=experiment.PREDS_DIR, cache_dir=experiment.MODELS_DIR,
               es_fraction=experiment.ES_FRACTION):
    """Refit one cell as validated, check it reproduces validation, score and log the test block."""
    settings = settings_of(val_row["params"])
    threshold = float(json.loads(val_row["params"])["threshold"])
    build = lambda cols=columns: make_pipeline(model.NAME, cols)
    # reuse a cached refit only if one exists (fit_final checks its settings); don't write 72 new ones
    cached = cache_dir is not None and (cache_dir / f"{model.NAME}_{feature_set}_seed{seed}.pkl").exists()
    t0 = time.time()
    fitted, pipe = experiment.fit_final(model, build, df, train_idx, settings, feature_set, seed=seed,
                                        es_fraction=es_fraction, cache_dir=cache_dir if cached else None)
    runtime = time.time() - t0

    y_val = df["isFraud"].to_numpy()[val_idx]
    val_check = float(evaluate(y_val, fitted.predict_proba(pipe.transform(df.iloc[val_idx])))["pr_auc"])
    if abs(val_check - float(val_row["pr_auc"])) > REPRO_TOL:
        raise RuntimeError(f"{model.NAME}/{feature_set} seed {seed}: the refit scores validation PR-AUC "
                           f"{val_check:.6f}, but the logged validation run scored {val_row['pr_auc']:.6f}. "
                           "The test score would not belong to the validated model; nothing was logged.")

    y_test = df["isFraud"].to_numpy()[test_idx]
    score = fitted.predict_proba(pipe.transform(df.iloc[test_idx]))
    metrics = evaluate(y_test, score, threshold)
    params = {**experiment._run_params(model, settings, es_fraction), "threshold": threshold,
              "_test": {"val_pr_auc_refit": round(val_check, 6), "reused_cached_model": bool(cached)}}
    log_run(log_path, model=model.NAME, feature_set=feature_set, cv_scheme="holdout", split="test",
            seed=seed, params=params, runtime_s=round(runtime, 2), **metrics)
    experiment.save_predictions(preds_path(preds_dir, model.NAME, feature_set, seed), df, test_idx, score,
                                seed=seed, params=json.dumps(settings, sort_keys=True))
    return {**metrics, "val_check": val_check, "runtime_s": runtime}


def run(model_name, freeze_confirmed=False, sets=FEATURE_SETS, seeds=SEEDS, log_path=LOG_PATH,
        preds_dir=experiment.PREDS_DIR, cache_dir=experiment.MODELS_DIR, data=None):
    """Score every cell of one model on the test block (skipping cells already logged).

    `data` = (df, train_idx, val_idx, test_idx) for tests; by default the full features_v2 table.
    """
    if not freeze_confirmed:
        raise PermissionError(FREEZE_MESSAGE)
    model = importlib.import_module(MODULES[model_name])
    rows = validation_rows(pd.read_csv(log_path), model_name, sets, seeds)
    if data is None:
        df, train_idx, val_idx, test_idx = feature_sets.load_features_with_test()
        spec = feature_sets.load_spec()
        columns = {g: feature_sets.feature_list(g, spec) for g in sets}
    else:
        df, train_idx, val_idx, test_idx, columns = data
    for g in sets:
        for s in seeds:
            done = logged_test_row(pd.read_csv(log_path), model_name, g, s)
            if done is not None:
                note = "" if preds_path(preds_dir, model_name, g, s).exists() else \
                    " (test scores not saved; delete its log row to rerun)"
                print(f"{model_name}/{g} seed {s}: test already logged{note}", flush=True)
                continue
            out = score_cell(model, df, train_idx, val_idx, test_idx, g, s, rows[(g, s)], columns[g],
                             log_path=log_path, preds_dir=preds_dir, cache_dir=cache_dir)
            print(f"{model_name}/{g} seed {s}: test pr_auc={out['pr_auc']:.4f} auroc={out['auroc']:.4f} "
                  f"(validation reproduced: {out['val_check']:.4f}; {out['runtime_s']:.0f}s)", flush=True)
            gc.collect()


def run_blend(freeze_confirmed=False, seeds=SEEDS, log_path=LOG_PATH, preds_dir=experiment.PREDS_DIR, lookup=None):
    """Log the segment blend's test rows from the saved G3 test scores of its four models (no training).

    `lookup` = (uid by TransactionID, train-block uids) for tests; by default read from features_v2.
    """
    if not freeze_confirmed:
        raise PermissionError(FREEZE_MESSAGE)
    from src import ensemble
    uid_by_id, train_uids = lookup or ensemble.uid_lookup()
    return ensemble.log_blend("test", uid_by_id, train_uids, seeds=seeds, log_path=log_path, preds_dir=preds_dir)


def results_table(log):
    """Mean ± sample sd over seeds of test PR-AUC per model and set, and the validation-to-test change."""
    log = experiment.current_runs(log)
    t = log[(log.split == "test") & log.model.isin(TABLE_MODELS) & log.feature_set.isin(FEATURE_SETS)]
    v = log[(log.split == "val") & (log.cv_scheme == "holdout") & log.model.isin(TABLE_MODELS)
            & log.feature_set.isin(FEATURE_SETS)]
    test = t.groupby(["model", "feature_set"])["pr_auc"].agg(["mean", "std", "count"])
    val = v.groupby(["model", "feature_set"])["pr_auc"].mean().rename("val_mean")
    out = test.join(val, how="left")
    out["change"] = out["mean"] - out["val_mean"]
    return out


def _missing(m, g):
    """Text for a cell with no test rows: the blend is G3 only by design; anything else was not run."""
    return "n/a" if m == "segment_blend" and g != "G3" else "not run"


def render(log):
    t = results_table(log)
    order = [m for m in TABLE_MODELS if m in t.index.get_level_values(0)]
    lines = ["# Test results (Step 19)", "",
             "Generated by `.venv/bin/python -m src.final_test --report` from `results/log.csv`. Each cell is the "
             "chosen settings refitted on the train block exactly as in its validation run and scored once on "
             "the test block (59,054 rows) with the threshold stored at validation. Mean ± sample sd over seeds "
             "0, 1, 2.", "", "## Test PR-AUC", "",
             "| Model | " + " | ".join(FEATURE_SETS) + " |", "| --- |" + " --- |" * len(FEATURE_SETS)]
    for m in order:
        cells = []
        for g in FEATURE_SETS:
            if (m, g) in t.index:
                r = t.loc[(m, g)]
                cells.append(f"{r['mean']:.4f} ± {r['std']:.4f}" + ("" if r["count"] == 3 else f" ({int(r['count'])} seeds)"))
            else:
                cells.append(_missing(m, g))
        lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    lines += ["", "## Change from validation to test (mean PR-AUC)", "",
              "| Model | " + " | ".join(FEATURE_SETS) + " |", "| --- |" + " --- |" * len(FEATURE_SETS)]
    for m in order:
        cells = [f"{t.loc[(m, g), 'change']:+.4f}" if (m, g) in t.index else _missing(m, g) for g in FEATURE_SETS]
        lines.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    n = int(((log.split == "test")).sum())
    lines += ["", f"Test rows in the log: {n} (expected {EXPECTED_TEST_ROWS}: 6 models x 4 sets x 3 seeds, "
              "plus 3 for the segment blend on G3). The blend is built on G3 only, so its G0–G2 cells are n/a.",
              "", "## Known vs unknown customers", "",
              "Test PR-AUC for known customers (uid seen in the train block) and unknown ones, with 95% bootstrap "
              "intervals, is in `notebooks/12_test.ipynb` section 4. It is computed from the saved test scores in "
              "`results/preds/` (shared on the group drive), not from the log, so this report does not rebuild it."]
    return "\n".join(lines) + "\n"


def write_report(log_path=LOG_PATH, report_path=REPORT_PATH):
    report_path.write_text(render(pd.read_csv(log_path)))
    return report_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODULES))
    ap.add_argument("--freeze-confirmed", action="store_true",
                    help="confirm the group has agreed the experiment freeze")
    ap.add_argument("--report", action="store_true", help="only rebuild reports/test_results.md from the log")
    ap.add_argument("--blend", action="store_true", help="log the segment blend's test rows from saved scores")
    args = ap.parse_args()
    if args.report:
        print(f"wrote {write_report()}")
        return
    if args.blend:
        run_blend(freeze_confirmed=args.freeze_confirmed)
    elif args.model:
        run(args.model, freeze_confirmed=args.freeze_confirmed)
    else:
        ap.error("give --model, --blend or --report")
    # The report is written only by --report, once all test rows are merged: a report written after one
    # person's run would show only their models and conflict with everyone else's.
    print("done; commit only results/log.csv. Once all test rows are merged: python -m src.final_test --report")


if __name__ == "__main__":
    main()
