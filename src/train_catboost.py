"""Resumable CatBoost time-CV tuning and one-parameter validation sensitivity runs.

Uses the shared feature builders, chronological indices, experiment functions and
results ledger. Run a single writer at a time. No command scores the test block.
"""
import argparse
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd
import yaml

from src import data, experiment
from src.catboost_pipeline import CatBoostPipeline
from src.features import build_feature_set
from src.models import catboost
from src.split import holdout_indices

BEST_PATH = data.ROOT / "config" / "catboost_best_params.yaml"
MANIFEST_PATH = data.ROOT / "results" / "catboost_manifest.json"
SEEDS = (0, 1, 2)
SOURCE_FILES = (
    "config/features.yaml", "src/data.py", "src/features.py", "src/split.py",
    "src/metrics.py", "src/results.py", "src/experiment.py",
    "src/catboost_pipeline.py", "src/models/catboost.py", "src/train_catboost.py",
)


def _write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


def check_manifest(data_path=data.CLEANED_PATH, log_path=experiment.LOG_PATH,
                   best_path=BEST_PATH, manifest_path=MANIFEST_PATH):
    """Refuse stale resume when data, source, dependencies or search policy changes."""
    files = {name: data.ROOT / name for name in SOURCE_FILES}
    files["processed_data"] = Path(data_path)
    hashes = {}
    for name, path in files.items():
        with path.open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {
        "sha256": hashes,
        "versions": {p: importlib.metadata.version(p) for p in
                     ("numpy", "pandas", "scikit-learn", "catboost", "torch")},
        "log_path": str(Path(log_path).resolve()),
        "best_path": str(Path(best_path).resolve()),
        "grid": catboost.GRID, "defaults": catboost.DEFAULTS,
        "sensitivity": catboost.SENSITIVITY, "search_seed": 0,
        "validation_seeds": list(SEEDS), "categorical_mode": "one_hot_no_target_statistics",
    }
    manifest_path = Path(manifest_path)
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("CatBoost inputs changed; use a new log, best-params file and manifest")
        return
    if Path(log_path).exists() and (pd.read_csv(log_path).model == catboost.NAME).any():
        raise ValueError("Existing CatBoost log has no matching manifest; use fresh output paths")
    if Path(best_path).exists():
        raise ValueError("Existing CatBoost best parameters have no manifest; use fresh output paths")
    _write(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run_cell(df, train_idx, val_idx, feature_set, make_pipeline, grid=None,
             log_path=experiment.LOG_PATH, best_path=BEST_PATH):
    """Select by mean training-block CV AP; refit on train for seeds 0, 1, 2."""
    grid = catboost.GRID if grid is None else grid
    cv = experiment.time_cv(catboost, make_pipeline, df, train_idx, grid,
                            feature_set, seed=0, log_path=log_path, resume=True)
    if cv.empty:
        raise ValueError("No time-CV folds; use training data spanning multiple 30-day bins")
    best = experiment.best_params(cv)
    path = Path(best_path)
    config = (yaml.safe_load(path.read_text()) or {}) if path.exists() else {}
    config.setdefault("models", {}).setdefault(catboost.NAME, {})[feature_set] = {
        "params": best, "defaults": catboost.DEFAULTS,
        "cv_scheme": "time", "search_seed": 0, "validation_seeds": list(SEEDS),
        "categorical_mode": "one_hot_no_target_statistics",
    }
    config["feature_config"] = "config/features.yaml (same approval status as shared implementation)"
    _write(path, yaml.safe_dump(config, sort_keys=False))
    for seed in SEEDS:
        if experiment.logged_run(catboost.NAME, feature_set, "val", seed, best, log_path):
            continue
        result = experiment.fit_validate(catboost, make_pipeline, df, train_idx, val_idx,
                                         best, feature_set, seed=seed, log_path=log_path)
        print(f"{feature_set} seed {seed}: validation AP={result['pr_auc']:.6f}", flush=True)
        del result
        gc.collect()
    return best


def load_best(feature_set, path=BEST_PATH):
    path = Path(path)
    if not path.exists():
        raise ValueError("Run CatBoost tuning before sensitivity analysis")
    return yaml.safe_load(path.read_text())["models"][catboost.NAME][feature_set]["params"]


def sensitivity_cases(baseline, sweeps=None):
    """Keep all tuned parameters fixed except the named parameter; tag log provenance."""
    for parameter, values in (catboost.SENSITIVITY if sweeps is None else sweeps).items():
        for value in values:
            yield {
                **baseline, parameter: value,
                "_sweep": {"parameter": parameter, "baseline": baseline},
            }


def run_sensitivity(df, train_idx, val_idx, feature_set, make_pipeline, baseline,
                    log_path=experiment.LOG_PATH, sweeps=None):
    """Validation-only RQ4 sweeps; refit preprocessing on training rows each time."""
    for params in sensitivity_cases(baseline, sweeps):
        for seed in SEEDS:
            if experiment.logged_run(catboost.NAME, feature_set, "val", seed, params, log_path):
                continue
            result = experiment.fit_validate(catboost, make_pipeline, df, train_idx, val_idx,
                                             params, feature_set, seed=seed, log_path=log_path)
            print(f"{feature_set} {params['_sweep']['parameter']} seed {seed}: "
                  f"validation AP={result['pr_auc']:.6f}", flush=True)
            del result
            gc.collect()


def sensitivity_summary(log_path, feature_set, baseline, sweeps=None):
    """Read exact sweep runs from the shared schema; reject missing/duplicated seeds."""
    rows = []
    for params in sensitivity_cases(baseline, sweeps):
        metrics = []
        for seed in SEEDS:
            row = experiment.logged_run(catboost.NAME, feature_set, "val", seed, params, log_path)
            if row is None:
                raise ValueError(f"Missing sensitivity run: {feature_set}/{params}/seed {seed}")
            if not pd.notna(row["pr_auc"]):
                raise ValueError("Sensitivity AP must be present")
            metrics.append(float(row["pr_auc"]))
        parameter = params["_sweep"]["parameter"]
        rows.append({"parameter": parameter, "value": params[parameter],
                     "pr_auc_mean": float(pd.Series(metrics).mean()),
                     "pr_auc_std": float(pd.Series(metrics).std(ddof=1)),
                     "seeds": len(metrics),
                     "tuned_baseline": params[parameter] == baseline.get(
                         parameter, catboost.DEFAULTS[parameter])})
    return pd.DataFrame(rows)


def write_sensitivity_report(log_path=experiment.LOG_PATH, feature_set="F2",
                             best_path=BEST_PATH, report_dir=None, figure_dir=None):
    """Write CSV, Markdown and one plot per parameter, all reconstructed from log rows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    baseline = load_best(feature_set, best_path)
    table = sensitivity_summary(log_path, feature_set, baseline)
    report_dir = Path(report_dir or data.ROOT / "reports")
    figure_dir = Path(figure_dir or data.ROOT / "reports" / "figures")
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(report_dir / f"catboost_sensitivity_{feature_set}.csv", index=False)
    lines = [f"# CatBoost {feature_set} sensitivity", "",
             "Validation average precision; mean and sample standard deviation over seeds 0, 1, 2. "
             "Each point varies one parameter from the time-CV-selected baseline. "
             "These scores use validation early stopping and are not final-test estimates.", "",
             "| Parameter | Value | Mean AP | Std AP | Tuned baseline |",
             "| --- | --- | ---: | ---: | --- |"]
    for parameter, group in table.groupby("parameter", sort=False):
        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(group))
        ax.errorbar(x, group.pr_auc_mean, yerr=group.pr_auc_std, marker="o", capsize=3)
        for i, row in enumerate(group.to_dict("records")):
            if row["tuned_baseline"]:
                ax.scatter(i, row["pr_auc_mean"], marker="*", s=150, color="tab:red",
                           label="Time-CV-selected baseline", zorder=3)
            lines.append(f"| {parameter} | {row['value']} | {row['pr_auc_mean']:.6f} | "
                         f"{row['pr_auc_std']:.6f} | {'yes' if row['tuned_baseline'] else ''} |")
        ax.set_xticks(list(x), group.value.astype(str))
        ax.set(xlabel=parameter, ylabel="Validation average precision", title=f"CatBoost {feature_set}")
        if group.tuned_baseline.any():
            ax.legend()
        fig.tight_layout()
        fig.savefig(figure_dir / f"sweep_catboost_{feature_set}_{parameter}.png", dpi=150)
        plt.close(fig)
    path = report_dir / f"catboost_sensitivity_{feature_set}.md"
    _write(path, "\n".join(lines) + "\n")
    return table


def main(mode="tune", feature_sets=None, data_path=data.CLEANED_PATH,
         log_path=experiment.LOG_PATH, best_path=BEST_PATH, manifest_path=MANIFEST_PATH):
    feature_sets = feature_sets or (["F0", "F1", "F2"] if mode == "tune" else ["F2"])
    if mode == "report":
        for feature_set in feature_sets:
            write_sensitivity_report(log_path, feature_set, best_path)
        return
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Missing {data_path}. Build joined.parquet and cleaned.parquet "
                                "with python -m src.data and python -m src.clean_data, "
                                "or pass --data /path/to/cleaned.parquet.")
    check_manifest(data_path, log_path, best_path, manifest_path)
    raw = data.load_cleaned(data_path)
    train_idx, val_idx, _ = holdout_indices(raw)
    if not len(val_idx):
        raise ValueError("Need at least ten rows for the shared 80/10/10 split")
    raw = raw.iloc[:val_idx[-1] + 1]
    _, v_drop = data.load_v_columns()
    for feature_set in feature_sets:
        df = build_feature_set(raw, feature_set)
        make_pipeline = lambda: CatBoostPipeline(v_drop=v_drop)
        if mode == "tune":
            run_cell(df, train_idx, val_idx, feature_set, make_pipeline,
                     log_path=log_path, best_path=best_path)
        elif mode == "sensitivity":
            run_sensitivity(df, train_idx, val_idx, feature_set, make_pipeline,
                            load_best(feature_set, best_path), log_path=log_path)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        del df
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["tune", "sensitivity", "report"], default="tune")
    parser.add_argument("--feature-sets", nargs="+", choices=["F0", "F1", "F2"])
    parser.add_argument("--data", type=Path, default=data.CLEANED_PATH)
    parser.add_argument("--log", type=Path, default=experiment.LOG_PATH)
    parser.add_argument("--best-params", type=Path, default=BEST_PATH)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    main(args.mode, args.feature_sets, args.data, args.log, args.best_params, args.manifest)
