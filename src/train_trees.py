"""Step 9: resumable time-CV tuning and three-seed validation for LR, RF and LightGBM.

Run one process at a time: results/log.csv is an append-only experiment ledger.
No test rows are passed to a model or preprocessing pipeline.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import yaml

from src import data, experiment
from src.features import build_feature_set
from src.models import lgbm, logreg, rf
from src.split import holdout_indices

MODELS = {"logreg": logreg, "rf": rf, "lgbm": lgbm}
BEST_PATH = data.ROOT / "config" / "best_params.yaml"
MANIFEST_PATH = data.ROOT / "results" / "step9_manifest.json"


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content)
    temporary.replace(path)


MANIFEST_FILES = ["data/processed/cleaned.parquet", "config/features.yaml",
                  "src/data.py", "src/features.py", "src/split.py", "src/metrics.py",
                  "src/experiment.py", "src/models/logreg.py", "src/models/rf.py",
                  "src/models/lgbm.py"]


def check_manifest(path=MANIFEST_PATH, files=MANIFEST_FILES, models=MODELS,
                   packages=("numpy", "pandas", "scikit-learn", "lightgbm")):
    """Refuse to mix resumed runs with changed inputs or model implementations."""
    hashes = {}
    for name in files:
        with (data.ROOT / name).open("rb") as handle:
            hashes[name] = hashlib.file_digest(handle, "sha256").hexdigest()
    import importlib.metadata
    versions = {name: importlib.metadata.version(name) for name in packages}
    manifest = {"sha256": hashes, "versions": versions,
                "search_seed": 0, "validation_seeds": [0, 1, 2],
                "grids": {name: module.GRID for name, module in models.items()}}
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError(f"{path.name}: inputs changed since it was saved; use a separate "
                         "experiment log and manifest rather than mixing results.")
    _write(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def run_cell(model, make_pipeline, df, train_idx, val_idx, feature_set, grid,
             log_path=experiment.LOG_PATH, best_path=BEST_PATH):
    """Resume exact CV folds and validation seeds, and save the chosen search parameters."""
    print(f"\n{model.NAME}/{feature_set}: {len(grid)} configs, time CV seed 0", flush=True)
    cv = experiment.time_cv(model, make_pipeline, df, train_idx, grid, feature_set,
                            seed=0, log_path=log_path, resume=True)
    best = experiment.best_params(cv)
    config = yaml.safe_load(best_path.read_text()) if best_path.exists() else {}
    config = config or {}
    config.setdefault("models", {}).setdefault(model.NAME, {})[feature_set] = {
        "params": best, "defaults": model.DEFAULTS,
        "cv_scheme": "time", "search_seed": 0, "validation_seeds": [0, 1, 2],
    }
    config["feature_config"] = "config/features.yaml (provisional until group approval)"
    config["lightgbm_early_stopping"] = {
        "metric": "sklearn average_precision_score", "patience": 50,
        "max_rounds": lgbm.DEFAULTS["n_estimators"],
        "weight_rule": "'balanced' resolves to negative/positive labels in each training part",
        "note": "Per-fit best iteration and resolved weight are in log params._fit.",
    }
    _write(best_path, yaml.safe_dump(config, sort_keys=False))
    print(f"Selected {best}", flush=True)
    for seed in (0, 1, 2):
        if experiment.logged_run(model.NAME, feature_set, "val", seed, best, log_path):
            print(f"seed {seed}: validation already logged", flush=True)
            continue
        val = experiment.fit_validate(model, make_pipeline, df, train_idx, val_idx,
                                      best, feature_set, seed=seed, log_path=log_path)
        print(f"seed {seed}: val pr_auc={val['pr_auc']:.6f}, "
              f"auroc={val['auroc']:.6f} ({val['runtime_s']:.0f}s)", flush=True)
        del val
        gc.collect()
    return best


def main(models, feature_sets):
    check_manifest()
    raw = data.load_cleaned()
    train_idx, val_idx, _ = holdout_indices(raw)
    # Feature history needs earlier rows only. Exclude the entire test block before building.
    raw = raw.iloc[:val_idx[-1] + 1]
    _, v_drop = data.load_v_columns()
    for feature_set in feature_sets:
        df = build_feature_set(raw, feature_set)
        for name in models:
            model = MODELS[name]
            if name == "logreg":
                make_pipeline = lambda: data.LinearPipeline(v_drop=v_drop)
            else:
                make_pipeline = lambda kind=name: data.TreePipeline(kind=kind, v_drop=v_drop)
            run_cell(model, make_pipeline, df, train_idx, val_idx, feature_set, model.GRID)
            gc.collect()
        del df
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--feature-sets", nargs="+", choices=["F0", "F1", "F2"],
                        default=["F0", "F1", "F2"])
    args = parser.parse_args()
    main(args.models, args.feature_sets)
