"""Step 10: resumable time-CV tuning and three-seed validation for the MLP on F0, F1, F2.

Same procedure as Step 9 (`src.train_trees.run_cell`): time CV with seed 0, best config by
mean CV PR-AUC, refit on the train block and score validation for seeds 0, 1, 2.
Run one training process at a time: results/log.csv is append-only.
"""
import argparse
import gc

import torch
import yaml

from src import data
from src.features import build_feature_set
from src.models import mlp
from src.split import holdout_indices
from src.train_trees import BEST_PATH, _write, check_manifest, run_cell

MANIFEST_PATH = data.ROOT / "results" / "step10_manifest.json"
MANIFEST_FILES = ["data/processed/cleaned.parquet", "config/features.yaml",
                  "src/data.py", "src/features.py", "src/split.py", "src/metrics.py",
                  "src/experiment.py", "src/models/mlp.py"]
THREADS = 4


def main(feature_sets):
    check_manifest(MANIFEST_PATH, MANIFEST_FILES, {"mlp": mlp},
                   ("numpy", "pandas", "scikit-learn", "torch"))
    torch.set_num_threads(THREADS)
    raw = data.load_cleaned()
    train_idx, val_idx, _ = holdout_indices(raw)
    # Feature history needs earlier rows only. Exclude the entire test block before building.
    raw = raw.iloc[:val_idx[-1] + 1]
    _, v_drop = data.load_v_columns()
    make_pipeline = lambda: data.LinearPipeline(v_drop=v_drop)
    for feature_set in feature_sets:
        df = build_feature_set(raw, feature_set)
        run_cell(mlp, make_pipeline, df, train_idx, val_idx, feature_set, mlp.GRID)
        config = yaml.safe_load(BEST_PATH.read_text())
        config["mlp_training"] = {
            "layers": "Linear(n, hidden) ReLU Dropout, Linear(hidden, hidden//2) ReLU Dropout, Linear(., 1)",
            "loss": "BCEWithLogitsLoss(pos_weight = negatives / positives of the training part)",
            "optimizer": "Adam", "batch_size": mlp.DEFAULTS["batch_size"],
            "early_stopping": f"validation PR-AUC, patience {mlp.DEFAULTS['patience']}, "
                              f"max {mlp.DEFAULTS['max_epochs']} epochs, best epoch restored",
            "device": f"cpu, {THREADS} threads",
            "note": "Per-fit loss and validation PR-AUC by epoch are in log params._fit.",
        }
        _write(BEST_PATH, yaml.safe_dump(config, sort_keys=False))
        del df
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-sets", nargs="+", choices=["F0", "F1", "F2"],
                        default=["F0", "F1", "F2"])
    main(parser.parse_args().feature_sets)
