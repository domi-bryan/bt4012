"""Append one row per training run to results/log.csv (Step 4)."""
import csv
import json
import subprocess
from datetime import date
from pathlib import Path

from src.data import ROOT

LOG_PATH = ROOT / "results" / "log.csv"
COLUMNS = ["date", "author", "model", "feature_set", "cv_scheme", "split", "seed", "params",
           "pr_auc", "auroc", "precision", "recall", "recall_top1pct", "runtime_s"]
REQUIRED = ["model", "feature_set", "cv_scheme", "split", "pr_auc"]
SPLITS = {"cv", "val", "test"}


def _git_author():
    try:
        out = subprocess.run(["git", "config", "user.name"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def log_run(path=LOG_PATH, **fields):
    """Append one row with the Context.md columns, writing the header if the file is new.

    `params` may be a dict (stored as sorted JSON). Put the CV fold number in params
    (e.g. {"C": 0.1, "fold": 2}), since the log has no fold column. `date` defaults to
    today and `author` to `git config user.name`. Metric fields left out are written empty,
    so `log_run(**evaluate(...), model=...)` works directly.
    """
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f"Unknown log fields: {sorted(unknown)}")
    missing = [c for c in REQUIRED if fields.get(c) is None]
    if missing:
        raise ValueError(f"Missing required log fields: {missing}")
    if fields["split"] not in SPLITS:
        raise ValueError(f"split must be one of {sorted(SPLITS)}, got {fields['split']!r}")

    row = {"date": date.today().isoformat(), "author": _git_author(), **fields}
    if isinstance(row.get("params"), dict):
        row["params"] = json.dumps(row["params"], sort_keys=True)
    row = {c: "" if row.get(c) is None else row[c] for c in COLUMNS}

    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with path.open(newline="") as handle:
            header = next(csv.reader(handle), [])
        if header != COLUMNS:
            raise ValueError(f"{path} has an unexpected header: {header}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")   # the log is LF throughout
        if new:
            writer.writeheader()
        writer.writerow(row)
    return path
