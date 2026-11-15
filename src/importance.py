"""Feature importance for the model notebooks.

- `intrinsic`: the model's own importance (gain, impurity, |coefficient|, ...), summed back to the raw input
  columns. Each model measures it differently, so it is only comparable within a model. The MLP has none.
- `permutation`: drop in validation PR-AUC when one input column's values are shuffled (all of the
  pipeline columns it produces, e.g. its one-hot and missing-indicator columns, with the same shuffle).
  The same measure for every model, so it is comparable across models. Correlated columns share credit:
  shuffling one leaves its near-copy in place, so a low value does not prove a column is useless.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def raw_column(name, raw_columns):
    """The raw input column a pipeline output column comes from (LinearPipeline adds suffixes)."""
    if name in raw_columns:
        return name
    for suffix in ("__missing", "__freq"):
        if name.endswith(suffix) and name[:-len(suffix)] in raw_columns:
            return name[:-len(suffix)]
    if "==" in name and name.split("==", 1)[0] in raw_columns:
        return name.split("==", 1)[0]
    raise ValueError(f"Can't map pipeline column {name!r} to a raw column")


def column_blocks(pipeline_columns, raw_columns):
    """{raw column: [pipeline columns it produces]}."""
    raw = set(raw_columns)
    blocks = {}
    for name in pipeline_columns:
        blocks.setdefault(raw_column(name, raw), []).append(name)
    return blocks


def intrinsic(model, fitted, pipeline_columns, raw_columns):
    """The model's own importance per raw column (summed over its pipeline columns), or None."""
    values = model.feature_importance(fitted, pipeline_columns)
    if values is None:
        return None
    raw = set(raw_columns)
    return values.groupby([raw_column(c, raw) for c in values.index]).sum().reindex(list(raw_columns)).fillna(0.0)


def permutation(fitted, X, y, raw_columns, n_rows=20_000, repeats=1, seed=0):
    """Mean (and sd over repeats) drop in PR-AUC when each raw column is shuffled, on up to `n_rows` rows of X."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    rows = np.sort(rng.choice(len(X), size=min(n_rows, len(X)), replace=False))
    X, y = X.iloc[rows].reset_index(drop=True), y[rows]
    base = average_precision_score(y, fitted.predict_proba(X))
    blocks = column_blocks(X.columns, raw_columns)
    out = []
    for raw in raw_columns:
        drops = []
        for _ in range(repeats):
            perm = rng.permutation(len(X))
            shuffled = X.copy(deep=False)
            for col in blocks.get(raw, []):
                shuffled[col] = X[col].iloc[perm].set_axis(X.index)   # keeps category dtypes
            drops.append(base - average_precision_score(y, fitted.predict_proba(shuffled)))
        out.append({"feature": raw, "pr_auc_drop": float(np.mean(drops)),
                    "drop_sd": float(np.std(drops, ddof=1)) if repeats > 1 else np.nan})
    return pd.DataFrame(out).set_index("feature").assign(base_pr_auc=base)


def feature_groups(spec):
    """{column: group} for G0-G3 from config/features_v2.yaml (the group that adds the column)."""
    return {c: g for g, cols in spec["feature_groups"].items() for c in cols}
