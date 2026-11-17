import json

import numpy as np
import pandas as pd
import pytest

from src import v_threshold_linear as vt
from src.data_profile import V_COLS


def _log(rows):
    return pd.DataFrame([{"model": m, "feature_set": vt.label(t), "cv_scheme": "time", "split": "cv",
                          "seed": seed, "pr_auc": pr, "auroc": 0.8,
                          "params": json.dumps({"fold": fold, "v_threshold": t, "n_v_kept": kept})}
                         for m, t, kept, fold, seed, pr in rows])


def test_table_pairs_on_fold_and_seed():
    rows = []
    for fold, base in ((1, 0.40), (2, 0.50)):
        rows.append(("logreg", 0.90, 229, fold, 0, base))
        rows.append(("logreg", 0.75, 140, fold, 0, base + (0.02 if fold == 1 else -0.01)))
    # an unrelated F2 row and another model must be ignored
    log = pd.concat([_log(rows), _log([("mlp", 0.90, 229, 1, 0, 0.9)]).assign(feature_set="F2")])
    table = vt.threshold_table(log, "logreg", n_folds=2).set_index("v_threshold")
    assert table.loc[0.75, "n_v_kept"] == 140 and table.loc[0.75, "runs"] == 2
    assert table.loc[0.75, "pr_auc_mean"] == pytest.approx(0.455)
    assert table.loc[0.75, "delta"] == pytest.approx(0.005)
    assert table.loc[0.75, "delta_se"] == pytest.approx(0.015)
    assert table.loc[0.75, "pairs_better"] == "1/2"
    assert np.isnan(table.loc[0.90, "delta"])


def test_table_rejects_incomplete_or_repeated_runs():
    full = [("logreg", t, 200, fold, 0, 0.4) for t in (0.75, 0.90) for fold in (1, 2)]
    with pytest.raises(ValueError, match="expected 2 runs"):
        vt.threshold_table(_log(full[:-1]), "logreg", n_folds=2)
    with pytest.raises(ValueError, match="expected 2 runs"):
        vt.threshold_table(_log(full + full[:1]), "logreg", n_folds=2)


def test_v_lists_cover_every_column_and_prune_more_at_lower_thresholds():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(500, 3))
    data = {c: base[:, i % 3] + rng.normal(scale=0.1 + 0.3 * (i % 5), size=500)
            for i, c in enumerate(V_COLS)}
    train = pd.DataFrame(data).astype("float32")
    lists = vt.v_lists(train, (0.5, 0.9))
    for keep, drop in lists.values():
        assert sorted(keep + drop) == sorted(V_COLS) and not set(keep) & set(drop)
    assert len(lists[0.5][0]) < len(lists[0.9][0])
