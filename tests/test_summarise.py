import pandas as pd
import pytest

from src import summarise
from src.results import COLUMNS


def fake_log():
    rows = []
    for i, model in enumerate(summarise.MODELS):
        for j, fs in enumerate(summarise.FEATURE_SETS):
            for seed in summarise.SEEDS:
                score = 0.1 * (i + 1) + 0.01 * j + 0.001 * seed
                rows.append(dict(date="2026-10-21", author="t", model=model, feature_set=fs,
                                 cv_scheme="holdout", split="val", seed=seed, params="{}",
                                 pr_auc=score, auroc=0.5 + score, precision=0.1, recall=0.1,
                                 recall_top1pct=score / 2, runtime_s=1.0))
    rows.append({**rows[0], "split": "cv", "pr_auc": 99.0})   # cv rows are ignored
    return pd.DataFrame(rows, columns=COLUMNS)


def test_grid_values_and_gains():
    grid = summarise.validation_grid(fake_log())
    mean, std, gain_model, gain_features = summarise.metric_table(grid, "pr_auc")
    # logreg is model 0 -> 0.1 + 0.01*j + mean(0, 0.001, 0.002)
    assert mean.loc["logreg", "F1"] == pytest.approx(0.111)
    assert std.loc["logreg", "F1"] == pytest.approx(0.001)
    assert gain_model["F0"] == pytest.approx(mean.loc["lgbm", "F0"] - mean.loc["logreg", "F0"])
    assert gain_model["F0"] == pytest.approx(0.2)
    assert gain_features.tolist() == pytest.approx([0.02] * 4)
    assert list(mean.index) == list(summarise.MODELS) and list(mean.columns) == summarise.FEATURE_SETS


def test_render_has_every_cell():
    text = summarise.render(fake_log())
    for name in summarise.MODELS.values():
        assert text.count(f"| {name} |") == 3            # one row per metric table
    assert text.count("| **LR → LightGBM** |") == 3
    assert "99.0" not in text


@pytest.mark.parametrize("change", ["drop", "duplicate"])
def test_incomplete_or_duplicated_cells_rejected(change):
    log = fake_log()
    if change == "drop":
        log = log.drop(index=4)
    else:
        log = pd.concat([log, log.iloc[[4]]])
    with pytest.raises(ValueError, match="seeds 0, 1, 2"):
        summarise.validation_grid(log)
