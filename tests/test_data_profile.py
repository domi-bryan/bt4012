import pytest
import yaml

from src import data_profile as dp
from src.data import JOINED_PATH, load_joined

pytestmark = pytest.mark.skipif(not JOINED_PATH.exists(), reason="joined.parquet not built")


def test_report_covers_every_column():
    cols = list(load_joined().columns)
    text = dp.REPORT_PATH.read_text()
    missing = [c for c in cols if f"| {c} |" not in text]
    assert missing == []


def test_every_v_column_in_exactly_one_list():
    doc = yaml.safe_load(dp.FEATURES_YAML.read_text())["v_columns_proposed"]
    keep, drop = doc["keep"], doc["drop"]
    assert not set(keep) & set(drop)
    assert sorted(keep + drop) == sorted(dp.V_COLS)
    assert len(keep + drop) == 339


def test_kept_columns_in_a_group_are_not_highly_correlated():
    train = dp.train_block(load_joined())
    for g in dp.v_proposal(train):
        corr = train[g["keep"]].astype("float64").corr().abs()
        for i, a in enumerate(g["keep"]):
            for b in g["keep"][i + 1:]:
                assert corr.loc[a, b] <= dp.CORR_THRESHOLD
