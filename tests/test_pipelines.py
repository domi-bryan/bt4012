import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src import data
from src.split import holdout_indices

N_TRAIN = 80


@pytest.fixture
def toy():
    """100 rows: train = first 80, validation = last 20 with extreme values and new levels."""
    rng = np.random.default_rng(0)
    n = 100
    amt = rng.lognormal(3, 1, n).astype("float32")
    amt[N_TRAIN:] = 1e6
    v1 = rng.normal(size=n).astype("float32")
    v1[rng.random(n) < 0.3] = np.nan
    v2 = rng.normal(size=n).astype("float32")   # no NaN in train, NaN in validation
    v2[N_TRAIN:N_TRAIN + 5] = np.nan
    small = rng.choice(["a", "b", "c"], n).astype(object)
    small[::7] = None
    small[N_TRAIN] = "zz"                        # level never seen in train
    big = np.array([f"d{i % 25}" for i in range(n)], dtype=object)
    big[N_TRAIN + 1] = "unseen"
    return pd.DataFrame({
        "TransactionID": np.arange(n, dtype="int32"),
        "isFraud": (rng.random(n) < 0.2).astype("int8"),
        "TransactionDT": np.arange(n, dtype="int32") * 1000,
        "TransactionDay": np.zeros(n, dtype="int16"),
        "TransactionAmt": amt,
        "V1": v1,
        "V2": v2,
        "V3": rng.normal(size=n).astype("float32"),
        "has_identity": (rng.random(n) < 0.5).astype("int8"),
        "small": pd.Categorical(small),
        "big": pd.Categorical(big),
    })


def split(df):
    return df.iloc[:N_TRAIN], df.iloc[N_TRAIN:]


def test_feature_columns_drop_ids_time_and_pruned_v(toy):
    cols = data.feature_columns(toy, v_drop=["V3"])
    assert cols == ["TransactionAmt", "V1", "V2", "has_identity", "small", "big"]


def test_linear_stats_come_from_training_rows_only(toy):
    train, _ = split(toy)
    pipe = data.LinearPipeline(v_drop=["V3"]).fit(train)
    num = pipe.numeric_
    imp = SimpleImputer(strategy="median").fit(train[num].astype("float64"))
    scaler = StandardScaler().fit(imp.transform(train[num].astype("float64")))
    for i, c in enumerate(num):
        assert pipe.medians_[c] == pytest.approx(np.nanmedian(train[c].astype("float64")))
        assert pipe.medians_[c] == pytest.approx(imp.statistics_[i])
        assert pipe.means_[c] == pytest.approx(scaler.mean_[i])
        assert pipe.stds_[c] == pytest.approx(scaler.scale_[i])
    # the extreme validation amounts would move the statistics if they leaked in
    full = data.LinearPipeline(v_drop=["V3"]).fit(toy)
    assert full.means_["TransactionAmt"] > 10 * pipe.means_["TransactionAmt"]


def test_linear_output_has_no_nan_and_scales_train(toy):
    train, val = split(toy)
    pipe = data.LinearPipeline(v_drop=["V3"]).fit(train)
    out_val = pipe.transform(val)
    assert not out_val.isna().any().any()
    assert (out_val.dtypes == "float32").all()
    assert list(out_val.columns) == pipe.columns_
    out_train = pipe.transform(train)
    assert np.allclose(out_train[pipe.numeric_].mean(), 0, atol=1e-5)
    assert np.allclose(out_train[pipe.numeric_].std(ddof=0), 1, atol=1e-4)


def test_linear_indicators_and_encodings(toy):
    train, val = split(toy)
    pipe = data.LinearPipeline(v_drop=["V3"]).fit(train)
    # indicators only for columns with missing training values (V2 has none in train)
    assert pipe.indicators_ == ["V1"]
    assert list(pipe.onehot_) == ["small"] and list(pipe.freq_) == ["big"]
    assert pipe.onehot_["small"] == ["a", "b", "c"]
    out = pipe.transform(val)
    onehot = [c for c in out if c.startswith("small==")]
    assert onehot == ["small==a", "small==b", "small==c", "small==__nan__"]
    # an unseen level gets all zeros; every other row has exactly one hot column
    assert out[onehot].iloc[0].sum() == 0
    assert (out[onehot].iloc[1:].sum(axis=1) == 1).all()
    assert out["big__freq"].iloc[1] == 0
    assert out["big__freq"].iloc[0] == pytest.approx((train["big"] == val["big"].iloc[0]).mean())
    # V2 has NaN only in validation: imputed with the training median, no indicator
    assert "V2__missing" not in out
    v2_imputed = (pipe.medians_["V2"] - pipe.means_["V2"]) / pipe.stds_["V2"]
    assert out["V2"].iloc[0] == pytest.approx(v2_imputed, abs=1e-6)


def test_tree_lgbm_keeps_nan_and_training_categories(toy):
    train, val = split(toy)
    pipe = data.TreePipeline("lgbm", v_drop=["V3"]).fit(train)
    out = pipe.transform(val)
    assert out["V1"].isna().sum() == val["V1"].isna().sum()
    assert out["small"].dtype == "category"
    assert list(out["small"].cat.categories) == ["a", "b", "c"]
    assert pd.isna(out["small"].iloc[0])        # "zz" was never seen in train
    # train and validation share identical category dtypes
    assert pipe.transform(train)["big"].dtype == out["big"].dtype


def test_tree_rf_fills_sentinel_and_codes(toy):
    train, val = split(toy)
    pipe = data.TreePipeline("rf", v_drop=["V3"]).fit(train)
    out = pipe.transform(val)
    assert not out.isna().any().any()
    assert (out.dtypes != "category").all()
    assert out["small"].iloc[0] == data.SENTINEL
    assert set(out["small"].iloc[1:]) <= {0.0, 1.0, 2.0, data.SENTINEL}
    assert (out.loc[val["V1"].isna(), "V1"] == data.SENTINEL).all()


def test_v_list_warns_until_approved():
    with pytest.warns(UserWarning, match="not approved"):
        keep, drop = data.load_v_columns()
    assert len(keep) == 229 and len(drop) == 110 and not set(keep) & set(drop)


@pytest.fixture(scope="module")
def real():
    if not data.CLEANED_PATH.exists():
        pytest.skip("cleaned.parquet not built")
    df = data.load_cleaned()
    tr, va, _ = holdout_indices(df)
    with pytest.warns(UserWarning):
        _, drop = data.load_v_columns()
    return df.iloc[tr], df.iloc[va], drop


def test_real_pipelines(real):
    train, val, drop = real
    lin = data.LinearPipeline(v_drop=drop).fit(train)
    out = lin.transform(val)
    assert not out.isna().any().any()
    assert np.isfinite(out.to_numpy()).all()
    rf = data.TreePipeline("rf", v_drop=drop).fit(train).transform(val)
    assert not rf.isna().any().any()
    lgbm = data.TreePipeline("lgbm", v_drop=drop).fit(train).transform(val)
    assert len(lgbm.columns) == len(rf.columns) == len(data.feature_columns(train, drop))
    assert not set(drop) & set(lgbm.columns)
