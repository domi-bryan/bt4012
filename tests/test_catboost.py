"""Contract, encoding policy and exact early-stopping metric."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from src import data
from src.catboost_pipeline import CatBoostPipeline, MISSING_CATEGORY
from src.features import agg_columns, build_feature_set
from src.models import catboost
from src.split import holdout_indices

SMALL = {"iterations": 12, "depth": 3, "thread_count": 1, "early_stopping_rounds": 3}


@pytest.fixture
def transactions():
    rng = np.random.default_rng(42)
    n = 600
    dt = np.linspace(86400, 151 * 86400, n).astype("int32")
    amount = rng.lognormal(3, 1, n).astype("float32")
    y = ((np.arange(n) % 7) == 0).astype("int8")
    category = np.where(y, "C", "W").astype(object)
    category[::11] = None
    category[-100:] = "future_only"
    category[1] = "__missing__"
    v1 = rng.normal(size=n).astype("float32")
    v1[::13] = np.nan
    return pd.DataFrame({
        "TransactionID": np.arange(n, dtype="int32"), "TransactionDT": dt,
        "TransactionDay": (dt // 86400).astype("int16"), "isFraud": y,
        "TransactionAmt": amount, "card1": (np.arange(n) % 5 + 1).astype("float32"),
        "addr1": np.ones(n, dtype="float32"), "D1": np.zeros(n, dtype="float32"),
        "ProductCD": pd.Categorical(category),
        "P_emaildomain": pd.Categorical(np.where(y, "a.test", "b.test")),
        "DeviceInfo": pd.Categorical([f"device_{i % 25}" for i in range(n)]),
        "V1": v1, "V2": np.ones(n, dtype="float32"),
    })


def test_pipeline_reuses_selection_and_training_vocabularies(transactions):
    tr, va, _ = holdout_indices(transactions)
    train, val = transactions.iloc[tr], transactions.iloc[va]
    before = transactions.copy(deep=True)
    pipe = CatBoostPipeline(v_drop=["V2"]).fit(train)
    actual = pipe.transform(val)
    shared = data.TreePipeline("lgbm", v_drop=["V2"]).fit(train).transform(val)
    assert list(actual) == list(shared) == data.feature_columns(train, ["V2"])
    assert "future_only" not in pipe.categories_["ProductCD"]
    assert actual.ProductCD.iloc[-1] == MISSING_CATEGORY
    assert pipe.transform(train).ProductCD.iloc[1] == "value:__missing__"
    assert pipe.transform(train).ProductCD.iloc[0] == MISSING_CATEGORY
    assert not actual[pipe.categorical_columns_].isna().any().any()
    pd.testing.assert_frame_equal(actual.drop(columns=pipe.categorical_columns_),
                                  shared.drop(columns=pipe.categorical_columns_))
    assert pd.api.types.is_numeric_dtype(actual.card1)
    pd.testing.assert_frame_equal(transactions, before)


@pytest.mark.parametrize("feature_set", ["F0", "F1", "F2"])
def test_shared_feature_sets_have_identical_model_columns(transactions, feature_set):
    df = build_feature_set(transactions, feature_set)
    pipe = CatBoostPipeline(v_drop=["V2"]).fit(df.iloc[:400])
    assert pipe.columns_ == data.feature_columns(df, ["V2"])
    assert "uid" not in pipe.columns_
    expected = ([] if feature_set == "F0" else agg_columns("card1"))
    expected += agg_columns("uid") if feature_set == "F2" else []
    assert [c for c in pipe.columns_ if c not in transactions] == expected


def test_forbidden_inputs_and_parameter_overrides(transactions):
    with pytest.raises(ValueError, match="raw uid"):
        CatBoostPipeline(v_drop=[]).fit(transactions.assign(uid="customer"))
    with pytest.raises(ValueError, match="card1 must remain numeric"):
        CatBoostPipeline(v_drop=[]).fit(transactions.assign(card1=pd.Categorical(transactions.card1)))
    X = CatBoostPipeline(v_drop=[]).fit(transactions).transform(transactions)
    for parameter, value in [("cat_features", ["card1"]), ("one_hot_max_size", 2),
                             ("simple_ctr", ["Borders"]), ("task_type", "GPU")]:
        with pytest.raises(ValueError, match="Unsupported"):
            catboost.fit(X, transactions.isFraud, params={parameter: value})
    with pytest.raises(ValueError, match="both classes"):
        catboost.fit(X, np.zeros(len(X)), params=SMALL)
    with pytest.raises(ValueError, match="Supply both"):
        catboost.fit(X, transactions.isFraud, X_val=X, params=SMALL)


def test_early_stopping_metric_is_unweighted_sklearn_ap():
    metric = catboost.AveragePrecision()
    y = [0, 1, 1, 0, 1]
    logits = [0.0, 0.4, 0.4, 0.6, -1.0]
    value, weight = metric.evaluate([logits], y, [1, 100, 100, 1, 100])
    assert value == average_precision_score(y, logits)
    assert metric.is_max_optimal()
    assert metric.get_final_error(value, weight) == value


def test_model_contract_weights_determinism_and_no_target_statistics(transactions, tmp_path):
    tr, va, _ = holdout_indices(transactions)
    pipe = CatBoostPipeline(v_drop=[]).fit(transactions.iloc[tr])
    X, V = pipe.transform(transactions.iloc[tr]), pipe.transform(transactions.iloc[va])
    y, v = transactions.isFraud.iloc[tr], transactions.isFraud.iloc[va]
    fitted = catboost.fit(X, y, V, v, SMALL, seed=2)
    score = catboost.predict_proba(fitted, V)
    assert score.shape == (len(va),) and np.isfinite(score).all()
    assert ((score >= 0) & (score <= 1)).all()
    assert fitted.run_metadata["resolved_scale_pos_weight"] == pytest.approx((y == 0).sum() / y.sum())
    assert "card1" not in fitted.run_metadata["categorical_columns"]
    assert 0 < fitted.run_metadata["best_iteration"] <= SMALL["iterations"]
    params = fitted.clf.get_all_params()
    assert params["has_time"] and params["task_type"] == "CPU"
    assert params["one_hot_max_size"] >= max(X[c].nunique() for c in pipe.categorical_columns_)
    assert fitted.clf.get_best_score()["validation"]["AveragePrecision"] == pytest.approx(
        average_precision_score(v, score))
    path = tmp_path / "model.json"
    fitted.clf.save_model(str(path), format="json")
    model_json = json.loads(path.read_text())
    assert not model_json["features_info"].get("ctrs")
    assert all(split["split_type"] != "OnlineCtr" for tree in model_json["oblivious_trees"]
               for split in (tree["splits"] or []))
    repeated = catboost.fit(X, y, V, v, SMALL, seed=2)
    np.testing.assert_array_equal(score, repeated.predict_proba(V))
    assert not (data.ROOT / "catboost_info").exists()
