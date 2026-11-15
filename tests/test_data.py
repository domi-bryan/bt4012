import pandas as pd
import pytest

from src import data


@pytest.fixture
def raw_dir(tmp_path):
    pd.DataFrame({
        "TransactionID": [3, 1, 2, 4],
        "isFraud": [0, 1, 0, 0],
        "TransactionDT": [86400 * 2, 86400, 86400, 100],
        "TransactionAmt": [10.5, 20.0, 5.0, 1.0],
        "ProductCD": ["W", "C", "W", "H"],
        "card1": [111, 222, 333, 444],
        "M1": ["T", None, "F", "T"],
    }).to_csv(tmp_path / "train_transaction.csv", index=False)
    pd.DataFrame({
        "TransactionID": [1, 4],
        "id_01": [-5.0, 0.0],
        "id_12": ["NotFound", "Found"],
        "DeviceType": ["mobile", "desktop"],
    }).to_csv(tmp_path / "train_identity.csv", index=False)
    return tmp_path


def test_join_sort_and_flags(raw_dir):
    df = data.load_raw(raw_dir)
    assert len(df) == 4
    # sorted by DT then ID, index reset
    assert df["TransactionID"].tolist() == [4, 1, 2, 3]
    assert list(df.index) == [0, 1, 2, 3]
    assert df["has_identity"].tolist() == [1, 1, 0, 0]
    assert df["TransactionDay"].tolist() == [0, 1, 1, 2]
    assert df.loc[df.TransactionID == 2, "id_01"].isna().all()


def test_dtypes(raw_dir):
    df = data.load_raw(raw_dir)
    for col in ["ProductCD", "M1", "id_12", "DeviceType"]:
        assert df[col].dtype == "category"
    assert df["TransactionAmt"].dtype == "float32"
    assert df["id_01"].dtype == "float32"
    assert df["card1"].dtype == "float32"


def test_parquet_roundtrip(raw_dir, tmp_path):
    df = data.load_raw(raw_dir)
    path = data.save_joined(df, tmp_path / "out" / "joined.parquet")
    back = data.load_joined(path)
    pd.testing.assert_frame_equal(df, back)


@pytest.mark.skipif(not data.JOINED_PATH.exists(), reason="joined.parquet not built")
def test_real_joined_table():
    df = data.load_joined()
    assert len(df) == 590_540
    assert df["isFraud"].sum() == 20_663
    assert df["has_identity"].sum() == 144_233
    assert df["TransactionDT"].is_monotonic_increasing
    obj = [c for c in df.columns if df[c].dtype == object]
    assert obj == []


def test_cleaned_preserves_missingness_and_extreme_values(raw_dir, tmp_path):
    df = data.load_raw(raw_dir)
    df.loc[0, "TransactionAmt"] = 1_000_000
    before = df.copy(deep=True)
    path = data.save_cleaned(df, tmp_path / "cleaned.parquet")
    pd.testing.assert_frame_equal(data.load_cleaned(path), before, check_exact=True)
    pd.testing.assert_frame_equal(df, before, check_exact=True)


@pytest.mark.parametrize("column,value,message", [
    ("TransactionID", 1, "unique"),
    ("isFraud", 2, "binary"),
    ("TransactionAmt", 0, "positive"),
    ("TransactionAmt", float("nan"), "nonmissing"),
    ("card1", float("inf"), "infinite"),
    ("TransactionDT", -1, "nonnegative"),
    ("TransactionDay", 99, "does not match"),
    ("has_identity", 0, "identity values"),
])
def test_cleaned_rejects_corruption_before_writing(raw_dir, tmp_path, column, value, message):
    df = data.load_raw(raw_dir)
    df.loc[0, column] = value
    path = tmp_path / "cleaned.parquet"
    with pytest.raises(ValueError, match=message):
        data.save_cleaned(df, path)
    assert not path.exists()


def test_cleaned_rejects_out_of_order_rows(raw_dir):
    df = data.load_raw(raw_dir).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="ordered"):
        data.validate_cleaned(df)


@pytest.mark.parametrize("path", [data.JOINED_PATH, data.RAW_DIR / "train_transaction.csv"])
def test_cleaned_cannot_overwrite_inputs(raw_dir, path):
    with pytest.raises(ValueError, match="must not overwrite"):
        data.save_cleaned(data.load_raw(raw_dir), path)
