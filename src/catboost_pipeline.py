"""Adapt the shared LightGBM tree input to CatBoost without changing feature sets."""
import pandas as pd

from src.data import TreePipeline

MISSING_CATEGORY = "__missing__"


class CatBoostPipeline(TreePipeline):
    """Keep shared feature selection, numeric NaN and fold-local category vocabularies.

    CatBoost requires nonmissing strings for categorical inputs. Prefix real values
    so a literal '__missing__' cannot collide with the missing/unseen marker.
    card1 remains numeric; raw uid is never an input to the shared F0/F1/F2 models.
    """

    def __init__(self, v_drop=None, columns=None):
        super().__init__(kind="lgbm", v_drop=v_drop, columns=columns)

    def fit(self, train_df):
        selected = train_df.columns if self.columns is None else self.columns
        if "uid" in selected:
            raise ValueError("The raw uid column must not be a model input")
        if "card1" in train_df and not pd.api.types.is_numeric_dtype(train_df.card1):
            raise ValueError("card1 must remain numeric, as in the shared pipelines")
        super().fit(train_df)
        self.categorical_columns_ = list(self.categories_)
        for column in self.columns_:
            if column not in self.categories_ and not pd.api.types.is_numeric_dtype(train_df[column]):
                raise ValueError(f"{column}: expected numeric or shared category dtype")
        return self

    def transform(self, df):
        out = super().transform(df)
        for column in self.categorical_columns_:
            out[column] = ("value:" + out[column].astype("string")).fillna(
                MISSING_CATEGORY).astype(object)
        return out
