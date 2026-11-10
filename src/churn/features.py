from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

_BINARY_MAP = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}


class TelcoFeatureEncoder(BaseEstimator, TransformerMixin):
    """Reproduces the notebook's manual binary-map + one-hot encoding as a
    fittable transformer, so it can be saved with the model and applied
    consistently (including to unseen categories) at inference time."""

    def __init__(self, binary_cols: list[str], categorical_cols: list[str]):
        self.binary_cols = binary_cols
        self.categorical_cols = categorical_cols

    def fit(self, X: pd.DataFrame, y=None) -> "TelcoFeatureEncoder":
        self.categories_ = {
            col: sorted(X[col].dropna().unique().tolist()) for col in self.categorical_cols
        }
        transformed = self._transform_core(X)
        self.output_columns_ = transformed.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        transformed = self._transform_core(X)
        return transformed.reindex(columns=self.output_columns_, fill_value=0)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.output_columns_)

    def _transform_core(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

        for col in self.binary_cols:
            df[col] = df[col].map(_BINARY_MAP).astype(int)

        for col in self.categorical_cols:
            cat = pd.Series(
                pd.Categorical(df[col], categories=self.categories_[col]), index=df.index
            )
            dummies = pd.get_dummies(cat, prefix=col, drop_first=True)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

        no_internet_cols = [c for c in df.columns if "No internet service" in c]
        if no_internet_cols:
            df["No_internet_service"] = df[no_internet_cols].any(axis=1).astype(int)
            df = df.drop(columns=no_internet_cols)

        no_phone_col = next((c for c in df.columns if "No phone service" in c), None)
        if no_phone_col is not None:
            df["No_phone_service"] = df[no_phone_col].astype(int)
            df = df.drop(columns=[no_phone_col])

        bool_cols = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        return df
