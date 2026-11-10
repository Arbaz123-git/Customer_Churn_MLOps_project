from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn.features import TelcoFeatureEncoder


def build_pipeline(
    params: dict[str, Any],
    y_train: pd.Series,
    binary_cols: list[str],
    categorical_cols: list[str],
) -> Pipeline:
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = XGBClassifier(**params, scale_pos_weight=scale_pos_weight)

    return Pipeline(
        [
            ("features", TelcoFeatureEncoder(binary_cols, categorical_cols)),
            ("model", model),
        ]
    )
