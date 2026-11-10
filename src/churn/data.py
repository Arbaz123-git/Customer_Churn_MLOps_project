from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import Config


def load_raw(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    if cfg.id_col in df.columns:
        df = df.drop(columns=[cfg.id_col])

    if cfg.target in df.columns and not pd.api.types.is_numeric_dtype(df[cfg.target]):
        df[cfg.target] = df[cfg.target].replace({"Yes": 1, "No": 0}).astype(int)

    return df


def split_xy(
    df: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[cfg.target])
    y = df[cfg.target]

    stratify = y if cfg.split.stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.split.test_size,
        random_state=cfg.split.random_state,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test
