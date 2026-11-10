from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

SWEEP_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def compute_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, y_pred, pos_label=1),
        "recall": recall_score(y_true, y_pred, pos_label=1),
        "f1": f1_score(y_true, y_pred, pos_label=1),
        "roc_auc": roc_auc_score(y_true, proba),
    }


def threshold_sweep(
    y_true: pd.Series, proba: np.ndarray, thresholds: list[float] = SWEEP_THRESHOLDS
) -> pd.DataFrame:
    rows = []
    for thresh in thresholds:
        y_pred = (proba >= thresh).astype(int)
        rows.append(
            {
                "threshold": thresh,
                "precision": precision_score(y_true, y_pred, pos_label=1),
                "recall": recall_score(y_true, y_pred, pos_label=1),
                "f1": f1_score(y_true, y_pred, pos_label=1),
            }
        )
    return pd.DataFrame(rows)
