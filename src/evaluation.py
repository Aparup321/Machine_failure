"""Metric helpers shared by training, cross-validation and prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Order used in every table / figure.
METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
PRETTY = {
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1-score",
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
}


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """All headline metrics for the failure (positive) class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
    return metrics


def markdown_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """Minimal DataFrame -> GitHub markdown table (no tabulate dependency)."""
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "-"
            return floatfmt.format(value)
        return str(value)

    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = [
        "| " + " | ".join(fmt(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, divider, *rows])


def metrics_row(name: str, metrics: dict) -> dict:
    row = {"model": name}
    row.update({c: metrics.get(c, np.nan) for c in METRIC_COLUMNS})
    for cell in ("tn", "fp", "fn", "tp"):
        row[cell] = metrics.get(cell)
    return row
