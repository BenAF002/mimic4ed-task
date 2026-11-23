"""
Metrics utilities for model evaluation.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class BasicMetrics:
    """Binary-classification metrics, exposed as plain floats via `to_dict()` / `dict`.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities of the positive class.
        cm: Confusion matrix as a 2x2 array or a flat (tn, fp, fn, tp) sequence.
    """

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
                 cm: tuple | list | np.ndarray):
        self.cm = cm
        self.tn, self.fp, self.fn, self.tp = np.ravel(cm)
        self.accuracy = accuracy_score(y_true, y_pred)
        self.precision = precision_score(y_true, y_pred, zero_division=0)
        self.recall = recall_score(y_true, y_pred, zero_division=0)
        self.specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0.0
        self.f1 = f1_score(y_true, y_pred, zero_division=0)
        self.roc_auc = roc_auc_score(y_true, y_prob)
        self.average_precision = average_precision_score(y_true, y_prob)

    def to_dict(self) -> dict[str, float]:
        """Scalar metrics and confusion counts as `{name: float}`, ready for `mlflow.log_metrics`."""
        names = ["tn", "fp", "fn", "tp", "accuracy", "precision", "recall",
                 "specificity", "f1", "roc_auc", "average_precision"]
        return {k: float(getattr(self, k)) for k in names}

    @property
    def dict(self) -> dict[str, float]:
        """Alias for `to_dict()`."""
        return self.to_dict()
