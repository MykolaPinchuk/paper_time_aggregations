from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class Metrics:
    roc_auc: float
    pr_auc: float


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Metrics:
    return Metrics(
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        pr_auc=float(average_precision_score(y_true, y_prob)),
    )

