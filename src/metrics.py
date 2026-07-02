"""
Shared evaluation metrics for imbalanced classification.

Design choice that matters: on a 0.17%-positive problem, ROC-AUC is misleading.
It can look excellent (0.97+) while the model is still useless in practice,
because the huge negative class dominates the false-positive-rate axis.
Precision-Recall AUC (a.k.a. average precision) is the honest headline metric
here - it focuses on the positive (fraud) class. We report ROC-AUC too, but PR-AUC
leads, and saying *why* out loud is part of the point.

All "at threshold" metrics take an explicit decision threshold so the cost
analysis can sweep it. Default 0.5 is just a starting reference, not a
recommendation - choosing the threshold is a core deliverable, not a default.
"""
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix,
)


@dataclass
class Metrics:
    pr_auc: float          # average precision - THE headline for imbalance
    roc_auc: float
    precision: float
    recall: float
    f1: float
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    def as_row(self) -> dict:
        return asdict(self)


def evaluate(y_true, y_score, threshold: float = 0.5) -> Metrics:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    pr_auc = average_precision_score(y_true, y_score)
    try:
        roc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc = float("nan")  # only one class present
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return Metrics(
        pr_auc=pr_auc,
        roc_auc=roc,
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        threshold=float(threshold),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def best_f1_threshold(y_true, y_score, grid=None) -> float:
    """Pick the threshold that maximizes F1 on the given data.

    Used as one of the imbalance 'strategies' (threshold tuning) and as a
    sensible operating point before the dollar-cost analysis refines it.
    """
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best_t, best_f1 = 0.5, -1.0
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    for t in grid:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t
