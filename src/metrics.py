"""Các chỉ số đánh giá dùng chung."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, precision_recall_fscore_support


def classification_metrics(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict[str, float]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_pred_arr = np.asarray(list(y_pred), dtype=int)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true_arr,
        y_pred_arr,
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "macro_f1": float(f1_macro),
        "qwk": float(cohen_kappa_score(y_true_arr, y_pred_arr, weights="quadratic")),
    }
