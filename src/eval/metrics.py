# -*- coding: utf-8 -*-
"""评测指标：Macro-F1（主指标）、各类 F1（中立类为关键分析指标）、混淆矩阵。"""
from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def macro_f1(y_true, y_pred, average: str = "macro", labels=None) -> float:
    return float(f1_score(y_true, y_pred, average=average, labels=labels, zero_division=0))


def per_class_f1(y_true, y_pred, id_to_label=None) -> dict:
    labels = sorted(set(list(np.asarray(y_true))) | set(list(np.asarray(y_pred))))
    scores = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)
    out = {}
    for i, s in zip(labels, scores):
        key = id_to_label.get(int(i), int(i)) if id_to_label else int(i)
        out[key] = float(s)
    return out


def neutral_f1(y_true, y_pred, neutral_id: int = 2) -> float:
    """中立类 F1（本项目关键分析指标：隐式表达/反讽/多目标冲突最集中）。"""
    return float(f1_score(y_true, y_pred, labels=[neutral_id], average="macro", zero_division=0))


def full_report(y_true, y_pred, id_to_label: dict | None = None) -> dict:
    labels = sorted(set(list(np.asarray(y_true))) | set(list(np.asarray(y_pred))))
    target_names = [id_to_label[i] for i in labels] if id_to_label else [str(i) for i in labels]
    return {
        "macro_f1": macro_f1(y_true, y_pred, labels=labels),
        "per_class_f1": per_class_f1(y_true, y_pred, id_to_label),
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=target_names,
            zero_division=0, digits=4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
