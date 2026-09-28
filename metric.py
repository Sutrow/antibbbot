from __future__ import annotations
import numpy as np
TARGET_RECALL = 0.7


def pr_curve(y_true, score):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos = int(y_true.sum())
    if n_pos == 0:
        return (np.array([]), np.array([]))
    order = np.argsort(-score, kind='mergesort')
    y, s = (y_true[order], score[order])
    tp = np.cumsum(y)
    k = np.arange(1, len(y) + 1)
    ends = np.r_[s[1:] != s[:-1], True]
    return (tp[ends] / k[ends], tp[ends] / n_pos)


def precision_at_recall(y_true, score, recall: float=TARGET_RECALL):
    prec, rec = pr_curve(y_true, score)
    if len(prec) == 0:
        return float('nan')
    ok = rec >= recall
    return float(prec[ok].max()) if ok.any() else 0.0


def recall_at_fpr(y_true, score, fpr: float=0.01):
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    n_pos, n_neg = (int(y_true.sum()), int((1 - y_true).sum()))
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    order = np.argsort(-score, kind='mergesort')
    y, s = (y_true[order], score[order])
    ends = np.r_[s[1:] != s[:-1], True]
    tp = np.cumsum(y)[ends]
    fp = np.cumsum(1 - y)[ends]
    allowed = fp <= fpr * n_neg
    return float(tp[allowed].max() / n_pos) if allowed.any() else 0.0
