"""Evaluation metrics (pure NumPy, no external deps).

Author: 晨星
"""

from __future__ import annotations

import numpy as np


def auc(scores_pos: np.ndarray, scores_neg: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (handles ties)."""
    pos = np.asarray(scores_pos, dtype=float)
    neg = np.asarray(scores_neg, dtype=float)
    n_pos, n_neg = pos.size, neg.size
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc needs both classes non-empty")
    all_s = np.concatenate([pos, neg])
    order = np.argsort(all_s, kind="mergesort")
    ranks = np.empty(all_s.size, dtype=float)
    i = 0
    while i < all_s.size:
        j = i
        while j + 1 < all_s.size and all_s[order[j + 1]] == all_s[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0  # average 1-based rank
        i = j + 1
    ranks_pos = ranks[:n_pos]
    u = ranks_pos.sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def tpr_at_fpr(scores_pos: np.ndarray, scores_neg: np.ndarray, fpr: float) -> float:
    """TPR at the empirical (1-fpr) quantile threshold of the negative scores."""
    neg = np.asarray(scores_neg, dtype=float)
    pos = np.asarray(scores_pos, dtype=float)
    thr = float(np.quantile(neg, 1.0 - fpr))
    return float(np.mean(pos > thr))


def kl_uniform_from_samples(xs: np.ndarray, V: int) -> float:
    """Empirical KL(uniform || empirical token distribution) = distortion proxy."""
    flat = np.asarray(xs, dtype=int).ravel()
    counts = np.bincount(flat, minlength=V).astype(float)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(np.sum(p * (np.log(p) + np.log(V))))
