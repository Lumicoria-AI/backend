"""Pure-Python implementations of the offline eval metrics.

We deliberately avoid pulling in scikit-learn for three small functions
that we control end-to-end:

  macro_f1            — multi-class macro-averaged F1.
  ndcg_at_k           — Normalised Discounted Cumulative Gain at K.
  expected_calibration_error
                      — ECE with equal-width bins (10 by default).

All three return a float in [0, 1] (well, ECE in [0, 1] where 0 is
perfectly calibrated). Empty inputs return 0.0 to keep CI deterministic.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


# ─────────────────────────────────────────────────────────────────────
# Classification — macro F1
# ─────────────────────────────────────────────────────────────────────


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro-averaged F1 across all classes that appear in y_true ∪ y_pred.

    Per-class F1 is 0 when the class never appears in the predictions
    (true positives = 0). Macro avg gives every class equal weight
    regardless of support — what we want for an eval over a small
    hand-labelled fixture set.
    """
    if not y_true or len(y_true) != len(y_pred):
        return 0.0

    classes = set(y_true) | set(y_pred)
    per_class: List[float] = []
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        if tp == 0:
            per_class.append(0.0)
            continue
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            per_class.append(0.0)
        else:
            per_class.append(2 * precision * recall / (precision + recall))

    return sum(per_class) / len(per_class) if per_class else 0.0


def per_class_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
) -> List[Tuple[str, float]]:
    """Return [(class, f1)] for each class — useful for the eval report."""
    if not y_true:
        return []
    classes = sorted(set(y_true) | set(y_pred))
    out: List[Tuple[str, float]] = []
    for c in classes:
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == c and b == c)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a != c and b == c)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == c and b != c)
        if tp == 0:
            out.append((c, 0.0))
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        out.append((c, 2 * p * r / (p + r) if (p + r) else 0.0))
    return out


# ─────────────────────────────────────────────────────────────────────
# Ranking — NDCG@K
# ─────────────────────────────────────────────────────────────────────


def _dcg(relevances: Iterable[float]) -> float:
    """Discounted Cumulative Gain at rank i is rel_i / log2(i + 2)."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(
    predicted_order: Sequence[str],
    ideal_order: Sequence[str],
    k: int = 5,
) -> float:
    """Normalised DCG at K.

    Relevance grade per item is its position in ``ideal_order`` mapped
    onto a graded relevance: ideal rank 0 → k, rank 1 → k-1, …, rank
    k-1 → 1. Items not in ideal get 0.

    Empty ideal returns 0.0. Empty prediction returns 0.0.
    """
    if not predicted_order or not ideal_order:
        return 0.0

    ideal_relevance_map = {
        item: max(0, k - i) for i, item in enumerate(ideal_order[:k])
    }
    pred_relevances = [ideal_relevance_map.get(item, 0) for item in predicted_order[:k]]
    ideal_relevances = [k - i for i in range(min(k, len(ideal_order)))]

    dcg_score = _dcg(pred_relevances)
    idcg_score = _dcg(ideal_relevances)
    return (dcg_score / idcg_score) if idcg_score > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────
# Calibration — ECE
# ─────────────────────────────────────────────────────────────────────


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool],
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error with equal-width bins.

    ECE = Σ over bins B_i of  (|B_i| / N) * |conf(B_i) - acc(B_i)|

    Where conf(B_i) is the mean predicted confidence within the bin and
    acc(B_i) is the empirical accuracy within the bin.

    Lower is better. A perfectly calibrated model has ECE = 0.
    """
    if not confidences or len(confidences) != len(correctness):
        return 0.0
    if n_bins < 1:
        return 0.0

    total = len(confidences)
    bin_edges = [i / n_bins for i in range(n_bins + 1)]

    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Last bin is inclusive of the right edge so confidence == 1.0 lands.
        if i == n_bins - 1:
            in_bin = [
                (c, ok) for c, ok in zip(confidences, correctness)
                if lo <= c <= hi
            ]
        else:
            in_bin = [
                (c, ok) for c, ok in zip(confidences, correctness)
                if lo <= c < hi
            ]
        if not in_bin:
            continue
        avg_conf = sum(c for c, _ in in_bin) / len(in_bin)
        avg_acc = sum(1 for _, ok in in_bin if ok) / len(in_bin)
        ece += (len(in_bin) / total) * abs(avg_conf - avg_acc)

    return ece
