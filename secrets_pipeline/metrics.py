"""Dependency-free ranking metrics and validation-only operating points.

Unknown labels are deliberately not accepted. Callers must keep unknown findings
as a separate alert-burden measurement, not silently count them as negatives.
"""
from __future__ import annotations

import math
from typing import Iterable


def _pairs(labels: Iterable[int], scores: Iterable[float]):
    ys, ps = list(labels), list(scores)
    if len(ys) != len(ps):
        raise ValueError("Labels and scores have different lengths")
    if any(y not in (0, 1) for y in ys):
        raise ValueError("Metrics require explicit binary labels")
    if any(not math.isfinite(float(p)) or not 0 <= float(p) <= 1 for p in ps):
        raise ValueError("Scores must be finite probabilities in [0,1]")
    return list(zip(ys, map(float, ps)))


def binary_metrics(labels, scores, threshold=0.5, beta=1.0):
    pairs = _pairs(labels, scores)
    if beta <= 0:
        raise ValueError("beta must be positive")
    if threshold is not None and (not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("Threshold must be None or a finite probability")
    tp = sum(y == 1 and threshold is not None and p >= threshold for y, p in pairs)
    fp = sum(y == 0 and threshold is not None and p >= threshold for y, p in pairs)
    fn = sum(y == 1 for y, _ in pairs) - tp
    tn = len(pairs) - tp - fp - fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    b2 = beta * beta
    fbeta = (1 + b2) * precision * recall / (b2 * precision + recall) if precision + recall else 0.0
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                fbeta=fbeta, predicted_positive=tp + fp, support=len(pairs),
                positive_support=tp + fn, threshold=threshold)


def average_precision(labels, scores):
    """Stepwise PR area, grouping equal scores (the sklearn AP convention)."""
    pairs = sorted(_pairs(labels, scores), key=lambda z: -z[1])
    positives = sum(y for y, _ in pairs)
    if not positives:
        return 0.0
    tp = count = 0
    ap = 0.0
    i = 0
    while i < len(pairs):
        score = pairs[i][1]
        added_tp = 0
        while i < len(pairs) and pairs[i][1] == score:
            added_tp += pairs[i][0]
            count += 1
            i += 1
        tp += added_tp
        ap += (added_tp / positives) * (tp / count)
    return ap


def wilson_lower(successes, trials, z=1.96):
    if trials <= 0:
        return 0.0
    p, z2 = successes / trials, z * z
    return (p + z2 / (2 * trials) - z * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials))) / (1 + z2 / trials)


def calibrate_thresholds(labels, scores, *, warn_beta=1.0, block_target_precision=0.98,
                         block_min_predictions=30, block_wilson_z=1.96):
    """Select WARN by F-beta; enable BLOCK only if its precision gate is met.

    BLOCK's precision is measured using a Wilson lower bound, not the optimistic
    point estimate. A missing qualifying operating point disables automatic
    blocking (``block=None``); callers must not substitute an arbitrary default.
    These intervals describe this validation set and are not deployment guarantees.
    """
    pairs = _pairs(labels, scores)
    if not pairs or {y for y, _ in pairs} != {0, 1}:
        raise ValueError("Threshold calibration needs validation positives and negatives")
    if not 0 < block_target_precision <= 1 or block_min_predictions < 1 or block_wilson_z < 0 or warn_beta <= 0:
        raise ValueError("Invalid BLOCK calibration constraints")
    ys, ps = zip(*pairs)
    # Calculate all unique thresholds in one descending pass (O(n log n)).
    sorted_pairs = sorted(pairs, key=lambda x: -x[1])
    total_pos = sum(ys)
    tp = fp = i = 0
    points = []
    while i < len(sorted_pairs):
        threshold = sorted_pairs[i][1]
        while i < len(sorted_pairs) and sorted_pairs[i][1] == threshold:
            tp += sorted_pairs[i][0]
            fp += 1 - sorted_pairs[i][0]
            i += 1
        p, r = tp / (tp + fp), tp / total_pos
        b2 = warn_beta * warn_beta
        fb = (1 + b2) * p * r / (b2 * p + r) if p + r else 0.0
        points.append(dict(threshold=threshold, precision=p, recall=r, fbeta=fb,
                           f1=2 * p * r / (p + r) if p + r else 0.0,
                           tp=tp, fp=fp, predicted_positive=tp + fp,
                           precision_lower=wilson_lower(tp, tp + fp, block_wilson_z)))
    warn = max(points, key=lambda d: (d['fbeta'], d['precision'], d['threshold']))
    eligible = [d for d in points if d['predicted_positive'] >= block_min_predictions
                and d['precision_lower'] >= block_target_precision
                and d['threshold'] >= warn['threshold']]
    block = max(eligible, key=lambda d: (d['recall'], d['precision_lower'], d['threshold'])) if eligible else None
    return dict(warn=warn['threshold'], block=block['threshold'] if block else None,
                block_supported=block is not None, warn_metrics=warn, block_metrics=block,
                block_reason='validation_precision_lower_bound_met' if block else 'no_validation_threshold_meets_precision_and_support_requirements',
                block_target_precision=block_target_precision, block_min_predictions=block_min_predictions,
                block_wilson_z=block_wilson_z, warn_beta=warn_beta,
                average_precision=average_precision(ys, ps), calibration_count=len(pairs),
                calibration_positive_count=total_pos)
