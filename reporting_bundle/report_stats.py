"""Dependency-free statistics for saved, explicitly labeled candidate scores.

These are candidate-classification statistics, not end-to-end span metrics.
Unknown labels are excluded rather than treated as negatives. No function in
this module returns source text, candidate values, or individual score rows.
"""
from __future__ import annotations

from collections import defaultdict
import math


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    try:
        value = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number in [0, 1]") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return value


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def candidate_statistics(rows, threshold, bins=10):
    """Summarize an iterable of ``{'label': 0|1|None, 'score': float}`` rows.

    Scores, including those on unknown rows, and the inclusive operating
    threshold must be finite probabilities. Labels must be explicit integers
    0/1 or None; booleans, missing fields and other values are rejected.

    Equal scores enter ranking curves together. Average precision is the
    stepwise precision-recall integral; ROC AUC uses trapezoids between grouped
    operating points. AP is None without positives and ROC AUC is None without
    both classes. Other undefined ratios are None, never synthetic zeroes.

    Reliability lower/upper are the fixed bin edges. The last bin includes 1;
    other bins are left-closed and right-open. Empty bins are retained with
    null means/rates. ECE uses these same bins and the labeled population.
    Log loss clips probabilities to [1e-15, 1 - 1e-15] to keep endpoint errors
    finite. Calibration diagnostics describe the supplied scores, not a fit.
    """
    threshold = _probability(threshold, "threshold")
    if type(bins) is not int or bins < 1:
        raise ValueError("bins must be a positive integer")

    count = unknown = 0
    pairs = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or "label" not in row or "score" not in row:
            raise ValueError(f"Row {index} must contain label and score")
        score = _probability(row["score"], f"Row {index} score")
        label = row["label"]
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError(f"Row {index} label must be 0, 1 or None")
        count += 1
        if label is None:
            unknown += 1
        else:
            pairs.append((label, score))

    n = len(pairs)
    positives = sum(label for label, _ in pairs)
    negatives = n - positives
    tp = sum(label == 1 and score >= threshold for label, score in pairs)
    fp = sum(label == 0 and score >= threshold for label, score in pairs)
    fn, tn = positives - tp, negatives - fp
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, positives)
    specificity = _ratio(tn, negatives)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    # Grouping ties makes AP, ROC and threshold curves independent of row order.
    ordered = sorted(pairs, key=lambda item: -item[1])
    pr_x, pr_y = ([0.0], [1.0]) if positives else ([], [])
    roc_x, roc_y = ([0.0], [0.0]) if positives and negatives else ([], [])
    threshold_x, threshold_p, threshold_r, threshold_f1 = [], [], [], []
    ap = 0.0 if positives else None
    area = 0.0 if positives and negatives else None
    seen_tp = seen_fp = i = 0
    while i < n:
        score = ordered[i][1]
        added_positives = 0
        while i < n and ordered[i][1] == score:
            added_positives += ordered[i][0]
            seen_tp += ordered[i][0]
            seen_fp += 1 - ordered[i][0]
            i += 1
        current_p = seen_tp / (seen_tp + seen_fp)
        current_r = _ratio(seen_tp, positives)
        current_f1 = _ratio(2 * seen_tp, 2 * seen_tp + seen_fp + positives - seen_tp)
        threshold_x.append(score)
        threshold_p.append(current_p)
        threshold_r.append(current_r)
        threshold_f1.append(current_f1)
        if positives:
            pr_x.append(current_r)
            pr_y.append(current_p)
            ap += added_positives / positives * current_p
        if positives and negatives:
            current_fpr = seen_fp / negatives
            area += (current_fpr - roc_x[-1]) * (current_r + roc_y[-1]) / 2
            roc_x.append(current_fpr)
            roc_y.append(current_r)

    bucket_count, bucket_positives = [0] * bins, [0] * bins
    bucket_scores = [0.0] * bins
    epsilon = 1e-15
    for label, score in pairs:
        bucket = min(int(score * bins), bins - 1)
        bucket_count[bucket] += 1
        bucket_positives[bucket] += label
        bucket_scores[bucket] += score

    def log_error(label, score):
        clipped = min(1.0 - epsilon, max(epsilon, score))
        return -math.log(clipped) if label else -math.log1p(-clipped)

    reliability = []
    ece_parts = []
    for bucket in range(bins):
        size = bucket_count[bucket]
        mean_score = bucket_scores[bucket] / size if size else None
        positive_rate = bucket_positives[bucket] / size if size else None
        reliability.append({"mean_score": mean_score, "positive_rate": positive_rate,
                            "count": size, "lower": bucket / bins,
                            "upper": (bucket + 1) / bins})
        if size:
            ece_parts.append(size / n * abs(mean_score - positive_rate))

    metrics = {
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "precision": precision, "recall": recall,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn),
        "accuracy": _ratio(tp + tn, n),
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "specificity": specificity, "npv": _ratio(tn, tn + fn),
        "fpr": _ratio(fp, negatives), "fnr": _ratio(fn, positives),
        "mcc": _ratio(tp * tn - fp * fn, mcc_denominator),
        "average_precision": ap, "roc_auc": area,
        "brier_score": math.fsum((score - label) ** 2 for label, score in pairs) / n if n else None,
        "log_loss": math.fsum(log_error(label, score) for label, score in pairs) / n if n else None,
        "ece": math.fsum(ece_parts) if n else None,
    }
    return {
        "count": count, "labeled_count": n, "unknown_count": unknown,
        "positive_count": positives, "negative_count": negatives,
        "threshold": threshold, "metrics": metrics,
        "curves": {
            "precision_recall": {"x": pr_x, "y": pr_y},
            "roc": {"x": roc_x, "y": roc_y},
            "thresholds": {"x": threshold_x, "y_precision": threshold_p,
                           "y_recall": threshold_r, "y_f1": threshold_f1},
        },
        "reliability": reliability,
    }


def summarize_groups(rows, keys, value_key):
    """Group finite numeric values, returning n/mean/sample-std/min/max.

    ``keys`` is an iterable of grouping field names (one string is also accepted).
    Values of None are skipped; groups containing no numeric values are omitted.
    A single observation has ``std=None`` because sample variance needs n >= 2.
    Input rows are not modified. Mixed-type grouping values are sorted stably by
    type and representation, without coercing the returned grouping fields.
    """
    keys = [keys] if isinstance(keys, str) else list(keys)
    if any(not isinstance(key, str) for key in keys) or len(keys) != len(set(keys)):
        raise ValueError("keys must contain unique field names")
    if not isinstance(value_key, str):
        raise ValueError("value_key must be a field name")
    reserved = {"n", "mean", "std", "min", "max"}
    if any(key in reserved for key in keys):
        raise ValueError("Grouping field names conflict with summary fields")
    grouped = defaultdict(list)
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict) or value_key not in row or any(key not in row for key in keys):
            raise ValueError(f"Row {index} lacks a grouping/value field")
        value = row[value_key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Row {index} {value_key} must be a finite number or None")
        try:
            value = float(value)
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Row {index} {value_key} must be a finite number or None") from exc
        if not math.isfinite(value):
            raise ValueError(f"Row {index} {value_key} must be a finite number or None")
        group = tuple(row[key] for key in keys)
        try:
            grouped[group].append(float(value))
        except TypeError as exc:
            raise ValueError(f"Row {index} grouping values must be hashable") from exc

    output = []
    ordering = lambda group: tuple((type(value).__name__, repr(value)) for value in group)
    for group in sorted(grouped, key=ordering):
        values = grouped[group]
        n = len(values)
        mean = math.fsum(value / n for value in values)
        std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (n - 1)) if n > 1 else None
        output.append({**dict(zip(keys, group)), "n": n, "mean": mean,
                       "std": std, "min": min(values), "max": max(values)})
    return output
