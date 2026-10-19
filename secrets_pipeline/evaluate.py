"""End-to-end span evaluation, keeping unknown alerts separate from false alarms.

Prediction rows contain document_id,start,end,score. No training labels are
required to propose candidates. Missing proposals remain false negatives.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict

from .schema import read_jsonl, validate_corpus, write_json


def overlap(left, right):
    return left["start"] < right["end"] and left["end"] > right["start"]


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def _matching(predictions, spans):
    """Maximum-cardinality one-to-one interval match (augmenting paths)."""
    graph = [[j for j, span in enumerate(spans) if overlap(pred, span)] for pred in predictions]
    matched = {}

    def augment(i, seen):
        for j in graph[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False

    for i in range(len(predictions)):
        augment(i, set())
    return matched


def _score_subset(docs, by_doc, threshold):
    tp = fp_negative = fp_duplicate = unknown = positives = negatives = negative_hit = 0
    proposals = alerts = candidate_reached = lines = 0
    categories = defaultdict(lambda: {"positive_spans": 0, "detected": 0, "candidate_reached": 0,
                                       "negative_spans": 0, "flagged_negative_spans": 0})
    missed = []
    for doc in docs:
        rows = by_doc.get(doc["id"], [])
        active = sorted((p for p in rows if p["score"] >= threshold), key=lambda p: (-p["score"], p["start"], p["end"]))
        pos = [s for s in doc["spans"] if s["label"] == 1]
        neg = [s for s in doc["spans"] if s["label"] == 0]
        matched = _matching(active, pos)
        reachable = _matching(rows, pos)
        found_predictions = set(matched.values())
        for i, pred in enumerate(active):
            if i in found_predictions:
                continue
            if any(overlap(pred, s) for s in pos):
                fp_duplicate += 1
            elif any(overlap(pred, s) for s in neg):
                fp_negative += 1
            else:
                unknown += 1
        for j, span in enumerate(pos):
            category = categories[span["category"]]
            category["positive_spans"] += 1
            category["detected"] += j in matched
            category["candidate_reached"] += j in reachable
            if j not in matched:
                missed.append({"document_id": doc["id"], "span_id": span["id"], "category": span["category"],
                               "candidate_available": j in reachable})
        for span in neg:
            hit = any(overlap(pred, span) for pred in active)
            negative_hit += hit
            categories[span["category"]]["negative_spans"] += 1
            categories[span["category"]]["flagged_negative_spans"] += hit
        tp += len(matched)
        candidate_reached += len(reachable)
        positives += len(pos)
        negatives += len(neg)
        proposals += len(rows)
        alerts += len(active)
        lines += max(1, len(doc["text"].splitlines()))
    fp = fp_negative + fp_duplicate
    for category in categories.values():
        category["recall"] = category["detected"] / category["positive_spans"] if category["positive_spans"] else None
        category["candidate_recall_ceiling"] = category["candidate_reached"] / category["positive_spans"] if category["positive_spans"] else None
        category["negative_flag_rate"] = category["flagged_negative_spans"] / category["negative_spans"] if category["negative_spans"] else None
    return {"documents": len(docs), "positive_spans": positives, "negative_spans": negatives,
            "true_positive": tp, "false_positive": fp, "false_negative": positives - tp,
            "explicit_negative_alerts": fp_negative, "duplicate_positive_alerts": fp_duplicate,
            "unknown_alerts": unknown, "unknown_alerts_per_kloc": unknown * 1000 / lines if lines else 0.0,
            "negative_flag_rate": negative_hit / negatives if negatives else None,
            "candidate_recall_ceiling": candidate_reached / positives if positives else None,
            "candidate_spans": proposals, "alerts": alerts, "lines": lines,
            **prf(tp, fp, positives - tp), "by_category": dict(categories), "missed_spans": missed}


def evaluate_documents(docs, predictions, threshold: float = 0.5):
    """Evaluate all documents and source/split/language strata.

    Precision is annotation-conditional: unknown regions are NOT certified clean.
    Duplicate predictions over an already matched secret count as excess alerts.
    The same threshold is applied everywhere; this function never tunes on test.
    """
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a finite probability in [0,1]")
    docs = list(docs)
    index = {doc["id"]: doc for doc in docs}
    if len(index) != len(docs):
        raise ValueError("Duplicate document ids")
    by_doc, seen = defaultdict(list), set()
    for pred in predictions:
        did = pred.get("document_id")
        if did not in index:
            raise ValueError(f"Prediction refers to unknown document {did}")
        start, end, score = pred.get("start"), pred.get("end"), pred.get("score")
        if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(index[did]["text"]):
            raise ValueError(f"Invalid prediction offsets in {did}")
        if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid prediction probability in {did}")
        key = (did, start, end)
        if key in seen:
            raise ValueError(f"Duplicate prediction span in {did}; fuse detectors before evaluation")
        seen.add(key)
        by_doc[did].append(pred)
    result = {"threshold": threshold, "protocol": "character-span-overlap-one-to-one",
              "precision_scope": "explicit-labels-plus-duplicate-positive-alerts",
              "unknown_regions_are_negative": False, "overall": _score_subset(docs, by_doc, threshold)}
    for axis in ("split", "source", "language"):
        groups = defaultdict(list)
        for doc in docs:
            groups[doc[axis]].append(doc)
        result["by_" + axis] = {key: _score_subset(part, by_doc, threshold) for key, part in sorted(groups.items())}
    # Real test and synthetic test must not be pooled as the headline number.
    result["headline_real_test"] = _score_subset([d for d in docs if d["split"] == "test" and d["source"] in ("creddata", "reviewed_real")], by_doc, threshold)
    return result


def _percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def bootstrap_by_repo(docs, predictions, threshold: float, repeats: int = 500, seed: int = 404):
    """Percentile 95% CIs by resampling whole repositories with replacement.

    Per-repository sufficient statistics are computed once; no detector/model
    reruns occur. Confidence intervals describe the supplied fixed experiment,
    not uncertainty from training seeds, threshold search, or label noise.
    """
    if type(repeats) is not int or not 1 <= repeats <= 100000:
        raise ValueError("bootstrap repeats must be an integer between 1 and 100000")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a probability in [0,1]")
    docs, predictions = list(docs), list(predictions)
    if not docs:
        raise ValueError("Cannot bootstrap an empty document set")
    # Reuse strict prediction validation rather than accepting malformed spans.
    evaluate_documents(docs, predictions, threshold)
    groups, by_doc = defaultdict(list), defaultdict(list)
    for doc in docs:
        groups[doc["repo_id"]].append(doc)
    for pred in predictions:
        by_doc[pred["document_id"]].append(pred)
    stats = [_score_subset(part, by_doc, threshold) for _, part in sorted(groups.items())]
    rng = random.Random(seed)
    draws = {key: [] for key in ("precision", "recall", "f1", "unknown_alerts_per_kloc")}
    for _ in range(repeats):
        tp = fp = fn = unknown = lines = 0
        for _ in stats:
            row = stats[rng.randrange(len(stats))]
            tp += row["true_positive"]
            fp += row["false_positive"]
            fn += row["false_negative"]
            unknown += row["unknown_alerts"]
            lines += row["lines"]
        for key, value in prf(tp, fp, fn).items():
            draws[key].append(value)
        draws["unknown_alerts_per_kloc"].append(unknown * 1000 / lines if lines else 0.0)
    return {"method": "repository-cluster-percentile-bootstrap", "confidence": 0.95, "repeats": repeats,
            "seed": seed, "repositories": len(stats), "sampling_unit": "repo_id",
            "intervals": {key: {"lower": _percentile(values, 0.025), "upper": _percentile(values, 0.975)}
                          for key, values in draws.items()},
            "limitations": ["One or few repositories give weak uncertainty estimates.",
                            "Does not include training-seed, model-selection, threshold-selection, or annotation uncertainty."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, required=True, help="Frozen validation-selected threshold")
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "test_ood"))
    parser.add_argument("--bootstrap-samples", type=int, default=0, help="Optional whole-repository 95%% confidence intervals")
    parser.add_argument("--bootstrap-seed", type=int, default=404)
    args = parser.parse_args(argv)
    docs = list(read_jsonl(args.documents))
    manifest = validate_corpus(docs)
    preds = list(read_jsonl(args.predictions))
    if args.split:
        docs = [d for d in docs if d["split"] == args.split]
        ids = {d["id"] for d in docs}
        preds = [p for p in preds if p.get("document_id") in ids]
    result = evaluate_documents(docs, preds, args.threshold)
    if args.bootstrap_samples < 0:
        parser.error("--bootstrap-samples must be nonnegative")
    if args.bootstrap_samples:
        result["bootstrap"] = bootstrap_by_repo(docs, preds, args.threshold, args.bootstrap_samples, args.bootstrap_seed)
    result["corpus_sha256"] = manifest["corpus_sha256"]
    write_json(args.out, result)
    print(json.dumps({k: v for k, v in result["overall"].items() if k not in ("by_category", "missed_spans")}))


if __name__ == "__main__":
    main()
