"""Build a training-only hard-example/review queue without assigning new labels."""
import argparse
from collections import Counter
import json
from pathlib import Path

from .schema import read_jsonl, write_jsonl, write_json


def mine_training_errors(candidates, predictions, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0,1]")
    by_id = {r["id"]: r for r in candidates if r["split"] == "train"}
    output = []
    for pred in predictions:
        row = by_id.get(pred["id"])
        if row is None:
            if pred.get("split") != "train":
                raise ValueError("Mining accepts training predictions only; held-out errors cannot guide generation")
            raise ValueError("Prediction is absent from training candidates")
        score = float(pred["score"])
        if not 0 <= score <= 1:
            raise ValueError("Invalid probability")
        label = row["label"]
        if label == 0 and score >= threshold:
            reason = "explicit_hard_negative"
        elif label == 1 and score < threshold:
            reason = "missed_labeled_positive"
        elif label is None and score >= threshold:
            reason = "unknown_requires_human_review"
        else:
            continue
        output.append({k: row[k] for k in ("id", "document_id", "repo_id", "split", "source", "path", "start", "end", "label", "category")})
        output[-1].update(score=score, reason=reason, human_verdict=None)
    return sorted(output, key=lambda r: (r["reason"], -r["score"], r["id"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    a = ap.parse_args(argv)
    rows = mine_training_errors(read_jsonl(a.data), read_jsonl(a.predictions), a.threshold)
    write_jsonl(a.out, rows)
    report = {"queue_counts": dict(Counter(r["reason"] for r in rows)), "new_labels_assigned": 0,
              "scope": "train_only", "raw_values_exported": False}
    write_json(Path(a.out).with_suffix(".report.json"), report)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
