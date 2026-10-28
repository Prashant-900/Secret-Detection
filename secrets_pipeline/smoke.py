"""Exercise data generation through evaluation on fictional fixtures, offline."""
import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

from .schema import stable_id, write_jsonl, write_json, read_jsonl
from .workflow import build_dataset
from .train import train_model
from .evaluate import evaluate_documents


def fixture_documents():
    result = []
    for i, split in enumerate(("train", "val", "test", "test_ood")):
        values = {"password": f"fictional-{i}-value", "resource_id": f"resource-{i}-000"}
        text = json.dumps({"service": f"unique-context-{i}", **values}, indent=2)
        spans = []
        for key, label in (("password", 1), ("resource_id", 0)):
            start = text.index(values[key])
            spans.append({"id": stable_id(split, key), "start": start, "end": start + len(values[key]),
                          "label": label, "category": "password" if label else "resource_identifier",
                          "label_origin": "fictional_smoke_fixture"})
        result.append({"id": "smoke-" + split, "repo_id": "smoke-" + split,
                       "group_id": "smoke-" + split, "split": split, "source": "smoke_fixture",
                       "path": f"service{i}.json", "language": "json", "text": text, "spans": spans, "metadata": {}})
    return result


def run_smoke(out):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Smoke output must be fresh; choose a new directory")
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "fixtures.jsonl", fixture_documents())
    config = {"seed": 404, "families": ["json_config", "toml", "python"],
              "split_ratios": {"train": 1, "val": 0, "test": 0}, "include_splits": ["train"],
              "workers": 2, "require_parsed": True, "max_reject_rate": 0.8,
              "cells": [{"family": "json_config", "secret_type": "password_low_entropy", "count": 2},
                        {"family": "toml", "secret_type": "opaque_token", "count": 2},
                        {"family": "python", "secret_type": "connection_string", "count": 2}]}
    prepared = build_dataset(out=out / "dataset", generation_config=config,
                             documents=[out / "fixtures.jsonl"], mock=True)
    data = out / "dataset/candidates.jsonl"
    dry = train_model({"allow_synthetic_validation": True, "precision": "fp32", "device": "cpu"},
                      data, out / "not-trained", dry_run=True)
    docs = list(read_jsonl(out / "dataset/documents.jsonl"))
    records = list(read_jsonl(data))
    # Exercise evaluation plumbing with the actual existing feature model scores.
    predictions = [{"document_id": r["document_id"], "start": r["start"], "end": r["end"],
                    "score": r["metadata"].get("legacy_score", 0.0)} for r in records]
    metrics = evaluate_documents(docs, predictions, threshold=0.45)
    write_json(out / "fixture_metrics.json", metrics)
    result = {"status": "passed", "mock_only": True, "model_trained": False,
              "documents": len(docs), "candidates": len(records),
              "training_dry_run": dry, "heldout_used_for_selection": False,
              "note": "Fixture metrics validate plumbing only; they are not model quality evidence."}
    write_json(out / "smoke_report.json", result)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", help="Fresh output directory; temporary directory when omitted")
    ap.add_argument("--train", action="store_true", help="Also execute tiny local CPU BERT framework smoke (requires ML environment)")
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="secrets-smoke-") as temporary:
        out = Path(a.out) if a.out else Path(temporary) / "run"
        result = run_smoke(out)
        if a.train:
            from .smoke_train import run_training_smoke
            result["framework_smoke"] = run_training_smoke(out / "tiny_training")
        print(json.dumps({"status": result["status"], "documents": result["documents"],
                          "candidates": result["candidates"], "model_trained": bool(a.train),
                          "benchmark_quality_claim": False, "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
