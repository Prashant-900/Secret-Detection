"""Evaluate a frozen selected model on a named split, without fitting/tuning."""
import argparse
import hashlib
import json
from pathlib import Path
from .schema import read_jsonl, validate_corpus, write_json, write_jsonl
from .evaluate import evaluate_documents
from .scan import read_detector, score_records, validate_thresholds
from .experiments import file_sha256


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--documents", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", required=True, choices=("val", "test", "test_ood"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--bootstrap-samples", type=int, default=500)
    ap.add_argument("--external-corpus", action="store_true", help="Explicitly evaluate a new independent corpus using frozen model/thresholds")
    a = ap.parse_args(argv)
    selected = json.loads(Path(a.bundle).read_text(encoding="utf-8"))
    if not a.external_corpus:
        if file_sha256(a.documents) != selected.get("documents_sha256") or file_sha256(a.data) != selected.get("input_sha256"):
            raise ValueError("Benchmark corpus/candidates differ from model selection. Use the frozen files, or explicitly --external-corpus for a new corpus.")
    docs = list(read_jsonl(a.documents))
    manifest = validate_corpus(docs)
    docs = [d for d in docs if d["split"] == a.split]
    if not docs:
        raise ValueError("Selected split has no documents")
    ids = {d["id"] for d in docs}
    records = [r for r in read_jsonl(a.data) if r["document_id"] in ids]
    if any(r["split"] != a.split for r in records):
        raise ValueError("Candidate/document split mismatch")
    by_id = {d["id"]: d for d in docs}
    for row in records:
        doc = by_id[row["document_id"]]
        if row.get("metadata", {}).get("document_text_sha256") != hashlib.sha256(doc["text"].encode()).hexdigest():
            raise ValueError("Candidate text provenance does not match evaluation document")
        if any(row[key] != doc[key] for key in ("repo_id", "group_id", "source", "path", "language")):
            raise ValueError("Candidate provenance does not match evaluation document")
    detector = read_detector(bundle=a.bundle)
    predictions = score_records(records, detector, batch_size=a.batch_size)
    warn, block = validate_thresholds(detector["thresholds"])
    out = Path(a.out)
    write_jsonl(out / "predictions.jsonl", predictions)
    result = {"split": a.split, "corpus_sha256": manifest["corpus_sha256"], "thresholds_tuned": False,
              "external_corpus": a.external_corpus,
              "warn": evaluate_documents(docs, predictions, warn),
              "block": evaluate_documents(docs, predictions, block) if block is not None else None,
              "thresholds": detector["thresholds"]}
    if a.bootstrap_samples:
        from .evaluate import bootstrap_by_repo
        # The primary interval excludes generated/modified examples.
        real = [d for d in docs if d["source"] in ("creddata", "reviewed_real")]
        real_ids = {d["id"] for d in real}
        if real:
            result["real_repository_bootstrap"] = bootstrap_by_repo(real, [p for p in predictions if p["document_id"] in real_ids], warn, repeats=a.bootstrap_samples)
    write_json(out / "metrics.json", result)
    headline = result["warn"]["headline_real_test"] if a.split == "test" else result["warn"]["overall"]
    print(json.dumps({"split": a.split, "scope": "real_repository_test" if a.split == "test" else "selected_split",
                      "documents": headline["documents"], "warn_f1": headline["f1"],
                      "warn_recall": headline["recall"], "out": str(out)}))


if __name__ == "__main__":
    main()
