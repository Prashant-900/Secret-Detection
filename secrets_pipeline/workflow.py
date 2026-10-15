"""Import, audit, generate, merge and prepare one frozen experimental corpus."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import ingest
from .audit import audit_documents
from .candidates import candidate_coverage, prepare_records
from .generate import generate_dataset
from .schema import read_jsonl, validate_corpus, write_json, write_jsonl


def _build_dataset(*, out, generation_config, tier_b=(), creddata=(), documents=(),
                  skip_invalid=False, mock=False, audit_quotas=False,
                  target_per_cell=100, max_cells=64, max_per_cell=100,
                  legacy_scanner="auto", repo_map=None, seed=404):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    base = out / "base.documents.jsonl"
    registry = out / "frozen_splits.json"
    argv = ["--out", str(base), "--registry", str(registry), "--seed", str(seed)]
    for flag, paths in (("--tier-b", tier_b), ("--creddata", creddata), ("--documents", documents)):
        for path in paths:
            argv += [flag, str(path)]
    if skip_invalid:
        argv.append("--skip-invalid")
    if repo_map:
        argv += ["--repo-map", str(repo_map)]
    ingest.main(argv)
    base_docs = list(read_jsonl(base))
    coverage = audit_documents(base_docs, target_per_cell, max_cells, max_per_cell)
    write_json(out / "coverage.before.json", coverage)
    config = dict(generation_config)
    if "generation" in config:
        config = dict(config["generation"])
    if audit_quotas:
        fragment = coverage["generation_config_fragment"]
        if not fragment["cells"]:
            raise ValueError("No deficient cells under these audit settings. Use a fresh explicit generation plan if needed.")
        config.update(fragment)
    # Exclude source templates without sending source text to the LLM.
    config["exclude_documents"] = sorted(set(config.get("exclude_documents", []) + [str(base.resolve())]))
    write_json(out / "generation.effective.json", config)
    generation = generate_dataset(config, out / "generation", mock=mock)
    if generation["status"] != "complete":
        write_json(out / "dataset_status.json", {"status": "generation_needs_review", "generation": generation})
        raise ValueError("Generation rejection rate exceeded the configured gate. Inspect generation/report.json before scaling.")
    merged = out / "documents.jsonl"
    merge_args = ["--documents", str(base), "--documents", str(out / "generation" / "documents.jsonl"),
                  "--out", str(merged), "--registry", str(registry), "--seed", str(seed)]
    if skip_invalid:
        merge_args.append("--skip-invalid")
    ingest.main(merge_args)
    corpus = list(read_jsonl(merged))
    manifest = validate_corpus(corpus)
    records = prepare_records(corpus, legacy_scanner=legacy_scanner)
    write_jsonl(out / "candidates.jsonl", records)
    ceiling = candidate_coverage(corpus, records)
    write_json(out / "candidate_coverage.json", ceiling)
    write_json(out / "coverage.after.json", audit_documents(corpus))
    # Separate data views make real-only versus augmentation comparisons repeatable.
    real = [row for row in records if row["source"] in ("creddata", "reviewed_real")]
    write_jsonl(out / "real.candidates.jsonl", real)
    status = {"status": "prepared", **manifest, "candidate_records": len(records),
              "candidate_recall_ceiling": ceiling["candidate_recall_ceiling"],
              "real_candidate_records": len(real), "generation_mock": generation["mock_data"],
              "real_validation_required_for_training": True, "heldout_used_for_generation_quotas": False,
              "documents": str(merged.resolve()), "candidates": str((out / "candidates.jsonl").resolve())}
    write_json(out / "dataset_status.json", status)
    return status


def build_dataset(**kwargs):
    """A failed/rejected rerun must never leave a previous success marker."""
    out = Path(kwargs["out"])
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dataset_status.json", {"status": "building", "usable_for_training": False})
    try:
        return _build_dataset(**kwargs)
    except (Exception, SystemExit) as exc:
        write_json(out / "dataset_status.json", {"status": "failed", "usable_for_training": False,
                   "error_type": type(exc).__name__, "message": str(exc),
                   "instruction": "Inspect stage reports. Partial or older output files are not a completed new dataset."})
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generation-config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier-b", action="append", default=[])
    ap.add_argument("--creddata", action="append", default=[])
    ap.add_argument("--documents", action="append", default=[])
    ap.add_argument("--repo-map")
    ap.add_argument("--skip-invalid", action="store_true", help="Explicitly quarantine invalid source files; reports retain every reason")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--audit-quotas", action="store_true", help="Replace configured cells using only observed training deficits")
    ap.add_argument("--target-per-cell", type=int, default=100)
    ap.add_argument("--max-cells", type=int, default=64)
    ap.add_argument("--max-per-cell", type=int, default=100)
    ap.add_argument("--legacy-scanner", default="auto")
    ap.add_argument("--seed", type=int, default=404)
    args = vars(ap.parse_args(argv))
    args["generation_config"] = json.loads(Path(args["generation_config"]).read_text(encoding="utf-8"))
    print(json.dumps(build_dataset(**args), indent=2))


if __name__ == "__main__":
    main()
