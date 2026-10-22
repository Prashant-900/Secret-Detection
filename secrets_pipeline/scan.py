"""Deploy the same candidate extraction and saved models used in evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import time

from .candidates import prepare_records, preprocessing_config
from .schema import language_for, stable_id, write_json


def read_detector(*, bundle=None, model_dir=None, legacy=False):
    if sum((bool(bundle), bool(model_dir), bool(legacy))) != 1:
        raise ValueError("Select exactly one of bundle, model_dir, or legacy")
    if legacy:
        path = Path(__file__).resolve().parents[1] / "Scanner/scanner/model_v2.json"
        return {"models": [], "thresholds": json.loads(path.read_text())["thresholds"], "kind": "legacy"}
    if bundle:
        path = Path(bundle).resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("test_used_for_selection") is not False or not data.get("models"):
            raise ValueError("Expected a completed validation-only selected.json")
        models = []
        for member in data["models"]:
            model_path = (path.parent / member["path"]).resolve()
            if member.get("sha256"):
                from .experiments import model_sha256
                if model_sha256(model_path) != member["sha256"]:
                    raise ValueError("Selected model files differ from the validated bundle")
            weight = float(member["weight"])
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("Ensemble weights must be positive and finite")
            models.append({"path": model_path, "weight": weight})
        total = sum(m["weight"] for m in models)
        for member in models:
            member["weight"] /= total
        return {"models": models, "thresholds": data["thresholds"], "kind": "encoder_bundle"}
    from .predict import resolve_model_dir
    path = resolve_model_dir(model_dir)
    return {"models": [{"path": path, "weight": 1.0}],
            "thresholds": json.loads((path / "thresholds.json").read_text(encoding="utf-8")), "kind": "encoder"}


def validate_thresholds(thresholds):
    warn, block = thresholds.get("warn"), thresholds.get("block")
    if not isinstance(warn, (float, int)) or not math.isfinite(warn) or not 0 <= warn <= 1:
        raise ValueError("Missing or invalid saved WARN threshold")
    if block is not None and (not isinstance(block, (int, float)) or not math.isfinite(block) or not warn <= block <= 1):
        raise ValueError("Invalid saved BLOCK threshold")
    return float(warn), float(block) if block is not None else None


def detector_preprocessing(detector):
    if detector["kind"] == "legacy":
        return preprocessing_config()
    settings = []
    for member in detector["models"]:
        path = Path(member["path"]) / "inference_config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        saved = config.get("preprocessing")
        if not saved:
            raise ValueError("Saved model has no candidate preprocessing provenance; re-export using this pipeline")
        current = preprocessing_config(saved["context_lines"], saved["max_chars"], saved["legacy_scanner"])
        if current != saved:
            raise ValueError("Candidate extraction code or scanner assets changed since training; restore the validated version")
        settings.append(saved)
    if any(s != settings[0] for s in settings):
        raise ValueError("Ensemble members use different candidate preprocessing")
    return settings[0]


def score_records(records, detector, *, batch_size=32, precision="fp32", device=None):
    """No ad-hoc provider floor: deployed scores match calibrated validation scores."""
    records = list(records)
    if detector["kind"] == "legacy":
        from .train import redact_prediction
        return [redact_prediction(r, r.get("metadata", {}).get("legacy_score", 0.0)) for r in records]
    if precision != "fp32":
        raise ValueError("Saved operating points were calibrated in fp32; use fp32 for consistent CI gating")
    expected = detector_preprocessing(detector)
    if any(r.get("metadata", {}).get("preprocessing") != expected for r in records):
        raise ValueError("Candidate preprocessing differs from the calibrated model")
    from .predict import predict_records
    result = None
    for member in detector["models"]:
        preds = predict_records(member["path"], records, batch_size=batch_size, precision=precision, device=device)
        if result is None:
            result = [{**p, "score": p["score"] * member["weight"]} for p in preds]
        else:
            if [p["id"] for p in preds] != [p["id"] for p in result]:
                raise ValueError("Ensemble members returned different candidate populations")
            for out, pred in zip(result, preds):
                out["score"] += member["weight"] * pred["score"]
    return result or []


def read_target(target, *, all_files=False, max_file_bytes=2_000_000):
    root = Path(target).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    files, skipped, errors = [], [], []
    base = root if root.is_dir() else root.parent
    excluded = {".git", "node_modules", ".venv", ".venv-train", ".venv-generator", ".runtime", "__pycache__"}
    if max_file_bytes < 1:
        raise ValueError("max_file_bytes must be positive")
    if root.is_dir():
        paths = []
        for directory, subdirs, names in os.walk(root, followlinks=False):
            subdirs[:] = sorted(d for d in subdirs if d not in excluded and not d.startswith(".venv"))
            paths.extend(Path(directory) / name for name in sorted(names))
    else:
        paths = [root]
    for path in paths:
        if not path.is_file() or any(part in excluded for part in path.relative_to(base).parts[:-1]):
            continue
        rel = path.relative_to(base).as_posix()
        if path.is_symlink():
            skipped.append({"path": rel, "reason": "symlink"})
            continue
        language = language_for(rel)
        if not all_files and language == "text" and root.is_dir():
            skipped.append({"path": rel, "reason": "unsupported_extension"})
            continue
        try:
            if path.stat().st_size > max_file_bytes:
                errors.append({"path": rel, "reason": "file_size_limit"})
                continue
            raw = path.read_bytes()
            if b"\x00" in raw:
                skipped.append({"path": rel, "reason": "binary"})
                continue
            text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        except (OSError, UnicodeError) as exc:
            errors.append({"path": rel, "reason": type(exc).__name__})
            continue
        files.append({"id": stable_id("scan", rel), "repo_id": "deployment", "group_id": "deployment",
                      "split": "test", "source": "deployment", "path": rel, "language": language,
                      "text": text, "spans": [], "metadata": {}})
    return files, skipped, errors


def findings_from_predictions(documents, predictions, thresholds):
    warn, block = validate_thresholds(thresholds)
    docs = {d["id"]: d for d in documents}
    findings = []
    for row in predictions:
        score = row["score"]
        if score < warn:
            continue
        text = docs[row["document_id"]]["text"]
        start, end = row["start"], row["end"]
        findings.append({"file": row["path"], "line": text.count("\n", 0, start) + 1,
                         "column": start - text.rfind("\n", 0, start), "start": start, "end": end,
                         "score": round(score, 6), "verdict": "BLOCK" if block is not None and score >= block else "WARN",
                         "match": "[REDACTED]"})
    return sorted(findings, key=lambda r: (r["file"], r["start"], -r["score"]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("target")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle", help="Validation-selected ensemble/single selected.json")
    mode.add_argument("--model-dir", help="Completed encoder run or best/ directory")
    mode.add_argument("--legacy", action="store_true", help="Run existing gradient-boosted scanner through the shared candidate pass")
    ap.add_argument("--out", required=True)
    ap.add_argument("--all-files", action="store_true")
    ap.add_argument("--max-file-bytes", type=int, default=2_000_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default="fp32")
    ap.add_argument("--device")
    a = ap.parse_args(argv)
    began = time.monotonic()
    detector = read_detector(bundle=a.bundle, model_dir=a.model_dir, legacy=a.legacy)
    docs, skipped, errors = read_target(a.target, all_files=a.all_files, max_file_bytes=a.max_file_bytes)
    prep = detector_preprocessing(detector)
    records = prepare_records(docs, context_lines=prep["context_lines"], max_chars=prep["max_chars"], legacy_scanner=prep["legacy_scanner"])
    predictions = score_records(records, detector, batch_size=a.batch_size, precision=a.precision, device=a.device)
    findings = findings_from_predictions(docs, predictions, detector["thresholds"])
    counts = Counter(f["verdict"] for f in findings)
    status = "ERROR" if errors or not docs else "BLOCK" if counts["BLOCK"] else "WARN" if counts["WARN"] else "PASS"
    report = {"status": status, "files_scanned": len(docs), "candidates": len(records), "findings": findings,
              "thresholds": detector["thresholds"], "detector": detector["kind"], "skipped": skipped,
              "errors": errors, "seconds": round(time.monotonic() - began, 3), "raw_secret_values_exported": False}
    write_json(a.out, report)
    print(json.dumps({"status": status, "files_scanned": len(docs), "BLOCK": counts["BLOCK"], "WARN": counts["WARN"],
                      "errors": len(errors), "out": a.out}))
    return 2 if status == "ERROR" else 1 if status == "BLOCK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
