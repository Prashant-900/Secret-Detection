"""Run offline detector baselines against the exact same held-out documents.

External tools must already be installed. Source files are written as inert text;
they are never imported or executed. Verification and update checks are disabled
for TruffleHog. Raw reports and snapshot text remain private local artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from .candidates import discover_candidates
from .evaluate import evaluate_documents
from .schema import (line_offsets, read_jsonl, safe_child, stable_id, validate_corpus,
                     write_json, write_jsonl)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BaselineError(RuntimeError):
    pass


def _private(path: Path, directory=False):
    try:
        path.chmod(0o700 if directory else 0o600)
    except OSError:
        # Windows mode bits do not provide POSIX ACL semantics; do not imply they do.
        if os.name != "nt":
            raise


@contextmanager
def _secure_open(path: Path, mode="w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    _private(path.parent, directory=True)
    fd = os.open(path, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
    _private(path)
    with os.fdopen(fd, mode, encoding=None if "b" in mode else "utf-8", newline=None if "b" in mode else "\n") as handle:
        yield handle


def _safe_leaf(logical_path: str):
    name = logical_path.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).rstrip(". ")
    if not name or name in (".", ".."):
        name = "source.txt"
    if name.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        name = "_" + name
    if len(name) > 100:
        suffix = Path(name).suffix[:16]
        name = name[:70] + "_" + stable_id(name)[:10] + suffix
    return name


def materialize_snapshot(docs: list[dict], directory: str | Path):
    """Return relative-path -> document mapping; document-controlled paths cannot escape."""
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _private(root, directory=True)
    mapping = {}
    for doc in docs:
        relative = stable_id(doc["id"]) + "/" + _safe_leaf(doc["path"])
        path = safe_child(root, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        _private(path.parent, directory=True)
        with _secure_open(path) as handle:
            handle.write(doc["text"])
        mapping[relative] = doc
    return mapping


def _lookup(file_path: str, mapping: dict[str, dict]):
    normalized = str(file_path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    matched = [(relative, doc) for relative, doc in mapping.items()
               if normalized == relative or normalized.endswith("/" + relative)]
    if len(matched) != 1:
        raise BaselineError("Tool report references an unmapped or ambiguous snapshot file")
    return matched[0][1]


def _column_to_char(line: str, column: int, *, end: bool, column_encoding="utf8_bytes") -> int:
    """Gitleaks columns are one-based inclusive; Go offsets count UTF-8 bytes."""
    if column < 1:
        raise BaselineError("Tool report has a nonpositive column")
    boundary = column if end else column - 1
    line = line.rstrip("\r\n")
    if column_encoding == "characters":
        if boundary > len(line):
            raise BaselineError("Tool report column lies outside source line")
        return boundary
    encoded = line.encode("utf-8")
    if boundary > len(encoded):
        raise BaselineError("Tool report byte column lies outside source line")
    try:
        return len(encoded[:boundary].decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise BaselineError("Tool report byte column bisects a Unicode character") from exc


def locate_lines(doc: dict, start_line: int, end_line: int | None = None,
                 start_column: int | None = None, end_column: int | None = None,
                 column_encoding="utf8_bytes") -> tuple[int, int, str]:
    """Convert metadata positions, never ground-truth/value searches, to offsets."""
    lines, offsets = line_offsets(doc["text"])
    end_line = end_line or start_line
    if not 1 <= start_line <= end_line <= len(lines):
        raise BaselineError("Tool report line lies outside source document")
    if start_column is not None and end_column is not None and start_column > 0 and end_column > 0:
        start = offsets[start_line - 1] + _column_to_char(lines[start_line - 1], start_column, end=False, column_encoding=column_encoding)
        end = offsets[end_line - 1] + _column_to_char(lines[end_line - 1], end_column, end=True, column_encoding=column_encoding)
        precision = "reported_columns"
    else:
        start = offsets[start_line - 1]
        end = offsets[end_line - 1] + len(lines[end_line - 1].rstrip("\r\n"))
        precision = "reported_line_range"
    if not 0 <= start < end <= len(doc["text"]):
        raise BaselineError("Tool report resolves to an empty or reversed span")
    return start, end, precision


def _deduplicate(predictions):
    unique = {}
    for row in predictions:
        key = (row["document_id"], row["start"], row["end"])
        if key not in unique:
            unique[key] = row
        else:
            unique[key]["score"] = max(unique[key]["score"], row["score"])
            existing = set(unique[key].setdefault("metadata", {}).get("detectors", []))
            existing.update(row.get("metadata", {}).get("detectors", []))
            unique[key]["metadata"]["detectors"] = sorted(existing)
    return [unique[key] for key in sorted(unique)]


def parse_gitleaks(rows: list[dict], mapping: dict[str, dict]):
    if rows is None:
        rows = []  # Some tool releases serialize a successful empty report as null.
    if not isinstance(rows, list):
        raise BaselineError("Gitleaks report must be a JSON list")
    output = []
    for row in rows:
        try:
            doc = _lookup(row["File"], mapping)
            start, end, precision = locate_lines(doc, int(row["StartLine"]), int(row.get("EndLine") or row["StartLine"]),
                                                int(row.get("StartColumn") or 0), int(row.get("EndColumn") or 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise BaselineError("Malformed Gitleaks finding metadata") from exc
        output.append({"document_id": doc["id"], "start": start, "end": end, "score": 1.0,
                       "metadata": {"detectors": [str(row.get("RuleID", "gitleaks"))], "localization": precision}})
    return _deduplicate(output)


def parse_trufflehog(lines, mapping: dict[str, dict]):
    output = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            fs = row["SourceMetadata"]["Data"]["Filesystem"]
            doc = _lookup(fs["file"], mapping)
            start, end, precision = locate_lines(doc, int(fs["line"]))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise BaselineError(f"Malformed TruffleHog finding metadata at output line {number}") from exc
        # TruffleHog filesystem metadata is line-granular. Do not search for Raw
        # inside a document: decoder outputs need not appear literally in source.
        output.append({"document_id": doc["id"], "start": start, "end": end, "score": 1.0,
                       "metadata": {"detectors": [str(row.get("DetectorName", "trufflehog"))],
                                    "localization": precision, "decoder": str(row.get("DecoderName", "unknown"))}})
    return _deduplicate(output)


def run_external(command: list[str], stdout_path: Path, stderr_path: Path,
                 cwd: Path, timeout: float, allowed_exit_codes=(0,)):
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    started = time.monotonic()
    try:
        with _secure_open(stdout_path, "wb") as stdout, _secure_open(stderr_path, "wb") as stderr:
            result = subprocess.run(command, cwd=str(cwd), shell=False, stdout=stdout, stderr=stderr,
                                    timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BaselineError(f"Detector exceeded {timeout}s; inspect its private stderr log") from exc
    except OSError as exc:
        raise BaselineError(f"Cannot start detector {Path(command[0]).name}; provide an executable for this OS") from exc
    if result.returncode not in allowed_exit_codes:
        raise BaselineError(f"Detector failed with exit {result.returncode}; inspect {stderr_path}")
    return {"seconds": time.monotonic() - started, "returncode": result.returncode}


def legacy_predictions(docs, scanner_path: str | Path):
    rows = []
    for doc in docs:
        for proposal in discover_candidates(doc, str(scanner_path)):
            score = proposal["metadata"].get("legacy_score")
            if score is None:
                continue
            rows.append({"document_id": doc["id"], "start": proposal["start"], "end": proposal["end"], "score": float(score),
                         "metadata": {"detectors": ["legacy_v2"], "localization": "candidate_value"}})
    model_path = Path(scanner_path).with_name("model_v2.json")
    if not model_path.is_file():
        raise BaselineError("Legacy model_v2.json is missing; baseline must use a known frozen model")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    thresholds = model.get("thresholds", {})
    if not all(key in thresholds for key in ("warn", "block")):
        raise BaselineError("Legacy model lacks warn/block thresholds")
    return _deduplicate(rows), {key: float(thresholds[key]) for key in ("warn", "block")}


def _save_predictions(path, rows):
    write_jsonl(path, rows)
    _private(Path(path))


def _digest_file(path):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _raw_provenance(path, report_path, *, command=None, timing=None, reuse=False):
    if reuse:
        if not path.exists():
            raise BaselineError("Raw cache lacks provenance metadata; run the detector once")
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("report_sha256") != _digest_file(report_path):
            raise BaselineError("Raw report changed since its successful detector run")
        return {**meta.get("timing", {}), "reused_raw": True, "binary_sha256": meta.get("binary_sha256")}
    meta = {"command": command, "binary_sha256": _digest_file(command[0]),
            "report_sha256": _digest_file(report_path), "timing": timing}
    write_json(path, meta)
    _private(path)
    return {**timing, "reused_raw": False, "binary_sha256": meta["binary_sha256"]}


def run_baselines(documents_path, output_dir, split="test", tools=("legacy", "gitleaks", "trufflehog"),
                  gitleaks=None, trufflehog=None, scanner=None, timeout=3600, reuse_raw=False):
    docs_all = list(read_jsonl(documents_path))
    manifest = validate_corpus(docs_all)
    docs = [doc for doc in docs_all if doc["split"] == split]
    if not docs:
        raise BaselineError(f"No documents in split {split}")
    allowed = {"legacy", "gitleaks", "trufflehog"}
    if not tools or set(tools) - allowed:
        raise ValueError("tools must be a nonempty subset of legacy,gitleaks,trufflehog")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _private(out, directory=True)
    snapshot, raw = out / "snapshot", out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    _private(raw, directory=True)
    run_manifest_path = out / "manifest.json"
    fingerprint = {"corpus_sha256": manifest["corpus_sha256"], "split": split,
                   "snapshot_documents": [{"id": doc["id"], "text_sha256": hashlib.sha256(doc["text"].encode()).hexdigest()} for doc in docs]}
    if run_manifest_path.exists():
        old = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != fingerprint:
            raise BaselineError("Output directory belongs to a different corpus/split; use a new output path")
    elif reuse_raw:
        raise BaselineError("--reuse-raw requires an existing matching manifest")
    mapping = materialize_snapshot(docs, snapshot)
    base_manifest = {"fingerprint": fingerprint, "protocol": "same-document-character-span-overlap-one-to-one",
                     "source_files_executed": False, "snapshot_paths": {k: d["id"] for k, d in mapping.items()},
                     "raw_contains_sensitive_text": True, "posix_permissions_enforced": os.name != "nt"}
    write_json(run_manifest_path, base_manifest)
    _private(run_manifest_path)
    summary = {"split": split, "documents": len(docs), "corpus_sha256": manifest["corpus_sha256"], "tools": {}}
    for tool in tools:
        try:
            timing = {"seconds": 0.0, "reused_raw": bool(reuse_raw)}
            if tool == "legacy":
                scanner_path = Path(scanner or PROJECT_ROOT / "Scanner/scanner/scan.py").resolve()
                started = time.monotonic()
                predictions, thresholds = legacy_predictions(docs, scanner_path)
                timing = {"seconds": time.monotonic() - started, "reused_raw": False,
                          "model_sha256": hashlib.sha256(scanner_path.with_name("model_v2.json").read_bytes()).hexdigest()}
            elif tool == "gitleaks":
                binary = str(Path(gitleaks or PROJECT_ROOT / "Scanner/tools/bin/gitleaks").resolve())
                report_path = raw / "gitleaks.json"
                if not reuse_raw:
                    with _secure_open(report_path) as handle:
                        handle.write("")
                    command = [binary, "dir", str(snapshot), "--no-banner", "--exit-code", "0", "--report-format", "json", "--report-path", str(report_path)]
                    timing = run_external(command, raw / "gitleaks.stdout.log", raw / "gitleaks.stderr.log", snapshot, timeout)
                if not report_path.exists():
                    raise BaselineError("Gitleaks did not produce its report")
                _private(report_path)
                predictions = parse_gitleaks(json.loads(report_path.read_text(encoding="utf-8")), mapping)
                timing = _raw_provenance(raw / "gitleaks.metadata.json", report_path, command=command if not reuse_raw else None,
                                         timing=timing, reuse=reuse_raw)
                thresholds = {"binary": 0.5}
            else:
                binary = str(Path(trufflehog or PROJECT_ROOT / "Scanner/tools/bin/trufflehog").resolve())
                report_path = raw / "trufflehog.jsonl"
                if not reuse_raw:
                    command = [binary, "filesystem", str(snapshot), "--json", "--no-verification", "--no-update"]
                    timing = run_external(command, report_path, raw / "trufflehog.stderr.log", snapshot, timeout)
                if not report_path.exists():
                    raise BaselineError("TruffleHog did not produce its report")
                with report_path.open(encoding="utf-8") as handle:
                    predictions = parse_trufflehog(handle, mapping)
                timing = _raw_provenance(raw / "trufflehog.metadata.json", report_path, command=command if not reuse_raw else None,
                                         timing=timing, reuse=reuse_raw)
                thresholds = {"binary": 0.5}
            _save_predictions(out / f"{tool}.predictions.jsonl", predictions)
            tool_result = {"status": "complete", "predictions": len(predictions), **timing, "operating_points": {}}
            for name, threshold in thresholds.items():
                result = evaluate_documents(docs, predictions, threshold)
                result.update({"tool": tool, "operating_point": name, "corpus_sha256": manifest["corpus_sha256"],
                               "localization_note": "TruffleHog is line-localized; Gitleaks uses reported match columns. Localization granularity can affect overlap scores.",
                               "binary_coverage_note": "Binary tools expose findings, not internal proposals; their reported candidate ceiling equals observed recall.",
                               "threshold_tuned_on_selected_split": False})
                if tool == "legacy":
                    result["legacy_training_overlap_caveat"] = (
                        "The shipped legacy ranker was trained on its original CredData dev split. "
                        "That split differs from this pipeline's frozen owner split, so legacy-model "
                        "results may include its training repositories. Treat it as the existing "
                        "deployed-model baseline, not clean held-out evidence for that model.")
                    result["legacy_model_clean_heldout_claim"] = False
                write_json(out / f"{tool}.{name}.evaluation.json", result)
                tool_result["operating_points"][name] = {key: result["overall"][key] for key in
                    ("precision", "recall", "f1", "candidate_recall_ceiling", "unknown_alerts", "negative_flag_rate")}
            summary["tools"][tool] = tool_result
        except (BaselineError, OSError, ValueError, json.JSONDecodeError) as exc:
            summary["tools"][tool] = {"status": "failed", "error": str(exc)}
            write_json(out / "summary.json", summary)
            raise BaselineError(f"{tool} baseline failed: {exc}") from exc
    write_json(out / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--split", choices=("val", "test", "test_ood"), required=True)
    parser.add_argument("--tools", default="legacy,gitleaks,trufflehog")
    parser.add_argument("--gitleaks")
    parser.add_argument("--trufflehog")
    parser.add_argument("--scanner")
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--reuse-raw", action="store_true", help="Re-score matching raw reports without running external tools")
    args = parser.parse_args(argv)
    result = run_baselines(args.documents, args.out, args.split, tuple(x.strip() for x in args.tools.split(",") if x.strip()),
                           args.gitleaks, args.trufflehog, args.scanner, args.timeout, args.reuse_raw)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
