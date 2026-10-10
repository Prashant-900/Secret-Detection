"""Label-independent candidate discovery shared by training and deployment.

All supervised examples are proposed by this function before annotations are
consulted. A missed positive stays missed and reduces end-to-end recall.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .schema import line_offsets, read_jsonl, stable_id, validate_corpus, write_json, write_jsonl

_QUOTED = re.compile(r'''(["'`])((?:\\.|(?!\1)[^\\\r\n]){2,4096}?)\1''')
_ASSIGN = re.compile(r'''(?:[\w.\-]+|["'][^"'\n]+["'])\s*(?:=|:)\s*(?P<value>[^\r\n]+)''')
_AUTH = re.compile(r"\b(?:Bearer|Basic)\s+([A-Za-z0-9_+/=.%~\-]{3,})", re.I)
_URL = re.compile(r"[a-z][a-z0-9+.-]*://[^\s:@/]+:([^\s@/]+)@", re.I)
_TOKEN = re.compile(r"[A-Za-z0-9_+/=.~!@#$%\-]{12,4096}")
_PEM = re.compile(r"-----BEGIN ([A-Z0-9 ]*PRIVATE KEY)-----[\s\S]*?-----END \1-----")


def _source_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def preprocessing_config(context_lines: int = 3, max_chars: int = 6000, legacy_scanner: str | None = "auto") -> dict:
    """Serializable settings/fingerprints binding training to deployment code."""
    if context_lines < 0 or max_chars < 256:
        raise ValueError("context_lines must be nonnegative and max_chars >= 256")
    mode = "none" if legacy_scanner in (None, "none") else str(legacy_scanner)
    legacy_path = None
    if mode == "auto":
        legacy_path = Path(__file__).resolve().parents[1] / "Scanner/scanner/scan.py"
    elif mode != "none":
        legacy_path = Path(mode).resolve()
        mode = str(legacy_path)
    if legacy_path is not None and not legacy_path.is_file():
        raise FileNotFoundError(f"Legacy scanner missing at {legacy_path}")
    return {"version": 1, "context_lines": context_lines, "max_chars": max_chars, "legacy_scanner": mode,
            "code_sha256": {"candidates": _source_hash(__file__),
                            "schema": _source_hash(Path(__file__).with_name("schema.py")),
                            "legacy_scan": _source_hash(legacy_path) if legacy_path else None,
                            "rules": _source_hash(legacy_path.with_name("rules.toml")) if legacy_path else None,
                            "model": _source_hash(legacy_path.with_name("model_v2.json")) if legacy_path else None}}


@lru_cache(maxsize=4)
def _legacy(path: str):
    path = str(Path(path).resolve())
    spec = importlib.util.spec_from_file_location("_secrets_pipeline_legacy_" + stable_id(path), path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot import scanner at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rules, ctx, keywords, no_keywords = module.load_rules(str(Path(path).with_name("rules.toml")))
    return module, ctx, keywords, no_keywords


def _entropy(value: str) -> float:
    return -sum((n / len(value)) * math.log2(n / len(value)) for n in Counter(value).values()) if value else 0.0


def discover_candidates(document: dict, legacy_scanner: str | None = "auto") -> list[dict]:
    """Return spans derived exclusively from document text, never annotations.

    This broad candidate pass deliberately retains low-entropy literals and
    harmless references so the model can learn to distinguish them. Supplying a
    legacy scanner adds its provider rules and scores without limiting coverage.
    """
    if legacy_scanner == "auto":
        legacy_scanner = str(Path(__file__).resolve().parents[1] / "Scanner" / "scanner" / "scan.py")
        if not Path(legacy_scanner).is_file():
            raise FileNotFoundError("Default legacy scanner is missing; provide --legacy-scanner PATH or explicitly 'none'")
    elif legacy_scanner == "none":
        legacy_scanner = None
    text = document["text"]
    candidates = {}

    def add(start, end, reason, legacy_score=None, provider_floor=None):
        if not 0 <= start < end <= len(text):
            return
        if not text[start:end].strip():
            return
        item = candidates.setdefault((start, end), {"start": start, "end": end, "metadata": {"detector_sources": []}})
        if reason not in item["metadata"]["detector_sources"]:
            item["metadata"]["detector_sources"].append(reason)
        if legacy_score is not None:
            item["metadata"]["legacy_score"] = max(legacy_score, item["metadata"].get("legacy_score", 0.0))
        if provider_floor is not None:
            item["metadata"]["provider_floor"] = max(provider_floor, item["metadata"].get("provider_floor", 0.0))

    pem_intervals = []
    for match in _PEM.finditer(text):
        add(*match.span(), "pem_block")
        pem_intervals.append(match.span())
    lines, starts = line_offsets(text)
    for number, line in enumerate(lines):
        offset = starts[number]
        # A PEM key is one semantic candidate; do not manufacture 20 body hits.
        if any(a <= offset < b for a, b in pem_intervals):
            continue
        quoted_intervals = []
        for match in _QUOTED.finditer(line):
            a, b = match.span(2)
            add(offset + a, offset + b, "quoted_literal")
            quoted_intervals.append((a, b))
        for match in _ASSIGN.finditer(line):
            raw = match.group("value")
            prefix = len(raw) - len(raw.lstrip())
            value = raw.strip().rstrip(",;").rstrip()
            # Quoted strings and containers are handled independently. Avoid a
            # large assignment candidate which swallows multiple different labels.
            if value and value[0] not in "\"'`[{" and not re.search(r"\s(?:#|//)", value):
                value = value.rstrip("}")
                if value and len(value) <= 4096:
                    add(offset + match.start("value") + prefix, offset + match.start("value") + prefix + len(value), "assignment")
        for pattern, reason in ((_AUTH, "authorization"), (_URL, "url_password")):
            for match in pattern.finditer(line):
                add(offset + match.start(1), offset + match.end(1), reason)
        for match in _TOKEN.finditer(line):
            value = match.group().strip(".=")
            if len(value) < 12 or _entropy(value) < 3.0:
                continue
            a = match.start() + len(match.group()) - len(match.group().lstrip(".="))
            # Retain one literal rather than many substrings of an already quoted
            # candidate. This also keeps one-to-one span evaluation interpretable.
            if any(x <= a and a + len(value) <= y for x, y in quoted_intervals):
                continue
            add(offset + a, offset + a + len(value), "broad_token")
    if legacy_scanner:
        module, ctx, keywords, no_keywords = _legacy(str(legacy_scanner))
        ext = module.ext_of(document["path"])
        for i, line in enumerate(lines):
            if len(line) > 2000:
                continue
            for candidate, features in module.candidate_features(line, ext, ctx, keywords, no_keywords,
                                                                 lines[i - 1] if i else ""):
                start, end = starts[i] + candidate["s"], starts[i] + candidate["e"]
                parent = next(((a, b) for a, b in pem_intervals if a <= start < b), None)
                if parent:
                    start, end = parent
                add(start, end, "legacy", module.score_candidate(candidate, features, ctx),
                    candidate["conf"] if candidate["specific"] else None)
    return [candidates[key] for key in sorted(candidates)]


def build_input(document: dict, candidate: dict, context_lines: int = 3, max_chars: int = 6000) -> str:
    """Put the target first, before any potentially truncated context.

    Group, source, split, labels, category, synthetic variant and secret_type are
    never model inputs. The file basename conveys format only.
    """
    if context_lines < 0 or max_chars < 256:
        raise ValueError("context_lines must be nonnegative and max_chars >= 256")
    text = document["text"]
    start, end = candidate["start"], candidate["end"]
    lo = start
    for _ in range(context_lines + 1):
        position = text.rfind("\n", 0, lo)
        if position < 0:
            lo = 0
            break
        lo = position
    hi = end
    for _ in range(context_lines + 1):
        position = text.find("\n", hi + 1)
        if position < 0:
            hi = len(text)
            break
        hi = position
    literal = text[start:end]
    # Keep beginning AND end for PEM/multiline tokens when character-limited.
    budget = max_chars // 2
    if len(literal) > budget:
        literal = literal[:budget // 2] + "\n[truncated]\n" + literal[-budget // 2:]
    header = f"<candidate>\n{literal}\n</candidate>\n<file>{Path(document['path']).name}</file>\n<language>{document['language']}</language>\n"
    before, after = text[lo:start], text[end:hi]
    remaining = max(0, max_chars - len(header) - 40)
    half = remaining // 2
    return header + "<context>\n" + before[-half:] + " [CANDIDATE] " + after[:half] + "\n</context>"


def _overlap(a, b):
    return a["start"] < b["end"] and a["end"] > b["start"]


def supervise_candidates(document: dict, proposals: list[dict]) -> list[dict]:
    """Attach only unambiguous explicit supervision after proposal generation.

    CredData sometimes annotates an entire line/range without value columns.
    If such a positive overlaps several proposals, the exact credential is
    unknown: do not turn the username, URL host, and password all into positives.
    The original positive stays in document.spans for end-to-end evaluation.
    """
    ambiguous = set()
    for span in document["spans"]:
        if span["label"] == 1 and span.get("metadata", {}).get("annotation_precision") == "line_range":
            if sum(_overlap(span, proposal) for proposal in proposals) > 1:
                ambiguous.add(span["id"])
    output = []
    for proposal in proposals:
        matched = [s for s in document["spans"] if _overlap(proposal, s)]
        precise = [s for s in matched if s["id"] not in ambiguous]
        labels = {s["label"] for s in precise}
        has_ambiguous = any(s["id"] in ambiguous for s in matched)
        label = next(iter(labels)) if len(labels) == 1 else None
        # A coarse positive cannot certify a competing overlapping negative.
        if has_ambiguous and label != 1:
            label = None
        output.append({"label": label, "category": precise[0]["category"] if label is not None else "unknown",
                       "matched_span_ids": [s["id"] for s in matched],
                       "annotation_precision": sorted({s.get("metadata", {}).get("annotation_precision", "value") for s in matched}),
                       "ambiguous_coarse_positive": has_ambiguous and label is None})
    return output


def prepare_records(docs, context_lines: int = 3, max_chars: int = 6000,
                    legacy_scanner: str | None = "auto") -> list[dict]:
    rows = []
    preprocessing = preprocessing_config(context_lines, max_chars, legacy_scanner)
    for doc in docs:
        text_sha256 = hashlib.sha256(doc["text"].encode()).hexdigest()
        proposals = discover_candidates(doc, legacy_scanner)
        for proposal, supervision in zip(proposals, supervise_candidates(doc, proposals)):
            label, category = supervision["label"], supervision["category"]
            metadata = dict(proposal["metadata"])
            metadata.update({key: value for key, value in supervision.items() if key not in ("label", "category")})
            metadata.update({
                             "candidate_version": 1, "preprocessing": preprocessing,
                             "document_text_sha256": text_sha256,
                             "generation_backend": doc.get("metadata", {}).get("backend"),
                             "mock_data": doc.get("metadata", {}).get("backend") == "mock"})
            rows.append({"id": stable_id(doc["id"], proposal["start"], proposal["end"]), "document_id": doc["id"],
                         **{k: doc[k] for k in ("repo_id", "group_id", "split", "source", "path", "language")},
                         "start": proposal["start"], "end": proposal["end"],
                         "input_text": build_input(doc, proposal, context_lines, max_chars),
                         "label": label, "category": category, "metadata": metadata})
    return rows


def candidate_coverage(docs, records):
    by_doc = {}
    for row in records:
        by_doc.setdefault(row["document_id"], []).append(row)
    total, reached, missed = 0, 0, []
    for doc in docs:
        for span in doc["spans"]:
            if span["label"] != 1:
                continue
            total += 1
            if any(_overlap(span, row) for row in by_doc.get(doc["id"], [])):
                reached += 1
            else:
                missed.append({"document_id": doc["id"], "span_id": span["id"], "split": doc["split"],
                               "source": doc["source"], "category": span["category"]})
    return {"positive_spans": total, "reachable_spans": reached,
            "candidate_recall_ceiling": reached / total if total else None, "missed_spans": missed,
            "record_labels": dict(Counter("unknown" if r["label"] is None else str(r["label"]) for r in records))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--report")
    parser.add_argument("--context-lines", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=6000)
    parser.add_argument("--legacy-scanner", default="auto", help="Existing scan.py to union provider candidates; auto (default) or none")
    args = parser.parse_args(argv)
    docs = list(read_jsonl(args.documents))
    manifest = validate_corpus(docs)
    records = prepare_records(docs, args.context_lines, args.max_chars, args.legacy_scanner)
    write_jsonl(args.out, records)
    report = {**candidate_coverage(docs, records), "corpus_sha256": manifest["corpus_sha256"],
              "legacy_scanner": args.legacy_scanner, "context_lines": args.context_lines, "max_chars": args.max_chars}
    write_json(args.report or str(Path(args.out).with_suffix(".coverage.json")), report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("missed_spans",)}))


if __name__ == "__main__":
    main()
