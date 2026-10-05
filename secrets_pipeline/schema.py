"""Versioned, dependency-free document interchange and leakage checks.

Offsets are Unicode character offsets, zero-based and end-exclusive. A missing
annotation means unknown, never a negative. Paths are metadata, not disk targets.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = 1
SPLITS = ("train", "val", "test", "test_ood")


def stable_id(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()[:24]


def read_jsonl(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            yield row


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    """Atomic replacement; interruption cannot leave a partially written corpus."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def write_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def validate_document(doc: dict) -> None:
    for key in ("id", "repo_id", "group_id", "source", "path", "language"):
        if not isinstance(doc.get(key), str) or not doc[key]:
            raise ValueError(f"Document {doc.get('id', '?')}: missing/non-string {key}")
    if doc.get("split") not in SPLITS:
        raise ValueError(f"Document {doc['id']}: invalid split {doc.get('split')!r}")
    if not isinstance(doc.get("text"), str):
        raise ValueError(f"Document {doc['id']}: text must be a string")
    if not isinstance(doc.get("spans"), list):
        raise ValueError(f"Document {doc['id']}: spans must be a list")
    span_ids = set()
    for span in doc["spans"]:
        if not isinstance(span, dict) or not isinstance(span.get("id"), str) or not span["id"]:
            raise ValueError(f"Document {doc['id']}: annotation needs an id")
        if span["id"] in span_ids:
            raise ValueError(f"Document {doc['id']}: duplicate annotation id {span['id']}")
        span_ids.add(span["id"])
        a, b = span.get("start"), span.get("end")
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(doc["text"]):
            raise ValueError(f"Document {doc['id']}: out-of-bounds annotation {span['id']}")
        if type(span.get("label")) is not int or span["label"] not in (0, 1):
            raise ValueError(f"Document {doc['id']}: only explicit binary labels belong in spans")
        if not isinstance(span.get("category"), str) or not span["category"]:
            raise ValueError(f"Document {doc['id']}: annotation category required")
    # A positive and negative on overlapping text is ambiguous supervision.
    ordered = sorted(doc["spans"], key=lambda s: s["start"])
    for i, left in enumerate(ordered):
        for right in ordered[i + 1:]:
            if right["start"] >= left["end"]:
                break
            if left["label"] != right["label"]:
                raise ValueError(f"Document {doc['id']}: conflicting overlapping annotations")


def masked_fingerprint(doc: dict) -> str:
    """Detect copied templates even when the injected values change."""
    text = doc["text"]
    intervals = []
    for span in sorted(doc.get("spans", []), key=lambda s: s["start"]):
        if intervals and span["start"] <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], span["end"])
        else:
            intervals.append([span["start"], span["end"]])
    for start, end in reversed(intervals):
        text = text[:start] + " <LITERAL> " + text[end:]
    text = re.sub(r"\s+", " ", text).strip()
    return hashlib.sha256(text.encode()).hexdigest()


def validate_corpus(docs: Iterable[dict], check_near_duplicates: bool = True, near_duplicate_threshold: float = 0.85) -> dict:
    """Fail closed on document, owner/repo, and exact/template split leakage."""
    seen, repo_split, group_split, content_split, template_split = {}, {}, {}, {}, {}
    from .validation import DuplicateIndex, masked_text
    near_duplicates = DuplicateIndex(near_duplicate_threshold) if check_near_duplicates else None
    counts = Counter()
    digest = hashlib.sha256()
    for doc in docs:
        validate_document(doc)
        did, split = doc["id"], doc["split"]
        if did in seen:
            raise ValueError(f"Duplicate document id: {did}")
        seen[did] = True
        for name, key, mapping in (("repository", doc["repo_id"].lower(), repo_split),
                                   ("group", doc["group_id"].lower(), group_split)):
            if key in mapping and mapping[key] != split:
                raise ValueError(f"Cross-split {name} leakage for {key}: {mapping[key]} / {split}")
            mapping[key] = split
        for name, key, mapping in (("content", hashlib.sha256(doc["text"].encode()).hexdigest(), content_split),
                                   ("masked template", masked_fingerprint(doc), template_split)):
            if key in mapping and mapping[key][0] != split:
                raise ValueError(f"Cross-split duplicate {name}: {mapping[key][1]} and {did}")
            mapping[key] = (split, did)
        if near_duplicates is not None:
            template = masked_text(doc)
            match = near_duplicates.match(template, doc["group_id"], split, cross_split_only=True)
            if match:
                raise ValueError(f"Cross-split near-duplicate template: {did} and group {match['group_id']} (Jaccard {match['similarity']})")
            near_duplicates.add(template, doc["group_id"], split)
        counts[split] += 1
        digest.update(json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return {"schema_version": SCHEMA_VERSION, "documents": len(seen), "splits": dict(counts),
            "repositories": len(repo_split), "groups": len(group_split), "corpus_sha256": digest.hexdigest(),
            "cross_split_near_duplicates_checked": check_near_duplicates,
            "near_duplicate_threshold": near_duplicate_threshold if check_near_duplicates else None}


def safe_child(root: str | Path, relative: str) -> Path:
    root = Path(root).resolve()
    candidate = (root / relative.replace("\\", "/")).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Path escapes dataset root: {relative}")
    return candidate


def line_offsets(text: str) -> tuple[list[str], list[int]]:
    # Preserve newline width so offsets refer to the exact persisted text.
    lines = text.splitlines(keepends=True)
    if not lines:
        lines = [""]
    offsets, offset = [], 0
    for line in lines:
        offsets.append(offset)
        offset += len(line)
    return lines, offsets


def language_for(path: str) -> str:
    name = Path(path).name.lower()
    if name.startswith(".env."):
        return "dotenv"
    special = {"dockerfile": "dockerfile", "jenkinsfile": "groovy", ".env": "dotenv",
               ".netrc": "config", ".npmrc": "config", ".pypirc": "config", ".credentials": "config",
               "credentials": "config", "inventory": "config", "id_rsa": "pem", "id_ed25519": "pem",
               "id_ecdsa": "pem", "id_dsa": "pem"}
    if name in special:
        return special[name]
    return {".tf": "hcl", ".tfvars": "hcl", ".yml": "yaml", ".yaml": "yaml", ".json": "json",
            ".toml": "toml", ".ini": "ini", ".conf": "config", ".cfg": "config", ".properties": "properties",
            ".env": "dotenv", ".pem": "pem", ".key": "pem",
            ".py": "python", ".js": "javascript", ".ts": "typescript", ".java": "java",
            ".go": "go", ".rb": "ruby", ".cs": "csharp", ".php": "php", ".sh": "shell",
            ".xml": "xml", ".groovy": "groovy", ".rs": "rust"}.get(Path(name).suffix, "text")
