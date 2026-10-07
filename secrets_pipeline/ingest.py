"""Import local CredData/Tier B data into a frozen, audited JSONL corpus.

No downloads occur. Invalid/missing files fail the command unless explicitly
quarantined with --skip-invalid. Existing split assignments never change.
"""
from __future__ import annotations

import argparse
import binascii
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

from .schema import (SPLITS, language_for, line_offsets, read_jsonl, safe_child,
                     stable_id, validate_corpus, validate_document, write_json, write_jsonl)


class SplitRegistry:
    def __init__(self, path: str | Path | None = None, seed: int = 404):
        self.path = Path(path) if path else None
        self.seed = seed
        self.groups = {}
        if self.path and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("seed") != seed:
                raise ValueError("Split registry already frozen with another seed")
            self.groups = data["groups"]
            if any(value not in SPLITS for value in self.groups.values()):
                raise ValueError("Invalid split registry")

    def assign(self, group: str, requested: str | None = None) -> str:
        group = group.lower()
        if requested and requested not in SPLITS:
            raise ValueError(f"Invalid imported split {requested!r}")
        if group in self.groups:
            if requested and requested != self.groups[group]:
                raise ValueError(f"Frozen split conflict for {group}: {self.groups[group]} versus {requested}")
            return self.groups[group]
        # 70/15/15 owner-group split, stable across process order and dataset growth.
        bucket = int(hashlib.sha256(f"{self.seed}:{group}".encode()).hexdigest()[:8], 16) % 10000
        split = requested or ("train" if bucket < 7000 else "val" if bucket < 8500 else "test")
        self.groups[group] = split
        return split

    def save(self):
        if self.path:
            write_json(self.path, {"version": 1, "seed": self.seed, "policy": "owner-70/15/15-imported-frozen",
                                   "groups": self.groups})


def _csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _canonical_repo(value: str) -> str:
    value = value.strip().replace("\\", "/").rstrip("/")
    if "://" in value:
        value = urlparse(value).path.strip("/")
    if value.endswith(".git"):
        value = value[:-4]
    return value.lower()


def _group(repo_id: str) -> str:
    return "owner:" + repo_id.split("/", 1)[0] if "/" in repo_id else "repo:" + repo_id


def _read_source(path):
    # Normalize CRLF once, before deriving offsets; this normalized text is persisted.
    return Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")


def load_tier_b(root: str | Path, registry: SplitRegistry, errors: list) -> list[dict]:
    root = Path(root).resolve()
    index = _csv(root / "files_index.csv")
    labels = defaultdict(list)
    for row in _csv(root / "labels.csv"):
        labels[row["file_id"]].append(row)
    docs, seen = [], set()
    for row in index:
        fid = row["file_id"]
        try:
            if fid in seen:
                raise ValueError("Duplicate file_id in index")
            seen.add(fid)
            repo = _canonical_repo(row["repo"])
            group = _group(repo)
            split = registry.assign(group, row["split"])
            path = safe_child(root / "files", fid + "/" + row["relpath"])
            text = _read_source(path)
            lines, starts = line_offsets(text)
            units = defaultdict(list)
            for label in labels[fid]:
                if label["split"] != split or _canonical_repo(label["repo"]) != repo:
                    raise ValueError("Label/index repository or split disagreement")
                number = int(label["line_no"])
                lo, hi = int(label["span_start"]), int(label["span_end"])
                if not 1 <= number <= len(lines):
                    raise ValueError("Label line outside file")
                line = lines[number - 1].rstrip("\n")
                if not 0 <= lo < hi <= len(line):
                    raise ValueError("Label columns outside line")
                if label["label"] not in ("secret", "benign"):
                    raise ValueError("Unexpected Tier B label")
                units[label["span_id"]].append((starts[number - 1] + lo, starts[number - 1] + hi, label))
            spans = []
            for sid, parts in sorted(units.items()):
                first = parts[0][2]
                if len({p[2]["label"] for p in parts}) != 1:
                    raise ValueError("Conflicting labels within a span")
                spans.append({"id": str(sid), "start": min(p[0] for p in parts), "end": max(p[1] for p in parts),
                              "label": int(first["label"] == "secret"), "category": first["category"],
                              "secret_type": first.get("secret_type", ""), "label_origin": "deterministic_injection",
                              "metadata": {k: first.get(k, "") for k in ("key_hint", "encoding", "commented", "operator", "pair_id", "variant")}})
            doc = {"id": "tier_b:" + stable_id(repo, fid, row["relpath"]), "repo_id": repo, "group_id": group,
                   "split": split, "source": "tier_b", "path": row["relpath"].replace("\\", "/"),
                   "language": row.get("fmt") or language_for(row["relpath"]), "text": text, "spans": spans,
                   "metadata": {"family": row.get("family", ""), "original_file_id": fid,
                                "syntax_validated": row.get("validated") == "1", "annotation_unit": "injected_span",
                                "input_sha256": hashlib.sha256(text.encode()).hexdigest()}}
            validate_document(doc)
            docs.append(doc)
        except (OSError, UnicodeError, ValueError, KeyError) as exc:
            errors.append({"source": "tier_b", "file_id": fid, "error": str(exc)})
    for fid in sorted(set(labels) - seen):
        errors.append({"source": "tier_b", "file_id": fid, "error": "Labels refer to an unindexed file"})
    return docs


def _snapshot_repos(root: Path) -> dict:
    mapping = {}
    path = root / "snapshot.json"
    if not path.exists():
        return mapping
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("CredData snapshot.json must be an object")
    for key, value in obj.items():
        if isinstance(value, dict):
            value = value.get("url") or value.get("repo") or value.get("RepoName")
        if not isinstance(value, str):
            continue
        repo = _canonical_repo(value)
        mapping[key] = repo
        try:
            mapping[f"{binascii.crc32(binascii.unhexlify(key)):08x}"] = repo
        except (ValueError, binascii.Error):
            pass
    return mapping


def load_creddata(root: str | Path, registry: SplitRegistry, errors: list,
                  repo_map: dict | None = None) -> list[dict]:
    root = Path(root).resolve()
    meta_files = sorted((root / "meta").glob("*.csv"))
    if not meta_files:
        raise ValueError(f"No CredData meta CSV files in {root / 'meta'}")
    mapping = _snapshot_repos(root)
    mapping.update(repo_map or {})
    by_file = defaultdict(list)
    for meta in meta_files:
        for number, row in enumerate(_csv(meta), 2):
            row["_meta_file"], row["_meta_line"] = meta.name, number
            by_file[row["FilePath"].replace("\\", "/")].append(row)
    docs = []
    for logical_path, rows in sorted(by_file.items()):
        try:
            text = _read_source(safe_child(root, logical_path))
            lines, starts = line_offsets(text)
            parts = logical_path.split("/")
            opaque = parts[1] if len(parts) > 2 and parts[0] == "data" else rows[0]["_meta_file"].removesuffix(".csv")
            raw_repo = mapping.get(opaque) or rows[0].get("RepoName") or rows[0].get("Repository")
            repo = _canonical_repo(raw_repo) if raw_repo else "creddata-opaque:" + opaque
            group = _group(repo)
            split = registry.assign(group)
            spans, unknown, dedup = [], [], set()
            for row in rows:
                gt = row["GroundTruth"].strip().upper()
                if gt not in ("T", "F", "X"):
                    raise ValueError(f"Unexpected GroundTruth {gt!r}")
                first, last = int(row["LineStart"]), int(row.get("LineEnd") or row["LineStart"])
                if not 1 <= first <= last <= len(lines):
                    raise ValueError("CredData annotation line outside file")
                raw_start, raw_end = row.get("ValueStart", ""), row.get("ValueEnd", "")
                precise = first == last and raw_start not in (None, "", "-1") and raw_end not in (None, "", "-1")
                if precise:
                    lo, hi = int(raw_start), int(raw_end)
                    if not 0 <= lo < hi <= len(lines[first - 1].rstrip("\n")):
                        raise ValueError("CredData annotation columns outside line")
                    start, end = starts[first - 1] + lo, starts[first - 1] + hi
                else:
                    start = starts[first - 1]
                    end = starts[last - 1] + len(lines[last - 1].rstrip("\n"))
                if start >= end:
                    raise ValueError("Empty CredData annotation")
                category = row.get("Category") or "unspecified"
                key = (start, end, gt, category)
                if key in dedup:
                    continue
                dedup.add(key)
                metadata = {"ground_truth": gt, "meta_file": row["_meta_file"], "row_id": row.get("Id", ""),
                            "annotation_precision": "value" if precise else "line_range"}
                if gt == "X":
                    unknown.append({"start": start, "end": end, "category": category, **metadata})
                    continue
                spans.append({"id": stable_id(*key), "start": start, "end": end, "label": int(gt == "T"),
                              "category": category, "label_origin": "creddata_annotation", "metadata": metadata})
            doc = {"id": "creddata:" + stable_id(repo, logical_path), "repo_id": repo, "group_id": group,
                   "split": split, "source": "creddata", "path": "/".join(parts[2:]) if len(parts) > 2 and parts[0] == "data" else logical_path,
                   "language": language_for(logical_path), "text": text, "spans": spans,
                   "metadata": {"original_path": logical_path, "owner_resolved": bool(raw_repo and "/" in repo),
                                "unknown_spans": unknown, "annotation_unit": "value_or_line_range",
                                "input_sha256": hashlib.sha256(text.encode()).hexdigest()}}
            validate_document(doc)
            docs.append(doc)
        except (OSError, UnicodeError, ValueError, KeyError) as exc:
            errors.append({"source": "creddata", "path": logical_path, "error": str(exc)})
    return docs


def merge_corpora(docs: list[dict], errors: list, skip_invalid: bool = False,
                  near_duplicate_threshold: float = 0.85) -> list[dict]:
    # Prefer held-out documents if duplicate templates appear in multiple splits.
    # Removing an overlapping train item is safer than deleting a held-out test.
    priority = {"test_ood": 0, "test": 1, "val": 2, "train": 3}
    accepted, ids, repos, groups, content, templates = [], set(), {}, {}, {}, {}
    from .schema import masked_fingerprint
    from .validation import DuplicateIndex, masked_text
    near_duplicates = DuplicateIndex(near_duplicate_threshold)
    for doc in sorted(docs, key=lambda d: (priority[d["split"]], d["id"])):
        conflict = None
        if doc["id"] in ids:
            conflict = "Duplicate document id"
        else:
            for title, mapping, key in (("repository", repos, doc["repo_id"].lower()), ("group", groups, doc["group_id"].lower()),
                                        ("content", content, hashlib.sha256(doc["text"].encode()).hexdigest()),
                                        ("masked_template", templates, masked_fingerprint(doc))):
                if key in mapping and mapping[key] != doc["split"]:
                    conflict = f"Cross-split {title} conflict (held-out-priority quarantine)"
                    break
        template = masked_text(doc)
        if conflict is None:
            match = near_duplicates.match(template, doc["group_id"], doc["split"], cross_split_only=True)
            if match:
                conflict = (f"Cross-split near-duplicate template, Jaccard={match['similarity']}, "
                            f"retained_group={match['group_id']} (held-out-priority quarantine)")
        if conflict:
            errors.append({"source": doc["source"], "document_id": doc["id"], "split": doc["split"], "error": conflict})
            if skip_invalid:
                continue
        accepted.append(doc)
        ids.add(doc["id"])
        repos[doc["repo_id"].lower()] = groups[doc["group_id"].lower()] = doc["split"]
        content[hashlib.sha256(doc["text"].encode()).hexdigest()] = doc["split"]
        templates[masked_fingerprint(doc)] = doc["split"]
        near_duplicates.add(template, doc["group_id"], doc["split"])
    return sorted(accepted, key=lambda d: d["id"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier-b", action="append", default=[], help="Tier B directory; repeat to import multiple")
    parser.add_argument("--creddata", action="append", default=[], help="Downloaded CredData root; no downloading")
    parser.add_argument("--documents", action="append", default=[], help="Already normalized JSONL; repeat")
    parser.add_argument("--repo-map", help="JSON mapping opaque CredData repository ids to owner/name")
    parser.add_argument("--out", required=True)
    parser.add_argument("--registry", help="Persistent frozen split JSON; defaults alongside corpus")
    parser.add_argument("--report", help="Manifest and quarantine report JSON")
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--skip-invalid", action="store_true", help="Explicitly quarantine invalid/missing/leaking documents")
    parser.add_argument("--near-duplicate-threshold", type=float, default=0.85)
    args = parser.parse_args(argv)
    if not (args.tier_b or args.creddata or args.documents):
        parser.error("Provide at least one data input")
    report_path = args.report or str(Path(args.out).with_suffix(".manifest.json"))
    registry = SplitRegistry(args.registry or str(Path(args.out).with_suffix(".splits.json")), args.seed)
    errors, docs = [], []
    # Import established real-repository splits before assigning new ones.
    for root in args.tier_b:
        docs.extend(load_tier_b(root, registry, errors))
    repo_map = json.loads(Path(args.repo_map).read_text(encoding="utf-8")) if args.repo_map else None
    for root in args.creddata:
        docs.extend(load_creddata(root, registry, errors, repo_map))
    for path in args.documents:
        for doc in read_jsonl(path):
            try:
                validate_document(doc)
                registry.assign(doc["group_id"], doc["split"])
                docs.append(doc)
            except ValueError as exc:
                errors.append({"source": str(path), "document_id": doc.get("id"), "error": str(exc)})
    docs = merge_corpora(docs, errors, args.skip_invalid, args.near_duplicate_threshold)
    if errors and not args.skip_invalid:
        write_json(report_path, {"status": "failed", "quarantine": errors, "documents_not_published": len(docs)})
        raise SystemExit(f"Import refused: {len(errors)} problems. Inspect {report_path}; --skip-invalid explicitly quarantines them.")
    if not docs:
        raise SystemExit("Import produced zero valid documents")
    manifest = validate_corpus(docs, near_duplicate_threshold=args.near_duplicate_threshold)
    creddata_docs = [doc for doc in docs if doc["source"] == "creddata"]
    unresolved_docs = [doc for doc in creddata_docs if not doc.get("metadata", {}).get("owner_resolved", False)]
    manifest.update({"status": "complete", "quarantine": errors, "unknown_is_negative": False,
                     "uuid_labels_preserved": True, "source_inputs": {"tier_b": args.tier_b, "creddata": args.creddata,
                                                                     "documents": args.documents},
                     "creddata_owner_resolution": {"documents": len(creddata_docs),
                                                   "resolved_documents": len(creddata_docs) - len(unresolved_docs),
                                                   "unresolved_documents": len(unresolved_docs),
                                                   "unresolved_repositories": len({doc['repo_id'] for doc in unresolved_docs}),
                                                   "owner_heldout_guaranteed": (not unresolved_docs) if creddata_docs else None,
                                                   "qualification": "Unresolved CredData owners use repository-id grouping only; supply snapshot.json or --repo-map for owner-held-out guarantees."}})
    write_jsonl(args.out, docs)
    registry.save()
    write_json(report_path, manifest)
    print(json.dumps({"documents": len(docs), "quarantined": len(errors), "corpus_sha256": manifest["corpus_sha256"]}))


if __name__ == "__main__":
    main()
