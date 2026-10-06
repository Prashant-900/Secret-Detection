"""Non-executing syntax checks, label checks, and value-masked deduplication.

No validation routine evaluates source code or verifies a credential online.
Parser availability is explicit; a missing parser never produces a parsed=True
claim. Grammar validity is not a claim of deployability or configuration semantics.
"""
from __future__ import annotations

import ast
import configparser
import hashlib
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from xml.etree import ElementTree


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    parsed: bool
    parser: str
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def parser_available(language):
    if language in {"json", "python", "xml", "ini", "dotenv"}:
        return True
    if language == "toml":
        return importlib.util.find_spec("tomllib") is not None or importlib.util.find_spec("tomli") is not None
    if language == "yaml":
        return importlib.util.find_spec("yaml") is not None
    if language == "hcl":
        return importlib.util.find_spec("hcl2") is not None
    return importlib.util.find_spec("tree_sitter_language_pack") is not None


def validate_syntax(text, language, require_parsed=True):
    """Accept only syntax supported by an installed parser or an explicit subset."""
    parser = language
    try:
        if language == "json":
            def unique_keys(pairs):
                out = {}
                for key, value in pairs:
                    if key in out:
                        raise ValueError("duplicate JSON key")
                    out[key] = value
                return out
            json.loads(text, object_pairs_hook=unique_keys)
            parser = "stdlib-json"
        elif language == "python":
            ast.parse(text)
            parser = "stdlib-ast"
        elif language == "toml":
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib
            tomllib.loads(text)
            parser = "tomllib"
        elif language == "yaml":
            import yaml
            class UniqueLoader(yaml.SafeLoader):
                pass
            def mapping(loader, node, deep=False):
                loader.flatten_mapping(node)
                result = {}
                for key_node, value_node in node.value:
                    key = loader.construct_object(key_node, deep=deep)
                    if key in result:
                        raise ValueError("duplicate YAML key")
                    result[key] = loader.construct_object(value_node, deep=deep)
                return result
            UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
            list(yaml.load_all(text, Loader=UniqueLoader))
            parser = "pyyaml-safe-unique"
        elif language == "hcl":
            import hcl2
            hcl2.loads(text)
            parser = "python-hcl2"
        elif language == "xml":
            if re.search(r"<!\s*(?:DOCTYPE|ENTITY)", text, re.I):
                raise ValueError("XML entities/doctype prohibited")
            ElementTree.fromstring(text)
            parser = "stdlib-elementtree-no-dtd"
        elif language == "ini":
            configparser.ConfigParser(interpolation=None, strict=True).read_string(text)
            parser = "stdlib-configparser"
        elif language == "dotenv":
            # Strict single-line dotenv subset. No shell execution or expansion.
            for line in text.splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                m = re.fullmatch(r"\s*(?:export\s+)?[A-Za-z_][A-Za-z0-9_]*\s*=\s*(.*?)\s*", line)
                if not m:
                    raise ValueError("not a dotenv assignment")
                value = m.group(1)
                if value.startswith('"'):
                    json.loads(value)
                elif value.startswith("'"):
                    if len(value) < 2 or not value.endswith("'") or "'" in value[1:-1]:
                        raise ValueError("invalid dotenv quote")
                elif re.search(r"\s|[\"'`]", value):
                    raise ValueError("invalid dotenv unquoted value")
            parser = "dotenv-single-line-subset"
        elif language in {"javascript", "typescript", "java", "go", "groovy"}:
            from tree_sitter_language_pack import get_parser
            tree = get_parser(language).parse(text.encode("utf-8"))
            if tree.root_node.has_error:
                raise ValueError("tree-sitter syntax error")
            parser = f"tree-sitter-{language}"
        else:
            return ValidationResult(not require_parsed, False, "unavailable", "unsupported_language")
        return ValidationResult(True, True, parser)
    except (ImportError, ModuleNotFoundError):
        return ValidationResult(not require_parsed, False, "unavailable", "missing_parser")
    except Exception as exc:
        # Diagnostic is a class, not source text or a generated literal.
        return ValidationResult(False, False, parser, f"parse_error:{type(exc).__name__}")


def validate_spans(document):
    text = document["text"]
    spans = sorted(document.get("spans", []), key=lambda s: (s["start"], s["end"]))
    last_end = 0
    seen = set()
    for span in spans:
        if span["id"] in seen:
            raise ValueError("duplicate span id")
        seen.add(span["id"])
        start, end = span["start"], span["end"]
        if not isinstance(start, int) or not isinstance(end, int) or not (0 <= start < end <= len(text)):
            raise ValueError("invalid span bounds")
        if start < last_end:
            raise ValueError("overlapping spans")
        if span["label"] not in (0, 1):
            raise ValueError("invalid label")
        if span.get("surface_sha256") != hashlib.sha256(text[start:end].encode()).hexdigest():
            raise ValueError("span content hash mismatch")
        last_end = end
    if "@@" in text:
        raise ValueError("unfilled marker")
    return True


def masked_text(document):
    text = document["text"]
    for span in sorted(document.get("spans", []), key=lambda s: s["start"], reverse=True):
        text = text[:span["start"]] + " __VALUE__ " + text[span["end"]:]
    return text


def canonical_tokens(text):
    # Ignore comments and formatting. Keep identifiers and operators: two independently
    # generated copies with new planted credentials must still collide.
    text = re.sub(r"@@[A-Z_]+:[a-z0-9_]+@@", " __VALUE__ ", text)
    text = re.sub(r"(?m)^\s*(?:#|//|;).*$", "", text)
    text = re.sub(r"/\*.*?\*/|<!--.*?-->", "", text, flags=re.S)
    return re.findall(r"__VALUE__|[\w]+|[^\s\w]", text.lower())


def fingerprint(text, n=5):
    tokens = canonical_tokens(text)
    normalized = " ".join(tokens)
    exact = hashlib.sha256(normalized.encode()).hexdigest()
    if len(tokens) < n:
        return exact, {exact}
    shingles = {hashlib.blake2b(" ".join(tokens[i:i + n]).encode(), digest_size=8).hexdigest()
                for i in range(len(tokens) - n + 1)}
    return exact, shingles


class DuplicateIndex:
    """Exact Jaccard checks using length and prefix filters (no LSH misses).

    Expected counterfactual variants sharing a group are allowed. Any duplicate
    between different groups is rejected, including those in the same split.

    Only ``n - ceil(threshold*n) + 1`` ordered shingles enter the inverted index.
    A qualifying pair must share a prefix shingle under a common total order.
    After 128 examples the order is frozen using bootstrap token frequencies,
    putting ubiquitous syntax last; older entries are reindexed at that point.
    Corpus growth does not subsequently change the order. Final decisions always
    use exact full-set Jaccard, including for short snippets.
    """
    def __init__(self, threshold=0.85):
        if not 0 < threshold <= 1:
            raise ValueError("near_duplicate_threshold must be in (0, 1]")
        self.threshold = threshold
        self.entries = []
        self.exact = defaultdict(lambda: defaultdict(list))
        self.inverted = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        self.bootstrap_frequency = Counter()
        self.frozen_frequency = None

    def _prefix(self, shingles):
        if self.frozen_frequency is None:
            ordered = sorted(shingles)
        else:
            ordered = sorted(shingles, key=lambda s: (self.frozen_frequency.get(s, 0), s))
        size = len(ordered) - math.ceil(self.threshold * len(ordered) - 1e-12) + 1
        return ordered[:max(1, size)]

    def _index_entry(self, index, entry):
        size = len(entry["shingles"])
        for shingle in self._prefix(entry["shingles"]):
            self.inverted[shingle][size][entry["split"]].add(index)

    @staticmethod
    def _match_info(entry, similarity, split):
        return {"group_id": entry["group_id"], "split": entry["split"],
                "similarity": round(similarity, 5), "cross_split": entry["split"] != split}

    def match(self, text, group_id, split, *, cross_split_only=False):
        """Return a duplicate, optionally ignoring every same-split entry.

        The filter is applied before exact lookup and candidate ranking; a same-
        split match cannot hide a later cross-split one. Split-partitioned postings
        avoid walking large same-split template families in corpus-only audits.
        """
        exact, shingles = fingerprint(text)
        for existing_split, ids in self.exact.get(exact, {}).items():
            if cross_split_only and existing_split == split:
                continue
            for index in ids:
                entry = self.entries[index]
                if entry["group_id"] != group_id or entry["split"] != split:
                    return self._match_info(entry, 1.0, split)
        size = len(shingles)
        lower, upper = math.ceil(self.threshold * size - 1e-12), math.floor(size / self.threshold + 1e-12)
        prefix_hits = Counter()
        for shingle in self._prefix(shingles):
            for other_size, split_ids in self.inverted.get(shingle, {}).items():
                if lower <= other_size <= upper:
                    for existing_split, ids in split_ids.items():
                        if not cross_split_only or existing_split != split:
                            prefix_hits.update(ids)
        # Most promising candidates first; common template duplicates are rejected
        # immediately instead of walking all postings for the same template family.
        for index, _ in sorted(prefix_hits.items(), key=lambda item: (-item[1], item[0])):
            entry = self.entries[index]
            if entry["group_id"] == group_id and entry["split"] == split:
                continue
            other = entry["shingles"]
            required = math.ceil((self.threshold / (1 + self.threshold)) * (size + len(other)) - 1e-12)
            overlap = len(shingles & other)
            if overlap < required:
                continue
            similarity = overlap / max(1, size + len(other) - overlap)
            if similarity >= self.threshold:
                return self._match_info(entry, similarity, split)
        return None

    def add(self, text, group_id, split):
        exact, shingles = fingerprint(text)
        index = len(self.entries)
        self.entries.append({"exact": exact, "shingles": shingles, "group_id": group_id, "split": split})
        self.exact[exact][split].append(index)
        if self.frozen_frequency is None:
            self.bootstrap_frequency.update(shingles)
            if len(self.entries) == 128:
                self.frozen_frequency = dict(self.bootstrap_frequency)
                self.inverted.clear()
                for old_index, entry in enumerate(self.entries):
                    self._index_entry(old_index, entry)
                return
        self._index_entry(index, self.entries[index])
