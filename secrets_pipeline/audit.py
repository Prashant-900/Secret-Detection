"""Measure corpus diversity and build a training-only generation allocation.

An absent format in a supplied subset is not proof of a gap in full CredData.
The report names the available sources and never uses held-out error examples
to design generation prompts or sampling weights.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .schema import read_jsonl, validate_corpus, write_json
from .taxonomy import DEFAULT_OOD_FAMILIES, FAMILIES as FAMILY_SPECS, SECRET_TYPES

FAMILIES = tuple(family for family in FAMILY_SPECS if family not in DEFAULT_OOD_FAMILIES)
TYPES = SECRET_TYPES


def family_of(doc):
    name = doc.get("metadata", {}).get("family")
    aliases = {"gha": "github_actions", "github-actions": "github_actions", "docker-compose": "compose",
               "k8s": "kubernetes", "cfn": "cloudformation", "env": "dotenv", "tf": "terraform"}
    if name:
        return aliases.get(name, name)
    return {"hcl": "terraform", "yaml": "yaml_unspecified", "env": "dotenv", "dotenv": "dotenv", "json": "json_config",
            "javascript": "javascript", "typescript": "javascript"}.get(doc["language"], doc["language"])


def type_of(span):
    name = (span.get("secret_type") or span.get("category", "")).lower()
    if name in TYPES:
        return name
    for token, secret_type in (("stripe", "stripe_secret"), ("slack", "slack_bot_token"),
                               ("gitlab", "gitlab_token"), ("huggingface", "huggingface_token"),
                               ("npm", "npm_token"), ("google", "google_api_key"),
                               ("azure", "azure_client_secret"), ("basic", "basic_auth")):
        if token in name:
            return secret_type
    if "pem" in name or "private key" in name:
        return "private_key_pem"
    if "github" in name:
        return "github_token"
    if "aws" in name and ("secret" in name or "key:secret" in name):
        return "aws_secret_key"
    if "aws" in name and ("access" in name or "client" in name):
        return "aws_access_key"
    if "jwt" in name:
        return "jwt"
    if "connection" in name or "url credential" in name:
        return "connection_string"
    if "base64" in name or span.get("metadata", {}).get("encoding") == "base64":
        return "base64_secret"
    if "human" in name or "low_entropy" in name:
        return "password_low_entropy"
    if "password" in name:
        return "password_random" if "random" in name else "password_unspecified"
    if "hex" in name:
        return "api_key_hex"
    if "token" in name or "secret" in name:
        return "opaque_token"
    return "unmapped"


def audit_documents(docs, target_per_cell=100, max_cells=64, max_per_cell=100, seed_catalog=None):
    docs = list(docs)
    if min(target_per_cell, max_cells, max_per_cell) <= 0:
        raise ValueError("Audit allocation limits must be positive")
    manifest = validate_corpus(docs)
    sources, languages, families, splits = Counter(), Counter(), Counter(), Counter()
    category, source_split, owner_count, label_count, train_cells = Counter(), Counter(), Counter(), Counter(), Counter()
    supervision, uncertain_types = Counter(), Counter()
    unknown = unvalidated = 0
    for doc in docs:
        family = family_of(doc)
        sources[doc["source"]] += 1
        languages[doc["language"]] += 1
        families[family] += 1
        splits[doc["split"]] += 1
        source_split[(doc["source"], doc["split"])] += 1
        owner_count[doc["group_id"]] += 1
        unknown += len(doc.get("metadata", {}).get("unknown_spans", []))
        unvalidated += doc.get("metadata", {}).get("syntax_validated") is False
        for span in doc["spans"]:
            category[span["category"]] += 1
            label_count[str(span["label"])] += 1
            supervision[span.get("metadata", {}).get("annotation_precision", "value")] += 1
            # Frozen val/test/test_ood data never affects generation quotas.
            if doc["split"] == "train" and span["label"] == 1:
                mapped_type = type_of(span)
                train_cells[(family, mapped_type)] += 1
                if mapped_type not in TYPES:
                    uncertain_types[mapped_type] += 1
    gaps = []
    for family in FAMILIES:
        for secret_type in TYPES:
            observed = train_cells[(family, secret_type)]
            deficit = max(0, target_per_cell - observed)
            if deficit:
                gaps.append((deficit, family, secret_type, observed))
    gaps.sort(key=lambda x: (-x[0], x[1], x[2]))
    # Round-robin families prevents the initial budget concentrating in one format.
    queues = defaultdict(list)
    for item in gaps:
        queues[item[1]].append(item)
    for family_index, family in enumerate(FAMILIES):
        # Rotate equal-priority types across formats: a small initial budget must
        # not consist entirely of the same first alphabetic provider formats.
        queues[family].sort(key=lambda item: (-item[0], (TYPES.index(item[2]) - family_index) % len(TYPES)))
    chosen = []
    while len(chosen) < max_cells and any(queues.values()):
        for family in FAMILIES:
            if queues[family] and len(chosen) < max_cells:
                chosen.append(queues[family].pop(0))
    cells = [{"family": family, "secret_type": secret_type, "count": min(deficit, max_per_cell)}
             for deficit, family, secret_type, observed in chosen]
    result = {**manifest, "sources": dict(sources), "languages": dict(languages), "families": dict(families),
              "splits": dict(splits), "source_split": [{"source": s, "split": p, "documents": n} for (s, p), n in sorted(source_split.items())],
              "span_labels": dict(label_count), "categories": dict(category), "annotation_precision": dict(supervision),
              "unknown_annotations_preserved": unknown, "syntax_unvalidated_documents": unvalidated,
              "largest_groups": owner_count.most_common(20), "creddata_present": "creddata" in sources,
              "generation_planning_scope": "train_only", "recommended_generation_cells": cells,
              "uncertain_training_type_spans": dict(uncertain_types),
              "type_count_interpretation": "Mapped counts are annotation-derived lower bounds; generic Password does not imply a random password.",
              "missing_cell_interpretation": "A zero is no mapped evidence in the supplied training subset, not proven absence from full CredData.",
              "generation_config_fragment": {"include_splits": ["train"], "cells": cells},
              "train_cell_coverage": [{"family": f, "secret_type": t, "positive_spans": n} for (f, t), n in sorted(train_cells.items())],
              "limitations": ["Format coverage is measured only on supplied documents, not on absent CredData repositories.",
                              "Synthetic diversity does not establish real-world generalization.",
                              "Jenkins/Pulumi OOD families are excluded from training generation allocations.",
                              "Keep original CredData labels, including UUID policy, fixed for honest comparisons."]}
    if seed_catalog:
        data = json.loads(Path(seed_catalog).read_text(encoding="utf-8"))
        formats = data.get("formats", {})
        repos = {row.get("full_name") for rows in formats.values() for row in rows if row.get("full_name")}
        result["seed_catalog"] = {"format_groups": sorted(formats), "unique_repositories": len(repos),
                                  "status": "unverified_planning_catalog", "licenses_verified": False}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed-catalog")
    parser.add_argument("--target-per-cell", type=int, default=100)
    parser.add_argument("--max-cells", type=int, default=64)
    parser.add_argument("--max-per-cell", type=int, default=100)
    parser.add_argument("--generation-config", help="Write just the allocation fragment for generation config merging")
    args = parser.parse_args(argv)
    result = audit_documents(read_jsonl(args.documents), args.target_per_cell, args.max_cells, args.max_per_cell, args.seed_catalog)
    write_json(args.out, result)
    if args.generation_config:
        write_json(args.generation_config, result["generation_config_fragment"])
    print(json.dumps({"documents": result["documents"], "creddata_present": result["creddata_present"],
                      "train_generation_cells": len(result["recommended_generation_cells"])}))


if __name__ == "__main__":
    main()
