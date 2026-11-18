#!/usr/bin/env python3
"""
to_sarif.py — Convert canonical findings JSON → SARIF 2.1.0
Accepts the normalised output of normalise.py (a flat list of findings).
"""
import json, sys


def level_of(verdict: str) -> str:
    return "error" if verdict == "BLOCK" else "warning"


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <normalised.json> <output.sarif>", file=sys.stderr)
        sys.exit(1)

    src, dst = sys.argv[1], sys.argv[2]
    findings = json.load(open(src))

    # Build de-duplicated rule table
    rules: dict[str, dict] = {}
    for f in findings:
        rid = f.get("reason", "unknown")
        if rid not in rules:
            rules[rid] = {
                "id": rid,
                "name": rid.replace(":", "_").replace("-", "_"),
                "shortDescription": {
                    "text": f"Secret detected ({rid})"
                },
                "fullDescription": {
                    "text": (f"A potential secret was detected via '{rid}'. "
                             "Review the flagged line and rotate or vault the credential.")
                },
                "helpUri": "https://github.com/your-org/secret-scanner-action#readme",
                "properties": {
                    "tags": ["security", "secret-detection"],
                    "precision": "high" if rid.startswith("rule:") else "medium",
                },
                "defaultConfiguration": {
                    "level": level_of(f["verdict"])
                },
            }

    results = []
    for f in findings:
        rid = f.get("reason", "unknown")
        mode_note = f"[{f.get('mode','?')} mode]  "
        results.append({
            "ruleId":  rid,
            "level":   level_of(f["verdict"]),
            "message": {
                "text": (
                    f"{mode_note}[{f['verdict']}] score={f['score']:.4f}  "
                    f"via {rid}  —  {f['match']}"
                )
            },
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": f["file"],
                        "uriBaseId": "%SRCROOT%",
                    },
                    "region": {
                        "startLine": max(1, int(f["line"])),
                    },
                }
            }],
            "properties": {
                "score":   f["score"],
                "verdict": f["verdict"],
                "mode":    f.get("mode", "unknown"),
            },
        })

    sarif = {
        "$schema": (
            "https://raw.githubusercontent.com/oasis-tcs/sarif-spec"
            "/master/Schemata/sarif-schema-2.1.0.json"
        ),
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name":           "SecretScannerAction",
                    "version":        "2.0.0",
                    "semanticVersion": "2.0.0",
                    "informationUri": "https://github.com/your-org/secret-scanner-action",
                    "rules":          list(rules.values()),
                }
            },
            "results":    results,
            "columnKind": "utf16CodeUnits",
        }],
    }

    with open(dst, "w") as fh:
        json.dump(sarif, fh, indent=2)

    print(f"to_sarif.py: {len(results)} results written → {dst}")


if __name__ == "__main__":
    main()