#!/usr/bin/env python3
"""
normalise.py — Unify fast-scanner and llm-scanner JSON into one canonical list.

Fast scanner output  (Scanner/scanner/scan.py --json):
  Top-level array:
    [ { "file", "line", "score", "reason", "match", "verdict" }, ... ]
  - Includes PASS-scored items (verdict == "PASS")
  - 'match' is partially redacted: "ghp_ab...cdef"
  - 'reason' is always present: "rule:aws-access-token" or "entropy:b64ish"

LLM scanner output  (python3 -m secrets_pipeline scan --out):
  Top-level object:
    {
      "status": "BLOCK|WARN|PASS|ERROR",
      "findings": [ { "file", "line", "column", "start", "end",
                       "score", "verdict", "match" }, ... ],
      "thresholds": {...}, "detector": "...", ...
    }
  - Only findings >= warn threshold appear (no PASS entries)
  - 'match' is always "[REDACTED]"
  - 'reason' is ABSENT

Canonical output (written to --out):
  [ { "file", "line", "score", "verdict", "reason", "match", "mode" }, ... ]
  - Only BLOCK / WARN items (PASS filtered out)
  - Sorted by verdict (BLOCK first) then score descending
"""

import argparse, json, sys

FAST_MODE = "fast"
LLM_MODE  = "llm"

def normalise_fast(raw: list, mode: str) -> list:
    out = []
    for f in raw:
        if f.get("verdict", "PASS") == "PASS":
            continue
        out.append({
            "file":    f["file"],
            "line":    int(f["line"]),
            "score":   round(float(f["score"]), 6),
            "verdict": f["verdict"],
            "reason":  f.get("reason", "unknown"),
            "match":   f.get("match", "[REDACTED]"),
            "mode":    mode,
        })
    return out


def normalise_llm(raw: dict, mode: str) -> list:
    findings = raw.get("findings", [])
    out = []
    for f in findings:
        score = round(float(f["score"]), 6)
        
        # If score >= 0.70, enforce BLOCK even if selected.json had block: null
        verdict = f.get("verdict", "WARN")
        if score >= 0.70:
            verdict = "BLOCK"
            
        out.append({
            "file":    f["file"],
            "line":    int(f["line"]),
            "score":   score,
            "verdict": verdict,
            "reason":  f.get("reason", "ml-encoder"),
            "match":   f.get("match", "[REDACTED]"),
            "mode":    mode,
        })
    return out


def sort_key(f):
    return (0 if f["verdict"] == "BLOCK" else 1, -f["score"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw",  required=True, help="Path to raw scanner JSON output")
    ap.add_argument("--out",  required=True, help="Path to write canonical findings JSON")
    ap.add_argument("--mode", required=True, choices=[FAST_MODE, LLM_MODE])
    args = ap.parse_args()

    with open(args.raw) as fh:
        raw = json.load(fh)

    if args.mode == FAST_MODE:
        if not isinstance(raw, list):
            print(f"::error::fast mode expected a JSON array, got {type(raw).__name__}",
                  file=sys.stderr)
            sys.exit(2)
        findings = normalise_fast(raw, args.mode)

    else:  # llm
        if not isinstance(raw, dict):
            print(f"::error::llm mode expected a JSON object, got {type(raw).__name__}",
                  file=sys.stderr)
            sys.exit(2)
        status = raw.get("status", "")
        if status == "ERROR":
            errors = raw.get("errors", [])
            print(f"::error::secrets_pipeline scan returned ERROR: {errors}", file=sys.stderr)
            sys.exit(2)
        findings = normalise_llm(raw, args.mode)

    findings.sort(key=sort_key)

    with open(args.out, "w") as fh:
        json.dump(findings, fh, indent=2)

    total = len(findings)
    blocked = sum(1 for f in findings if f["verdict"] == "BLOCK")
    warned  = sum(1 for f in findings if f["verdict"] == "WARN")
    print(f"normalise.py: {total} findings  (BLOCK={blocked}  WARN={warned})  → {args.out}")


if __name__ == "__main__":
    main()