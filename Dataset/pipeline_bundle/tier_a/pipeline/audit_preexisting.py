#!/usr/bin/env python3
"""audit_preexisting.py - find secret-LOOKING strings that are already in the fetched real repos.

Why: Tier B treats every line we did not inject as benign. If a real file already holds a
(test or live) credential, that assumption is wrong. This script lists such lines so a human can
review them; mutate_real.py then skips those files (--exclude-hits).

It is a deliberately simple built-in detector (vendor regexes + sensitive-key/high-entropy rule).
For a stronger pass, also run gitleaks / trufflehog over the repos and review their hits too.

usage: python audit_preexisting.py --repos repos --manifest manifest.csv --out preexisting_hits.csv
Never copy a hit that looks live into the dataset or publish it; report it to the owner.
"""
import argparse
import csv
import math
import os
import re
from collections import Counter

from mutate_real import walk_files
import secrets_lib as S

VENDOR = {
    "aws_access_key": re.compile(r"\b(AKIA|ASIA)[A-Z2-7]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
    "stripe_key": re.compile(r"\b[sr]k_live_[0-9a-zA-Z]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{10,}\b"),
}
KV = re.compile(r"""(?P<key>[A-Za-z_][\w.\-]*)\s*[:=]\s*["']?(?P<val>[^\s"'#,]{12,})["']?""")
BENIGN_VAL = re.compile(r"^(\$|\{\{|var\.|data\.|<|\.\.\.|x{4,}|\*{4,}|changeme|example|your|true|false|http|/|\./)", re.I)


def entropy(s):
    c = Counter(s)
    return -sum(n / len(s) * math.log2(n / len(s)) for n in c.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hits, kloc = [], 0
    for m in csv.DictReader(open(a.manifest)):
        rdir = os.path.join(a.repos, m["repo"].replace("/", "__"))
        if not os.path.isdir(rdir):
            continue
        for path in walk_files(rdir):
            try:
                if os.path.getsize(path) > 500_000:
                    continue
                lines = open(path, encoding="utf-8").read().split("\n")
            except (UnicodeDecodeError, OSError):
                continue
            rel = os.path.relpath(path, rdir)
            kloc += len(lines) / 1000
            for i, ln in enumerate(lines, 1):
                for name, rx in VENDOR.items():
                    if rx.search(ln):
                        hits.append((m["repo"], rel, i, name, ln.strip()[:120]))
                for mm in KV.finditer(ln):
                    v = mm.group("val")
                    if (S.is_sensitive_key(mm.group("key")) and not BENIGN_VAL.match(v) and entropy(v) >= 3.3
                            and re.search(r"\d", v) and re.search(r"[A-Za-z]", v)       # needs letters AND digits
                            and not re.search(r"[(){}$`<>\[\],]", v)                     # not a code expression
                            and not re.fullmatch(r"\w+(\.\w+)+", v)):                    # not obj.attr.chain
                        hits.append((m["repo"], rel, i, "sensitive_key_high_entropy", ln.strip()[:120]))
                        break
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo", "relpath", "line", "detector", "snippet", "human_verdict"])
        for h in hits:
            w.writerow(list(h) + [""])
    files = {(h[0], h[1]) for h in hits}
    print(f"scanned ~{kloc:,.0f} KLOC; {len(hits)} candidate lines in {len(files)} files -> {a.out}")
    print("Fill human_verdict (real / test_fixture / false_positive). Files with hits are skipped by mutate_real.py.")
    print(f"Built-in detector rate: {len(hits) / max(kloc, 1e-9):.2f} candidate lines per KLOC (before human review)")


if __name__ == "__main__":
    main()
