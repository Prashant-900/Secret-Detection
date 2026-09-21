#!/usr/bin/env python3
"""verify_labels.py - independent consistency gate for a generated dataset directory.

Checks: every file re-parses (yaml/json/hcl/py); each label's line_no and span lie inside the file;
secret spans are non-empty; no (file, line) labeled twice with different labels; paired triplets differ
only on their labeled lines. Exit code 1 on any failure - run it in CI before freezing the dataset.

usage: python verify_labels.py --data out/tierB
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

from mutate_real import parse_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    idx = {r["file_id"]: r for r in csv.DictReader(open(os.path.join(a.data, "files_index.csv")))}
    labels = list(csv.DictReader(open(os.path.join(a.data, "labels.csv"))))
    bad, texts = [], {}
    for fid, r in idx.items():
        p = os.path.join(a.data, "files", fid, r["relpath"])
        t = open(p, encoding="utf-8").read()
        texts[fid] = t.split("\n")
        ok, _ = parse_ok(r["fmt"], t)
        if not ok:
            bad.append((fid, "does_not_parse"))
    seen = {}
    for l in labels:
        lines = texts.get(l["file_id"])
        if lines is None:
            bad.append((l["file_id"], "label_without_file"))
            continue
        n, s, e = int(l["line_no"]), int(l["span_start"]), int(l["span_end"])
        if not 1 <= n <= len(lines):
            bad.append((l["file_id"], f"line_out_of_range:{n}"))
            continue
        if not (0 <= s < e <= len(lines[n - 1])):
            bad.append((l["file_id"], f"bad_span:{n}:{s}-{e}"))
        key = (l["file_id"], n)
        if key in seen and seen[key] != l["label"]:
            bad.append((l["file_id"], f"conflicting_labels:{n}"))
        seen[key] = l["label"]
    pairs = defaultdict(dict)
    for fid, r in idx.items():
        if fid.endswith(("__secret", "__reference", "__placeholder")):
            pairs[fid.rsplit("__", 1)[0]][fid.rsplit("__", 1)[1]] = fid
    lab_lines = defaultdict(set)
    for l in labels:
        lab_lines[l["file_id"]].add(int(l["line_no"]))
    for base, v in pairs.items():
        if len(v) != 3:
            bad.append((base, "incomplete_pair"))
            continue
        ref = texts[v["secret"]]
        for var in ("reference", "placeholder"):
            o = texts[v[var]]
            if len(o) != len(ref):
                bad.append((base, f"pair_length_mismatch:{var}"))
                continue
            diff = {i + 1 for i, (x, y) in enumerate(zip(ref, o)) if x != y}
            if not diff <= (lab_lines[v["secret"]] | lab_lines[v[var]]):
                bad.append((base, f"pair_differs_outside_label:{var}"))
    print(f"checked {len(idx)} files, {len(labels)} labels, {len(pairs)} pairs -> {len(bad)} problems")
    for b in bad[:30]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
