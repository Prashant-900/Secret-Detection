#!/usr/bin/env python3
"""make_split.py - freeze the REPO-level split BEFORE any mutation or training.

Rules
  1. jenkins + pulumi repos  -> `test_ood`  (unseen-family stress test, reported separately)
  2. everything else is split by OWNER (all repos of one org/user land in the same split), so
     near-duplicate repos (terraform-aws-modules/*, geerlingguy/ansible-role-*) cannot leak
     across train and test.
  3. within that, families are balanced greedily to ~70 / 10 / 20 (train / val / test).

usage: python make_split.py --manifest manifest.csv --out splits.csv [--seed 13]
Commit splits.csv and never regenerate it after seeing results.
"""
import argparse
import csv
import hashlib
import random
from collections import Counter, defaultdict

OOD_FAMILIES = {"jenkins", "pulumi"}
TARGET = {"train": 0.70, "val": 0.10, "test": 0.20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = list(csv.DictReader(open(a.manifest)))
    for r in rows:
        r["owner"] = r["repo"].split("/")[0]
    split_of, owner_split = {}, {}

    for r in rows:
        if r["family"] in OOD_FAMILIES:
            split_of[r["repo"]] = "test_ood"

    by_family = defaultdict(list)
    for r in rows:
        if r["repo"] not in split_of:
            by_family[r["family"]].append(r)

    for fam in sorted(by_family, key=lambda f: -len(by_family[f])):
        repos = by_family[fam]
        groups = defaultdict(list)
        for r in repos:
            groups[r["owner"]].append(r)
        order = sorted(groups, key=lambda o: (-len(groups[o]), rng.random()))
        count = Counter()
        for r in repos:  # owners already locked by an earlier family
            if r["owner"] in owner_split:
                count[owner_split[r["owner"]]] += 1
        for owner in order:
            if owner in owner_split:
                continue
            total = max(1, len(repos))
            # pick the split with the largest remaining deficit relative to its target
            best = max(TARGET, key=lambda s: TARGET[s] * total - count[s])
            owner_split[owner] = best
            count[best] += len(groups[owner])
        for r in repos:
            split_of[r["repo"]] = owner_split[r["owner"]]

    # every non-OOD family with >=3 owners must have at least one val owner (needed for threshold tuning)
    for fam in sorted(by_family):
        owners = {r["owner"] for r in by_family[fam]}
        if len(owners) >= 3 and not any(owner_split[o] == "val" for o in owners):
            fam_n = len(by_family[fam])
            movable = [o for o in owners if owner_split[o] == "train"
                       and sum(r["owner"] == o and r["family"] == fam for r in rows) <= max(2, 0.2 * fam_n)]
            if movable:
                o = min(movable, key=lambda x: (sum(r["owner"] == x for r in rows), x))
                owner_split[o] = "val"
                for r in rows:
                    if r["owner"] == o:
                        split_of[r["repo"]] = "val"

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["repo", "family", "owner", "split"])
        for r in rows:
            w.writerow([r["repo"], r["family"], r["owner"], split_of[r["repo"]]])
    digest = hashlib.sha256(open(a.out, "rb").read()).hexdigest()[:16]

    print(f"wrote {a.out}  (sha256 {digest})  -- commit it, then never change it\n")
    fams = sorted({r["family"] for r in rows})
    print(f"{'family':10s} " + " ".join(f"{s:>8s}" for s in ["train", "val", "test", "test_ood"]))
    for fam in fams:
        c = Counter(split_of[r["repo"]] for r in rows if r["family"] == fam)
        print(f"{fam:10s} " + " ".join(f"{c[s]:8d}" for s in ["train", "val", "test", "test_ood"]))
    tot = Counter(split_of.values())
    print(f"{'TOTAL':10s} " + " ".join(f"{tot[s]:8d}" for s in ["train", "val", "test", "test_ood"]))
    print("\nwarnings:")
    warned = False
    for fam in fams:
        c = Counter(split_of[r["repo"]] for r in rows if r["family"] == fam)
        if fam not in OOD_FAMILIES and (c["test"] == 0 or c["val"] == 0):
            print(f"  - {fam}: no repos in {'test' if c['test'] == 0 else 'val'} (too few owners to split)")
            warned = True
    if not warned:
        print("  none")


if __name__ == "__main__":
    main()
