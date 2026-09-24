#!/usr/bin/env python3
"""Shrink a CredData checkout to N repositories (in place), so download_data.py
fetches a few repos instead of all 337.  Only run it on a throw-away clone.

  python3 subset_creddata.py datasets/CredData_quick 15 [seed]
"""
import binascii, csv, json, os, random, sys

d, n = sys.argv[1], int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
snap_p = os.path.join(d, "snapshot.json")
snap = json.load(open(snap_p))

def new_id(repo_id):                       # same mapping download_data.py uses
    return f"{binascii.crc32(binascii.unhexlify(repo_id)):08x}"

def n_true(repo_id):
    p = os.path.join(d, "meta", new_id(repo_id) + ".csv")
    if not os.path.exists(p):
        return 0
    with open(p, newline="") as f:
        return sum(r["GroundTruth"] == "T" for r in csv.DictReader(f))

# keep only repos that actually contain true credentials, pick N at random (seeded)
cands = sorted(k for k in snap if n_true(k) >= 5)
random.Random(seed).shuffle(cands)
keep = set(cands[:n])
json.dump({k: v for k, v in snap.items() if k in keep}, open(snap_p, "w"), indent=1)
keep_csv = {new_id(k) + ".csv" for k in keep}
for fn in os.listdir(os.path.join(d, "meta")):
    if fn.endswith(".csv") and fn not in keep_csv:
        os.remove(os.path.join(d, "meta", fn))
print(f"kept {len(keep)} repos:", ", ".join(snap[k] for k in sorted(keep)))
