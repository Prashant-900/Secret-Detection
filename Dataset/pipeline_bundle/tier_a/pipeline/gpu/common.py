"""common.py - load generated datasets and build line-level candidates (shared by baseline / training / eval)."""
import csv
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mutate_real import parse_line  # noqa: E402

SECRET = "secret"


def load_dirs(dirs):
    """dirs: list of dataset dirs (each has files_index.csv, labels.csv, files/). Returns (idx, labels)."""
    idx, labs = {}, []
    for d in dirs:
        src = os.path.basename(os.path.normpath(d))
        for r in csv.DictReader(open(os.path.join(d, "files_index.csv"))):
            r["dir"], r["source"] = d, src
            idx[r["file_id"]] = r
        for r in csv.DictReader(open(os.path.join(d, "labels.csv"))):
            r["source"] = src
            labs.append(r)
    return idx, labs


def file_lines(row):
    p = os.path.join(row["dir"], "files", row["file_id"], row["relpath"])
    return open(p, encoding="utf-8").read().split("\n")


def window(lines, line_no, ctx=6, mask_key=False, fmt="yaml"):
    """Context window with the target line marked '>> '. Optionally mask the key name on the target line."""
    lo, hi = max(0, line_no - 1 - ctx), min(len(lines), line_no + ctx)
    out = []
    for i in range(lo, hi):
        ln = lines[i]
        if i == line_no - 1:
            if mask_key:
                p = parse_line(fmt, ln)
                if p:
                    ln = ln.replace(p["key"], "KEY", 1)
            out.append(">> " + ln)
        else:
            out.append("   " + ln)
    return "\n".join(out)


def build_candidates(idx, labs, neg_per_file=6, seed=13, min_val_len=6):
    """Labeled lines (secret=1; reference/placeholder/trap=0) + sampled unlabeled assignment lines (=0).
    Unlabeled lines are ASSUMED benign (they were not injected); audit_preexisting.py reduces the noise."""
    rng = random.Random(seed)
    rows, labeled = [], set()
    for l in labs:
        n = int(l["line_no"])
        labeled.add((l["file_id"], n))
        f = idx[l["file_id"]]
        rows.append(dict(file_id=l["file_id"], line_no=n, y=int(l["label"] == SECRET), category=l["category"],
                         span_id=l["span_id"], split=f["split"], fmt=f["fmt"], source=f["source"],
                         key_hint=l["key_hint"], operator=l["operator"], secret_type=l["secret_type"],
                         encoding=l["encoding"], commented=l["commented"], family=f["family"]))
    for fid, f in idx.items():
        lines = file_lines(f)
        cand = []
        for i, ln in enumerate(lines, 1):
            if (fid, i) in labeled:
                continue
            p = parse_line(f["fmt"], ln)
            if p and (p["ve"] - p["vs"]) >= min_val_len:
                cand.append(i)
        for i in rng.sample(cand, min(neg_per_file, len(cand))):
            rows.append(dict(file_id=fid, line_no=i, y=0, category="unlabeled", span_id="", split=f["split"],
                             fmt=f["fmt"], source=f["source"], key_hint="", operator="", secret_type="",
                             encoding="", commented="0", family=f["family"]))
    return rows
