#!/usr/bin/env python3
"""llm_baseline.py - zero-shot LLM baseline: for every candidate line, P(YES = hard-coded secret).

Candidates = all labeled lines + a sample of unlabeled assignment-like lines (assumed benign).
The prompt never mentions operators / labels. Use a DIFFERENT model family than the one that generated data.
Score = P("YES") / (P("YES") + P("NO")) from first-token logprobs, so you can sweep thresholds / compute AP.

usage: python llm_baseline.py --data out/tierB,out/gen --splits test,test_ood --out preds_llm.csv --model <judge-model>
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_candidates, file_lines, load_dirs, window  # noqa: E402
from llm_backend import add_backend_args, make_backend  # noqa: E402

SYSTEM = "You are a secrets-detection expert reviewing infrastructure-as-code files."


def make_prompt(relpath, win):
    return (f"File: {relpath}\n\n```\n{win}\n```\n\n"
            "The line marked with '>>' is under review. Does it contain a hard-coded secret (a password, token, API key, "
            "private key, or credentials embedded in a URL) that should not be committed to source control? "
            "Placeholders, variable or secret-store references, hashes, digests, UUIDs, resource IDs and obvious example values "
            "are NOT secrets. Answer with exactly one word: YES or NO.")


def main():
    ap = argparse.ArgumentParser()
    add_backend_args(ap)
    ap.add_argument("--data", required=True, help="comma-separated dataset dirs")
    ap.add_argument("--splits", default="test,test_ood")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neg-per-file", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=4000)
    a = ap.parse_args()
    idx, labs = load_dirs(a.data.split(","))
    want = set(a.splits.split(","))
    cands = [c for c in build_candidates(idx, labs, a.neg_per_file) if c["split"] in want]
    print(f"{len(cands)} candidate lines in splits {sorted(want)}")
    be = make_backend(a)
    cache, rows = {}, []
    for s in range(0, len(cands), a.chunk):
        part = cands[s:s + a.chunk]
        prompts = []
        for c in part:
            f = idx[c["file_id"]]
            if c["file_id"] not in cache:
                cache[c["file_id"]] = file_lines(f)
            win = window(cache[c["file_id"]], c["line_no"], a.ctx)
            prompts.append((SYSTEM, make_prompt(f["relpath"], win)))
        scores = be.yes_no(prompts)
        rows += [(c["file_id"], c["line_no"], round(sc, 5)) for c, sc in zip(part, scores)]
        print(f"  scored {min(s + a.chunk, len(cands))}/{len(cands)}")
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_id", "line_no", "score"])
        w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
