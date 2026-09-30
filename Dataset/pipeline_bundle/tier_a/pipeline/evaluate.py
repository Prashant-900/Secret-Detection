#!/usr/bin/env python3
"""evaluate.py - score any detector against the dataset labels, at SPAN level.

A labeled span = one injected value (a multi-line PEM key is ONE span, not 20 lines).
  secret spans   -> recall (TP / FN), broken down by category, secret_type, key_hint, encoding, commented, operator
  benign spans   -> hard-negative false-positive rate (placeholder / reference / trap)
  other flagged lines (not labeled) are reported as `flagged_unlabeled` and per-KLOC. They are assumed
  false positives, EXCEPT where the real file already held a credential: review those with audit_preexisting.py.

Detectors (repeat any of them):
  --pred   name=predictions.csv          columns file_id,line_no,score  (llm_baseline.py / train_codebert.py output)
  --gitleaks name=report.json            gitleaks JSON report run over <data>/files
  --trufflehog name=report.jsonl         trufflehog --json output run over <data>/files
  --thr    name=0.5|val                  threshold for score detectors ('val' = best F1 on split=val; default)

usage: python evaluate.py --data out/tierB,out/gen --pred codebert=runs/codebert/predictions.csv \
         --gitleaks gitleaks=gl.json --trufflehog trufflehog=th.jsonl --out results.json
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu"))
from common import load_dirs  # noqa: E402

SPLITS = ["val", "test", "test_ood"]


def fid_from_path(path):
    m = re.search(r"/files/([^/]+)/", path.replace("\\", "/"))
    return m.group(1) if m else None


def load_gitleaks(path):
    sc = {}
    for r in json.load(open(path)) or []:
        fid = fid_from_path(r.get("File", ""))
        for ln in range(int(r.get("StartLine", 0)), int(r.get("EndLine", r.get("StartLine", 0))) + 1):
            if fid:
                sc[(fid, ln)] = 1.0
    return sc


def load_trufflehog(path):
    sc = {}
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
            fs = o["SourceMetadata"]["Data"]["Filesystem"]
        except (ValueError, KeyError):
            continue
        fid = fid_from_path(fs.get("file", ""))
        if fid and fs.get("line"):
            sc[(fid, int(fs["line"]))] = 1.0
    return sc


def load_pred(path):
    return {(r["file_id"], int(r["line_no"])): float(r["score"]) for r in csv.DictReader(open(path))}


def build_spans(idx, labs):
    spans = {}
    for l in labs:
        k = (l["file_id"], l["span_id"])
        s = spans.setdefault(k, dict(file_id=l["file_id"], lines=set(), y=int(l["label"] == "secret"), category=l["category"],
                                     secret_type=l["secret_type"], key_hint=l["key_hint"], encoding=l["encoding"],
                                     commented=l["commented"], operator=l["operator"].split(":")[0],
                                     split=idx[l["file_id"]]["split"], family=idx[l["file_id"]]["family"]))
        s["lines"].add(int(l["line_no"]))
    return spans


def span_scores(spans, scores):
    return {k: max((scores.get((s["file_id"], n), 0.0) for n in s["lines"]), default=0.0) for k, s in spans.items()}


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return p, r, f


def best_thr(spans, ss, split):
    ks = [k for k, s in spans.items() if s["split"] == split]
    if not ks:
        return 0.5
    best, bt = -1, 0.5
    for t in [x / 100 for x in range(5, 96, 2)]:
        tp = sum(1 for k in ks if spans[k]["y"] and ss[k] >= t)
        fp = sum(1 for k in ks if not spans[k]["y"] and ss[k] >= t)
        fn = sum(1 for k in ks if spans[k]["y"] and ss[k] < t)
        f = prf(tp, fp, fn)[2]
        if f == f and f > best:
            best, bt = f, t
    return bt


def evaluate(name, scores, spans, idx, thr_spec, is_binary):
    ss = span_scores(spans, scores)
    thr = 0.5 if is_binary else (float(thr_spec) if thr_spec not in (None, "val") else best_thr(spans, ss, "val"))
    labeled_lines = {(s["file_id"], n) for s in spans.values() for n in s["lines"]}
    res = dict(name=name, threshold=thr, splits={})
    for sp in SPLITS:
        ks = [k for k, s in spans.items() if s["split"] == sp]
        if not ks:
            continue
        pos = [k for k in ks if spans[k]["y"]]
        neg = [k for k in ks if not spans[k]["y"]]
        tp = sum(ss[k] >= thr for k in pos)
        fp = sum(ss[k] >= thr for k in neg)
        fn = len(pos) - tp
        p, r, f = prf(tp, fp, fn)
        files = {i for i, f_ in idx.items() if f_["split"] == sp}
        kloc = sum(int(idx[i]["n_lines"]) for i in files) / 1000.0
        flagged_unl = sum(1 for (fid, ln), v in scores.items() if fid in files and v >= thr and (fid, ln) not in labeled_lines)
        d = dict(n_secret_spans=len(pos), n_benign_spans=len(neg), precision_labeled=p, recall=r, f1=f,
                 flagged_unlabeled=flagged_unl, flagged_unlabeled_per_kloc=flagged_unl / kloc if kloc else float("nan"))
        try:
            from sklearn.metrics import average_precision_score
            if pos and neg and not is_binary:
                d["ap"] = float(average_precision_score([spans[k]["y"] for k in ks], [ss[k] for k in ks]))
        except ImportError:
            pass
        for dim in ("category", "secret_type", "key_hint", "encoding", "commented", "operator", "family"):
            rec, fpr = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
            for k in ks:
                v = spans[k][dim] or "-"
                if spans[k]["y"]:
                    rec[v][1] += 1
                    rec[v][0] += ss[k] >= thr
                else:
                    fpr[v][1] += 1
                    fpr[v][0] += ss[k] >= thr
            d[f"recall_by_{dim}"] = {v: [c[0] / c[1], c[1]] for v, c in sorted(rec.items())}
            d[f"fpr_by_{dim}"] = {v: [c[0] / c[1], c[1]] for v, c in sorted(fpr.items())}
        res["splits"][sp] = d
    return res


def fmt(x):
    return "  n/a" if x != x else f"{x:5.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", action="append", default=[])
    ap.add_argument("--gitleaks", action="append", default=[])
    ap.add_argument("--trufflehog", action="append", default=[])
    ap.add_argument("--thr", action="append", default=[])
    ap.add_argument("--out")
    a = ap.parse_args()
    idx, labs = load_dirs(a.data.split(","))
    spans = build_spans(idx, labs)
    thr = dict(t.split("=", 1) for t in a.thr)
    dets = [(n, load_pred(p), False) for n, p in (x.split("=", 1) for x in a.pred)]
    dets += [(n, load_gitleaks(p), True) for n, p in (x.split("=", 1) for x in a.gitleaks)]
    dets += [(n, load_trufflehog(p), True) for n, p in (x.split("=", 1) for x in a.trufflehog)]
    results = [evaluate(n, sc, spans, idx, thr.get(n), b) for n, sc, b in dets]

    for sp in SPLITS:
        if not any(sp in r["splits"] for r in results):
            continue
        print(f"\n== {sp} ==  (span level)")
        print(f"{'detector':16s} {'thr':>5s} {'P(lab)':>6s} {'recall':>6s} {'F1':>6s} {'AP':>6s} {'unlab/KLOC':>10s} {'n+':>5s} {'n-':>5s}")
        for r in results:
            d = r["splits"].get(sp)
            if d:
                print(f"{r['name']:16s} {r['threshold']:5.2f} {fmt(d['precision_labeled']):>6s} {fmt(d['recall']):>6s} {fmt(d['f1']):>6s} "
                      f"{fmt(d.get('ap', float('nan'))):>6s} {d['flagged_unlabeled_per_kloc']:10.2f} {d['n_secret_spans']:5d} {d['n_benign_spans']:5d}")
    for dim in ("category", "key_hint", "encoding", "commented"):
        print(f"\n-- recall by {dim} (test split) --")
        for r in results:
            d = r["splits"].get("test")
            if d:
                print(f"{r['name']:16s} " + "  ".join(f"{k}={v[0]:.2f}(n={v[1]})" for k, v in d[f"recall_by_{dim}"].items()))
    print("\n-- false-positive rate on benign look-alikes (test split) --")
    for r in results:
        d = r["splits"].get("test")
        if d:
            print(f"{r['name']:16s} " + "  ".join(f"{k}={v[0]:.2f}(n={v[1]})" for k, v in d["fpr_by_category"].items()))
    if a.out:
        json.dump(results, open(a.out, "w"), indent=1)
        print("\nwrote", a.out)
    c = Counter(s["split"] for s in spans.values())
    print(f"\nspans per split: {dict(c)}   (small n => report intervals, not just point estimates)")


if __name__ == "__main__":
    main()
