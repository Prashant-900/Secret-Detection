#!/usr/bin/env python3
"""
Train the scanner-v2 re-ranker (scanner_v2/model_v2.json).

Training data (nothing from the CredData TEST repos is ever read):
  * CredData, 'dev' repos only (repo-level split: crc32(repo) % 5 in (0,1) -> test, else dev)
      - labeled lines: candidate label = overlaps a True value span (T) / otherwise negative (F, X)
      - pure-'UUID' True rows are treated as NEGATIVE (a UUID is an identifier, not an IaC secret;
        the evaluator drops those rows with --drop-categories UUID)
      - optional: candidates on UNLABELED lines of the dev files as negatives (assumption: lines
        CredSweeper's candidate pass did not select are benign) - needed to keep false alarms low
        on ordinary code (--unlabeled-negatives, default on)
  * synthetic IaC benchmark, 'train' families only (test families = cfn, k8s are never read)

Thresholds are chosen on OUT-OF-FOLD predictions of the dev data (GroupKFold by repo):
  warn  = threshold with the best F1;  block = lowest threshold with precision >= --block-precision

Usage:
  python3 training/train_v2.py --creddata-dir datasets/CredData --synthetic-dir dataset/generated
"""
import argparse, collections, csv, importlib.util, json, math, multiprocessing as mp, os, random
import sys, time, zlib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def split_of(repo):
    return "test" if zlib.crc32(repo.encode()) % 5 in (0, 1) else "dev"


def repo_of(path):
    p = path.split("/")
    return p[1] if len(p) > 2 and p[0] == "data" else p[0]


_S = {}


def _init(scan_path, rules_path):
    spec = importlib.util.spec_from_file_location("scan_v2", scan_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rules, ctx, kw, nokw = mod.load_rules(rules_path)
    ctx["model"] = None            # features only, no model needed
    _S.update(mod=mod, ctx=ctx, kw=kw, nokw=nokw)


def _file_job(job):
    """job = (path, ext, labeled: {line: [(gt, vs, ve, row_id, multi)]}, all_lines: bool, weight, group)
       returns (units, examples)"""
    path, ext, labeled, all_lines, weight, group, unl_keep = job
    mod, ctx, kw, nokw = _S["mod"], _S["ctx"], _S["kw"], _S["nokw"]
    try:
        lines = open(path, errors="replace").read().split("\n")
    except OSError:
        return [], []
    rng = random.Random(zlib.crc32(path.encode()))
    units = {}        # unit key -> is_T
    ex = []           # (features, label, unit_key or None, weight, group, kind)
    target = set(labeled) | (set(range(1, len(lines) + 1)) if all_lines else set())
    for ln in sorted(target):
        if ln > len(lines):
            continue
        line = lines[ln - 1]
        if len(line) > 2000 or len(line) < 6:
            continue
        info = labeled.get(ln)
        if info is None and rng.random() > unl_keep:
            continue
        cf = mod.candidate_features(line, ext, ctx, kw, nokw, lines[ln - 2] if ln >= 2 else "")
        if info is None:                                   # unlabeled line -> negative sample
            for c, f in cf:
                ex.append((f, 0, None, weight, group, "unl"))
            continue
        trows = [r for r in info if r[0] == "T"]
        for r in info:                                     # units for threshold selection
            if r[0] == "T":
                units[("T", r[3])] = True
            elif not trows:
                units[("N", ln)] = False
        for c, f in cf:
            pos = False
            for gt, vs, ve, rid, multi in trows:
                if multi or vs is None or ve is None or (c["s"] < ve and c["e"] > vs):
                    pos = True
            if trows:
                u = ("T", trows[0][3])
            elif ("N", ln) in units:
                u = ("N", ln)
            else:
                u = None
            ex.append((f, int(pos), u, weight, group, "lab"))
        # T rows with no candidate on this line still count as units (recall ceiling) - handled above
    return list(units.items()), ex


def read_creddata(cred_dir, drop_uuid=True):
    meta_dir = os.path.join(cred_dir, "meta")
    byfile = collections.defaultdict(lambda: collections.defaultdict(list))
    n_rows = 0
    for fn in sorted(os.listdir(meta_dir)):
        if not fn.endswith(".csv"):
            continue
        for r in csv.DictReader(open(os.path.join(meta_dir, fn), newline="")):
            fp = r["FilePath"].replace("\\", "/")
            if split_of(repo_of(fp)) != "dev":
                continue                                   # NEVER touch test repos
            gt = r["GroundTruth"].strip().upper()
            if gt not in ("T", "F", "X"):
                continue
            if gt == "T" and drop_uuid and r["Category"] == "UUID":
                gt = "F"
            s, e = int(r["LineStart"]), int(r["LineEnd"] or r["LineStart"])
            vs = int(r["ValueStart"]) if r["ValueStart"] not in ("", "-1") else None
            ve = int(r["ValueEnd"]) if r["ValueEnd"] not in ("", "-1") else None
            multi = e > s
            rid = int(r["Id"])
            for ln in range(s, e + 1):
                byfile[fp][ln].append((gt, vs, ve, rid, multi))
            n_rows += 1
    return byfile, n_rows


def read_synthetic(syn_dir):
    byfile = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(os.path.join(syn_dir, "labels.csv"), newline="")):
        if r["split"] != "train":
            continue                                       # NEVER touch test families
        gt = "T" if r["kind"].startswith("real") else "F"
        byfile[r["file"]][int(r["line"])].append((gt, None, None, hash(r["file"] + r["line"]) & 0xFFFFFFF, False))
    return byfile


def threshold_table(unit_score, unit_is_t, t_total_extra=0):
    ths = np.round(np.arange(0.02, 0.99, 0.01), 2)
    T = unit_is_t.sum() + t_total_extra
    rows = []
    for t in ths:
        flagged = unit_score >= t
        tp = int((flagged & unit_is_t).sum())
        fp = int((flagged & ~unit_is_t).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / T if T else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        rows.append((float(t), p, r, f1))
    return rows


def export_hgb(clf, feature_names, extra):
    trees = []
    for it in clf._predictors:
        nodes = it[0].nodes
        left = np.where(nodes["is_leaf"], -1, nodes["left"].astype(np.int64)).tolist()
        right = np.where(nodes["is_leaf"], -1, nodes["right"].astype(np.int64)).tolist()
        trees.append(dict(l=left, r=right, f=nodes["feature_idx"].astype(int).tolist(),
                          t=[float(x) for x in nodes["num_threshold"]],
                          v=[float(x) for x in nodes["value"]]))
    return dict(feature_names=feature_names, init=float(np.ravel(clf._baseline_prediction)[0]),
                lr=1.0, trees=trees, **extra)


def py_predict(model, x):
    raw = model["init"]
    for t in model["trees"]:
        i = 0
        while t["l"][i] != -1:
            i = t["l"][i] if x[t["f"][i]] <= t["t"][i] else t["r"][i]
        raw += model["lr"] * t["v"][i]
    return 1 / (1 + math.exp(-raw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creddata-dir", default=os.path.join(ROOT, "datasets", "CredData"))
    ap.add_argument("--synthetic-dir", default=os.path.join(ROOT, "dataset", "generated"))
    ap.add_argument("--scanner-dir", default=os.path.join(ROOT, "scanner_v2"))
    ap.add_argument("--out", default=None, help="default: <scanner-dir>/model_v2.json")
    ap.add_argument("--no-unlabeled-negatives", action="store_true")
    ap.add_argument("--unlabeled-keep", type=float, default=0.35,
                    help="fraction of unlabeled lines (in dev files) sampled as negatives")
    ap.add_argument("--synthetic-weight", type=float, default=5.0)
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--block-precision", type=float, default=0.90)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--cache", default=os.path.join(HERE, "features_cache.npz"),
                    help="extracted training features are cached here")
    ap.add_argument("--reuse-features", action="store_true", help="skip feature extraction, load --cache")
    args = ap.parse_args()
    t0 = time.time()
    scan_py = os.path.join(args.scanner_dir, "scan.py")
    rules = os.path.join(args.scanner_dir, "rules.toml")
    out = args.out or os.path.join(args.scanner_dir, "model_v2.json")

    spec = importlib.util.spec_from_file_location("scan_v2_main", scan_py)
    scan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scan)
    feature_names = scan.FEATURE_NAMES

    if args.reuse_features and os.path.exists(args.cache):
        z = np.load(args.cache, allow_pickle=True)
        X, y, w, unit_idx, unit_is_t, groups, kinds = (z[k] for k in
                                                       ("X", "y", "w", "unit_idx", "unit_is_t", "groups", "kinds"))
        groups = list(groups)
        assert list(z["feature_names"]) == list(feature_names), "cache from a different feature set"
        print(f"loaded cached features from {args.cache}: {len(y)} candidates")
    else:
        jobs = []
        cb, n_rows = read_creddata(args.creddata_dir)
        print(f"CredData dev: {len(cb)} labeled files, {n_rows} rows")
        for fp, labeled in cb.items():
            ext = scan.ext_of(fp)
            jobs.append((os.path.join(args.creddata_dir, fp), ext, dict(labeled),
                         not args.no_unlabeled_negatives, 1.0, repo_of(fp), args.unlabeled_keep))
        if not args.no_synthetic:
            sb = read_synthetic(args.synthetic_dir)
            # every train-split file (labeled or not) is scanned in full; unlabeled lines are benign
            for dp, _, fns in os.walk(os.path.join(args.synthetic_dir, "train")):
                for fn in fns:
                    full = os.path.join(dp, fn)
                    rel = os.path.relpath(full, args.synthetic_dir).replace("\\", "/")
                    grp = "syn:" + rel.split("/")[1].rsplit("-", 1)[0]
                    jobs.append((full, scan.ext_of(fn), dict(sb.get(rel, {})), True,
                                 args.synthetic_weight, grp, 1.0))
            print(f"synthetic train files: {sum(1 for j in jobs if j[5].startswith('syn:'))}")

        X, y, w, groups, kinds, unit_idx = [], [], [], [], [], []
        unit_is_t = []
        unit_ids = {}
        with mp.Pool(args.jobs, initializer=_init, initargs=(scan_py, rules)) as pool:
            for k, (units, ex) in enumerate(pool.imap_unordered(_file_job, jobs, chunksize=8)):
                base = {}
                for key, is_t in units:
                    gid = len(unit_is_t)
                    unit_is_t.append(is_t)
                    base[key] = gid
                for f, lab, u, wt, grp, kind in ex:
                    X.append(f); y.append(lab); w.append(wt); groups.append(grp); kinds.append(kind)
                    unit_idx.append(base.get(u, -1) if u is not None else -1)
                if (k + 1) % 2000 == 0:
                    print(f"  processed {k + 1}/{len(jobs)} files, {len(X)} candidates ({time.time() - t0:.0f}s)")
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        w = np.asarray(w, dtype=np.float64)
        unit_idx = np.asarray(unit_idx)
        unit_is_t = np.asarray(unit_is_t, dtype=bool)

        np.savez_compressed(args.cache, X=X, y=y, w=w, unit_idx=unit_idx, unit_is_t=unit_is_t,
                            groups=np.asarray(groups), kinds=kinds, feature_names=np.asarray(feature_names))
    gcodes = {g: i for i, g in enumerate(sorted(set(groups)))}
    g = np.asarray([gcodes[x] for x in groups])
    kinds = np.asarray(kinds)
    print(f"{len(X)} candidates ({int(y.sum())} positive), {len(unit_is_t)} units "
          f"({int(unit_is_t.sum())} true), {len(gcodes)} groups, features={X.shape[1]}")

    def make():
        return HistGradientBoostingClassifier(max_iter=args.iters, learning_rate=0.08, max_leaf_nodes=31,
                                              min_samples_leaf=20, l2_regularization=1.0,
                                              early_stopping=False, random_state=0)

    # ---- out-of-fold predictions for threshold selection
    oof = np.zeros(len(y))
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, g)):
        m = make().fit(X[tr], y[tr], sample_weight=w[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        print(f"  fold {fold}: AUC {roc_auc_score(y[te], oof[te]):.4f}  ({time.time() - t0:.0f}s)")
    lab_mask = kinds == "lab"
    print(f"OOF AUC (labeled candidates): {roc_auc_score(y[lab_mask], oof[lab_mask]):.4f}")

    unit_score = np.zeros(len(unit_is_t))
    ok = unit_idx >= 0
    np.maximum.at(unit_score, unit_idx[ok], oof[ok])
    tab = threshold_table(unit_score, unit_is_t)
    best = max(tab, key=lambda r: r[3])
    blk = next((r for r in tab if r[1] >= args.block_precision and r[2] > 0.02), tab[-1])
    print(f"WARN  threshold {best[0]:.2f}: P={best[1]:.3f} R={best[2]:.3f} F1={best[3]:.3f}  (dev, out-of-fold)")
    print(f"BLOCK threshold {blk[0]:.2f}: P={blk[1]:.3f} R={blk[2]:.3f} F1={blk[3]:.3f}")
    cand_recall = float((unit_score[unit_is_t] > 0).mean())
    print(f"candidate recall ceiling on dev true units: {cand_recall:.3f}")

    # ---- final model on all dev data
    clf = make().fit(X, y, sample_weight=w)
    meta = dict(thresholds=dict(warn=round(best[0], 2), block=round(blk[0], 2)),
                oof=dict(warn=dict(precision=round(best[1], 4), recall=round(best[2], 4), f1=round(best[3], 4)),
                         block=dict(precision=round(blk[1], 4), recall=round(blk[2], 4), f1=round(blk[3], 4)),
                         auc=round(float(roc_auc_score(y[lab_mask], oof[lab_mask])), 4),
                         candidate_recall=round(cand_recall, 4)),
                trained_on=dict(creddata_split="dev (crc32(repo)%5 not in {0,1})",
                                synthetic_split="train families only",
                                candidates=int(len(y)), positives=int(y.sum()),
                                unlabeled_negatives=not args.no_unlabeled_negatives,
                                synthetic_weight=None if args.no_synthetic else args.synthetic_weight,
                                drop_uuid_positive=True))
    model = export_hgb(clf, feature_names, meta)
    # sanity: exported pure-python model == sklearn
    idx = np.random.RandomState(0).choice(len(X), size=min(300, len(X)), replace=False)
    ref = clf.predict_proba(X[idx])[:, 1]
    mine = np.array([py_predict(model, X[i].tolist()) for i in idx])
    err = float(np.abs(ref - mine).max())
    print(f"export check: max |sklearn - exported| = {err:.2e}")
    assert err < 1e-4, "export mismatch"
    with open(out, "w") as f:
        json.dump(model, f, separators=(",", ":"))
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.2f} MB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
