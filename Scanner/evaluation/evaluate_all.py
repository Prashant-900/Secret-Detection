#!/usr/bin/env python3
"""
evaluate_all.py - one script to evaluate our scanner against Gitleaks,
TruffleHog and detect-secrets on

  * the synthetic IaC benchmark   (dataset/generated + labels.csv)
  * CredData (Samsung)            (datasets/CredData, labels in meta/*.csv)

Everything is scored by the SAME code, so the tools are compared fairly.

Scoring protocols (both are always reported)
--------------------------------------------
  labeled : only lines that carry a ground-truth label are judged.
              flag on a T line            -> TP
              flag on a labeled F/X line  -> FP
              flag anywhere else          -> ignored
            This is the protocol CredData was designed for: it only labels
            candidate lines, so an unlabeled line is "unknown", not "negative".
  strict  : like `labeled`, but a flag on an unlabeled line (inside the
            labeled files) is ALSO a false positive. Harsh for broad scanners.
            (This is how the original evaluation/evaluate.py scored the
            synthetic data, where every non-labeled line really is benign.)

  Recall is the same in both: detected T rows / all T rows.
  Hits are always de-duplicated to unique (file, line) pairs, so a tool that
  reports 3 findings on one line is not penalised 3 times.
  --tolerance K : a flag within +-K lines of a T row counts as detecting it.

Raw tool output is cached in results/raw/, so you can re-score with other
options (--tolerance, --x-policy ...) without re-running the slow tools:
      python3 evaluation/evaluate_all.py --reuse-raw --tolerance 2

Usage
-----
  python3 evaluation/evaluate_all.py                       # both datasets, all tools
  python3 evaluation/evaluate_all.py --dataset synthetic
  python3 evaluation/evaluate_all.py --dataset creddata --tools ours,gitleaks
  python3 evaluation/evaluate_all.py --help
"""
import argparse, collections, concurrent.futures, csv, importlib.util, json, os, shutil
import subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

ALL_TOOLS = ["ours", "gitleaks", "trufflehog", "detect-secrets", "credsweeper"]
# optional extra "ours" variants (scanner dirs next to scanner/), e.g. scanner_v2



# ----------------------------------------------------------------------------
# datasets -> list of ground-truth rows
#   row = dict(file, start, end, gt in {"T","F","X"}, cat)
# ----------------------------------------------------------------------------
def load_synthetic(dataset_root):
    path = os.path.join(dataset_root, "labels.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        for r in reader:
            if "file_id" in fields and "line_no" in fields:
                file_id = (r.get("file_id") or "").strip()
                relpath = (r.get("relpath") or "").strip()
                if file_id:
                    file = f"files/{file_id}/{relpath}" if relpath else f"files/{file_id}"
                else:
                    file = (r.get("file") or "").replace("\\", "/")
                ln = int(r["line_no"])
                label = (r.get("label") or "").strip().lower()
                if label == "secret":
                    gt = "T"
                elif label == "benign":
                    gt = "F"
                else:
                    gt = "T" if (r.get("kind") or "").startswith("real") else "F"
                cat = (r.get("category") or "")
                if cat and r.get("secret_type"):
                    cat = f"{cat}:{r['secret_type']}"
                elif not cat:
                    cat = r.get("secret_type") or (r.get("kind") or "")
                rows.append(dict(file=file.replace("\\", "/"), start=ln, end=ln,
                                 gt=gt, cat=cat))
                continue

            ln = int(r["line"])
            kind = r["kind"]
            rows.append(dict(file=r["file"].replace("\\", "/"), start=ln, end=ln,
                             gt="T" if kind.startswith("real") else "F",
                             cat=f'{kind}:{r["secret_type"]}'))
    return rows


def load_creddata(cred_root):
    meta_dir = os.path.join(cred_root, "meta")
    data_dir = os.path.join(cred_root, "data")
    if not os.path.isdir(meta_dir):
        raise FileNotFoundError(f"{meta_dir} not found - did you clone CredData?")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"{data_dir} not found - run download_data.py first "
                                "(see README_TESTING.md, step 3)")
    rows, seen, missing = [], {}, 0
    for fn in sorted(os.listdir(meta_dir)):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(meta_dir, fn), newline="") as f:
            for r in csv.DictReader(f):
                fp = r["FilePath"].replace("\\", "/")
                if fp not in seen:
                    seen[fp] = os.path.exists(os.path.join(cred_root, fp))
                if not seen[fp]:
                    missing += 1      # repo not downloaded (partial download) -> skip row
                    continue
                gt = r["GroundTruth"].strip().upper()
                if gt not in ("T", "F", "X"):
                    continue
                s = int(r["LineStart"]); e = int(r["LineEnd"] or s)
                rows.append(dict(file=fp, start=s, end=max(s, e), gt=gt,
                                 cat=r.get("Category", "") or "(none)"))
    if missing:
        print(f"  [creddata] skipped {missing} label rows whose files are not on disk "
              f"(partial download?)")
    return rows


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
class Scorer:
    def __init__(self, rows, tol=0, x_policy="negative", strict_scope="labeled-files"):
        self.rows, self.tol, self.x_policy = rows, tol, x_policy
        self.pos = collections.defaultdict(list)   # (file,line) -> [T row idx]  (tolerance-expanded)
        self.neg = {}                              # (file,line) -> "F"/"X"       (exact lines)
        self.labeled_files = set()
        self.t_rows = []
        for i, r in enumerate(rows):
            self.labeled_files.add(r["file"])
            if r["gt"] == "T":
                self.t_rows.append(i)
                for ln in range(r["start"] - tol, r["end"] + tol + 1):
                    self.pos[(r["file"], ln)].append(i)
            else:
                for ln in range(r["start"], r["end"] + 1):
                    self.neg.setdefault((r["file"], ln), r["gt"])
        self.strict_scope = strict_scope
        self.ignore_files = set()      # e.g. labels.csv sits inside the scanned folder

    def classify(self, key):
        """TP | FPL (labeled F/X line) | FPU (unlabeled line) | IGN"""
        if key[0] in self.ignore_files:
            return "IGN"
        if key in self.pos:
            return "TP"
        g = self.neg.get(key)
        if g is not None:
            return "IGN" if (g == "X" and self.x_policy == "ignore") else "FPL"
        if self.strict_scope == "labeled-files" and key[0] not in self.labeled_files:
            return "IGN"
        return "FPU"

    def evaluate(self, hits):
        tp = fpl = fpu = 0
        detected = set()
        for key in hits:
            c = self.classify(key)
            if c == "TP":
                tp += 1
                detected.update(self.pos[key])
            elif c == "FPL":
                fpl += 1
            elif c == "FPU":
                fpu += 1
        n_t = len(self.t_rows)
        rec = len(detected) / n_t if n_t else 0.0

        def prf(fp):
            p = tp / (tp + fp) if tp + fp else 0.0
            f1 = 2 * p * rec / (p + rec) if p + rec else 0.0
            return round(p, 4), round(f1, 4)
        p_l, f_l = prf(fpl)
        p_s, f_s = prf(fpl + fpu)
        return dict(hits=tp + fpl + fpu, TP=tp, FP_labeled=fpl, FP_unlabeled=fpu,
                    T_rows=n_t, T_detected=len(detected),
                    recall=round(rec, 4),
                    precision_labeled=p_l, f1_labeled=f_l,
                    precision_strict=p_s, f1_strict=f_s)

    def per_category(self, hits, top=15):
        """recall for T categories, flag-rate for F/X categories."""
        detected = set()
        flagged_neg = collections.Counter()
        for key in hits:
            if key in self.pos:
                detected.update(self.pos[key])
            elif key in self.neg:
                flagged_neg[key] = 1
        t_tot, t_det, n_tot, n_flag = (collections.Counter() for _ in range(4))
        for i, r in enumerate(self.rows):
            if r["gt"] == "T":
                t_tot[r["cat"]] += 1
                t_det[r["cat"]] += i in detected
            else:
                n_tot[r["cat"]] += 1
                n_flag[r["cat"]] += any((r["file"], ln) in flagged_neg
                                        for ln in range(r["start"], r["end"] + 1))
        t = [(c, t_tot[c], round(t_det[c] / t_tot[c], 3)) for c, _ in t_tot.most_common(top)]
        n = [(c, n_tot[c], round(n_flag[c] / n_tot[c], 3)) for c, _ in n_tot.most_common(top)]
        return dict(secret_recall_by_category=t, nonsecret_flag_rate_by_category=n)


# ----------------------------------------------------------------------------
# tool runners -> list of (file, line, score, reason); paths relative to dataset root
# ----------------------------------------------------------------------------
def rel(path, root, cwd=None):
    p = path if os.path.isabs(path) else os.path.join(cwd or root, path)
    return os.path.relpath(os.path.abspath(p), root).replace("\\", "/")


def find_bin(name, bin_dir):
    for d in (bin_dir, None):
        p = shutil.which(name, path=d) if d else shutil.which(name)
        if p:
            return p
    return None


def run_cmd(cmd, cwd=None):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    return p, time.time() - t0


def run_gitleaks(binp, root, scan_dir, rawdir, **_):
    rep = os.path.join(rawdir, "gitleaks_report.json")
    if os.path.exists(rep):
        os.remove(rep)
    p, dt = run_cmd([binp, "dir", scan_dir, "--no-banner", "--exit-code", "0",
                     "-f", "json", "-r", rep])
    hits = []
    if os.path.exists(rep):
        try:
            data = json.load(open(rep))
        except json.JSONDecodeError:
            data = []
        for f in data or []:
            hits.append((rel(f.get("File", ""), root), int(f.get("StartLine", 0)),
                         1.0, "rule:" + str(f.get("RuleID", ""))))
    else:
        print("  [gitleaks] no report written. stderr:", p.stderr[-400:])
    return hits, dt


def run_trufflehog(binp, root, scan_dir, rawdir, **_):
    p, dt = run_cmd([binp, "filesystem", scan_dir, "--json", "--no-verification", "--no-update"])
    hits = []
    for ln in p.stdout.splitlines():
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            continue
        fs = ((d.get("SourceMetadata") or {}).get("Data") or {}).get("Filesystem") or {}
        if fs.get("file"):
            hits.append((rel(fs["file"], root), int(fs.get("line", 0) or 0),
                         1.0, "detector:" + str(d.get("DetectorName", ""))))
    if not hits and p.returncode not in (0, 183):
        print("  [trufflehog] exit", p.returncode, p.stderr[-400:])
    return hits, dt


def run_detect_secrets(root, scan_dir, paths=None, **_):
    p, dt = run_cmd([sys.executable, "-m", "detect_secrets", "scan", "--all-files", "--no-verify"]   # --no-verify: never call out to AWS/Telegram/... with found secrets
                    
                    + [os.path.relpath(x, root) for x in (paths or [scan_dir])], cwd=root)
    hits = []
    try:
        data = json.loads(p.stdout)
    except json.JSONDecodeError:
        print("  [detect-secrets] could not parse output. stderr:", p.stderr[-400:])
        return hits, dt
    for fname, items in data.get("results", {}).items():
        for it in items:
            hits.append((rel(fname, root, cwd=root), int(it.get("line_number", 0)),
                         1.0, "plugin:" + str(it.get("type", ""))))
    return hits, dt


def run_credsweeper(binp, root, scan_dir, rawdir, jobs=4, paths=None, **_):
    out = os.path.join(rawdir, "credsweeper_report.json")
    if os.path.exists(out):
        os.remove(out)
    p, dt = run_cmd([binp, "--path"] + [os.path.relpath(x, root) for x in (paths or [scan_dir])] + ["--jobs", str(jobs),
                     "--save-json", out, "--no-stdout", "--sort"], cwd=root)
    hits = []
    if not os.path.exists(out):
        print("  [credsweeper] no report written. stderr:", p.stderr[-400:])
        return hits, dt
    for c in json.load(open(out)):
        ld = (c.get("line_data_list") or [{}])[0]
        if "path" not in ld:
            continue
        mp = c.get("ml_probability")
        hits.append((rel(ld["path"], root, cwd=root), int(ld.get("line_num", 0)),
                     float(mp) if mp is not None else 1.0, "rule:" + str(c.get("rule", ""))))
    return hits, dt


# --- our scanner: imported in-process, run on every file regardless of extension ---
_G = {}


def _init(scan_mod_path, rules_path, fixed):
    spec = importlib.util.spec_from_file_location("iac_scan", scan_mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rules, ctx, kw, nokw = mod.load_rules(rules_path)
    if fixed:   # evaluation-only patch: treat the 1st capture group as the secret
        for r in rules:
            if "secret_group" not in r:
                r["secret_group"] = 0
            rx = r.get("compiled_re") or r.get("rx")
            if r.get("secret_group", 0) == 0 and getattr(rx, "groups", 0) == 1:
                r["secret_group"] = 1
    _G.update(mod=mod, ctx=ctx, kw=kw, nokw=nokw)


def _scan_one(args):
    path, root = args
    try:
        return [(f["file"], f["line"], f["score"], f["reason"])
                for f in _G["mod"].scan_file(path, root, _G["ctx"], _G["kw"], _G["nokw"])]
    except Exception:
        return []


def run_ours(root, scan_dir, scanner_dir, fixed=False, jobs=None, keep=None, **_):
    scan_py = os.path.join(scanner_dir, "scan.py")
    rules = os.path.join(scanner_dir, "rules.toml")
    files = []
    for dp, dn, fns in os.walk(scan_dir):
        dn[:] = [d for d in dn if d not in (".git", "node_modules")]
        files += [(os.path.join(dp, fn), root) for fn in fns
                  if keep is None or keep(os.path.relpath(os.path.join(dp, fn), root).replace("\\", "/"))]
    t0 = time.time()
    out = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=jobs, initializer=_init,
                                                initargs=(scan_py, rules, fixed)) as ex:
        for res in ex.map(_scan_one, files, chunksize=32):
            out.extend(res)
    dt = time.time() - t0
    # scan_file returns paths relative to `root` already
    return [(f.replace("\\", "/"), l, s, r) for f, l, s, r in out], dt


# ----------------------------------------------------------------------------
# helpers for reporting
# ----------------------------------------------------------------------------
def best_per_line(findings, min_score=0.0):
    best = {}
    for f, l, s, r in findings:
        if s >= min_score and ((f, l) not in best or s > best[(f, l)][0]):
            best[(f, l)] = (s, r)
    return best


def sweep(findings, scorer, thresholds):
    best = best_per_line(findings)
    out = []
    for t in thresholds:
        hits = {k for k, (s, _) in best.items() if s >= t}
        m = scorer.evaluate(hits)
        out.append(dict(threshold=t, **m))
    return out


def top_fp_reasons(findings, scorer, min_score, top=10):
    cnt = collections.Counter()
    for key, (s, r) in best_per_line(findings, min_score).items():
        if scorer.classify(key) in ("FPL", "FPU"):
            cnt[r.split(":")[0] + ":" + r.split(":")[1] if r.startswith("rule:") else r.split(":")[0]] += 1
    return cnt.most_common(top)


def md_table(results):
    hdr = ("| Tool | Prec (labeled) | Recall | F1 (labeled) | Prec (strict) | F1 (strict) "
           "| Hits | Time (s) |\n|---|---|---|---|---|---|---|---|\n")
    body = ""
    for name, r in results.items():
        m = r["metrics"]
        body += (f"| {name} | {m['precision_labeled']:.3f} | {m['recall']:.3f} | "
                 f"{m['f1_labeled']:.3f} | {m['precision_strict']:.3f} | {m['f1_strict']:.3f} | "
                 f"{m['hits']} | {r['seconds']:.1f} |\n")
    return hdr + body


def make_chart(path, title, results, sweep_rows):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed - skipping chart)")
        return
    names = list(results)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    w = 0.26
    cols = {"Precision": "#3b5b92", "Recall": "#2a9d8f", "F1": "#e9a23b"}
    for j, (lab, key) in enumerate([("Precision", "precision_labeled"), ("Recall", "recall"),
                                    ("F1", "f1_labeled")]):
        xs = [i + (j - 1) * w for i in range(len(names))]
        vals = [results[n]["metrics"][key] for n in names]
        a1.bar(xs, vals, w, label=lab, color=cols[lab])
        for x, v in zip(xs, vals):
            a1.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=7)
    a1.set_xticks(range(len(names)))
    a1.set_xticklabels(names, rotation=18, ha="right", fontsize=8)
    a1.set_ylim(0, 1.12)
    a1.set_title("Labeled-lines protocol")
    a1.legend(fontsize=8)
    for tag, rows in sweep_rows.items():
        rows = sorted(rows, key=lambda r: (r["recall"], r["precision_labeled"]))
        a2.plot([r["recall"] for r in rows], [r["precision_labeled"] for r in rows],
                marker=".", label=f"{tag} (threshold sweep)")
    for n in names:
        if not n.startswith("ours"):
            m = results[n]["metrics"]
            a2.scatter([m["recall"]], [m["precision_labeled"]], s=60, marker="s", label=n, zorder=3)
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision (labeled)")
    a2.set_xlim(0, 1.02); a2.set_ylim(0, 1.02)
    a2.set_title("Precision-recall"); a2.legend(fontsize=7); a2.grid(alpha=.3)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# repo-level dev/test split for CredData (deterministic; 60% dev / 40% test)
# ----------------------------------------------------------------------------
def repo_of(path):
    parts = path.split("/")
    return parts[1] if len(parts) > 2 and parts[0] == "data" else parts[0]


def split_of(repo):
    import zlib
    return "test" if zlib.crc32(repo.encode()) % 5 in (0, 1) else "dev"


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def evaluate_dataset(name, args):
    print(f"\n=== {name} ===")
    if name == "synthetic":
        root = os.path.abspath(args.synthetic_dir)
        rows = load_synthetic(root)
        scan_dir = root
        scope = args.strict_scope or "all-files"
    else:
        root = os.path.abspath(args.creddata_dir)
        rows = load_creddata(root)
        scan_dir = os.path.join(root, "data")
        scope = args.strict_scope or "labeled-files"
    drop = {c.strip() for c in args.drop_categories.split(",") if c.strip()}
    if drop and name == "creddata":
        n0 = len(rows)
        rows = [r for r in rows if not (r["gt"] == "T" and r["cat"] in drop)]
        print(f"  dropped {n0 - len(rows)} true rows with category in {sorted(drop)}")
    if name == "creddata" and args.repo_split != "all":
        rows = [r for r in rows if split_of(repo_of(r["file"])) == args.repo_split]
        print(f"  repo split = {args.repo_split}: {len({repo_of(r['file']) for r in rows})} repos")
        in_split = lambda f: split_of(repo_of(f)) == args.repo_split
    elif name == "synthetic" and args.synthetic_split != "all":
        rows = [r for r in rows if r["file"].startswith(args.synthetic_split + "/")]
        in_split = lambda f: f.startswith(args.synthetic_split + "/") or f == "labels.csv"
    else:
        in_split = lambda f: True
    scorer = Scorer(rows, args.tolerance, args.x_policy, scope)
    sfx = "" if (args.repo_split == "all" or name != "creddata") and (
        args.synthetic_split == "all" or name != "synthetic") else "_" + (
        args.repo_split if name == "creddata" else args.synthetic_split)
    scan_paths = None
    if name == "creddata" and args.repo_split != "all":
        scan_paths = sorted(os.path.join(scan_dir, d) for d in os.listdir(scan_dir)
                            if split_of(d) == args.repo_split)
    if name == "synthetic":
        scorer.ignore_files = {"labels.csv"}
    n_t = len(scorer.t_rows)
    print(f"  ground truth: {len(rows)} rows ({n_t} T, "
          f"{sum(r['gt']=='F' for r in rows)} F, {sum(r['gt']=='X' for r in rows)} X), "
          f"{len(scorer.labeled_files)} labeled files | tolerance={args.tolerance} "
          f"x_policy={args.x_policy} strict_scope={scope}")
    rawdir = os.path.join(args.out, "raw", name if name == "synthetic" else
                          f"{name}_{os.path.basename(root.rstrip('/'))}")
    os.makedirs(rawdir, exist_ok=True)

    runs = {}   # display name -> (findings, seconds)

    def cached(tag, fn):
        p = os.path.join(rawdir, f"{tag}.json")
        if args.reuse_raw and os.path.exists(p):
            d = json.load(open(p))
            print(f"  [{tag}] reused cached output ({len(d['findings'])} findings)")
            return [tuple(x) for x in d["findings"]], d["seconds"]
        print(f"  [{tag}] running ...", flush=True)
        findings, dt = fn()
        json.dump(dict(seconds=dt, findings=findings), open(p, "w"))
        print(f"  [{tag}] {len(findings)} findings in {dt:.1f}s")
        return findings, dt

    own_th = {}
    tools = args.tools
    scanner_dir = os.path.abspath(args.scanner_dir)
    if "ours" in tools:
        keep = None if in_split is None else (lambda f: in_split(f))
        runs["ours"] = cached("ours" + sfx, lambda: run_ours(root, scan_dir, scanner_dir, False, args.jobs, keep))
        for spec in args.extra_scanner:
            nm, d = spec.split("=", 1)
            nm = nm if nm.startswith("ours") else "ours-" + nm
            mp_ = os.path.join(d, "model_v2.json")          # a scanner dir may ship its own thresholds
            if os.path.exists(mp_):
                th = json.load(open(mp_)).get("thresholds", {})
                own_th[nm] = (th.get("block", args.block_at), th.get("warn", args.warn_at))
            runs[nm] = cached(f"extra_{nm}" + sfx, lambda d=d: run_ours(root, scan_dir, os.path.abspath(d), False, args.jobs, keep))
        if not args.no_fixed:
            runs["ours+fix"] = cached("ours_fix" + sfx, lambda: run_ours(root, scan_dir, scanner_dir, True, args.jobs, keep))
    for tool, runner, binname in (("gitleaks", run_gitleaks, "gitleaks"),
                                  ("trufflehog", run_trufflehog, "trufflehog")):
        if tool in tools:
            b = find_bin(binname, args.bin_dir)
            if not b and not (args.reuse_raw and os.path.exists(os.path.join(rawdir, f"{tool}.json"))):
                print(f"  [{tool}] binary not found (run evaluation/setup_tools.sh) - skipped")
                continue
            runs[tool] = cached(tool, lambda r=runner, b=b: r(b, root, scan_dir, rawdir))
    if "credsweeper" in tools:
        b = find_bin("credsweeper", args.bin_dir) or find_bin("credsweeper", os.path.dirname(sys.executable))
        if b or (args.reuse_raw and os.path.exists(os.path.join(rawdir, f"credsweeper{sfx}.json"))):
            runs["credsweeper"] = cached("credsweeper" + sfx, lambda: run_credsweeper(b, root, scan_dir, rawdir, paths=scan_paths,
                                                                                 jobs=args.jobs or 4))
        else:
            print("  [credsweeper] not installed (pip install credsweeper) - skipped")
    if "detect-secrets" in tools:
        try:
            import detect_secrets  # noqa
            have = True
        except ImportError:
            have = False
        if have or (args.reuse_raw and os.path.exists(os.path.join(rawdir, f"detect-secrets{sfx}.json"))):
            runs["detect-secrets"] = cached("detect-secrets" + sfx, lambda: run_detect_secrets(root, scan_dir, scan_paths))
        else:
            print("  [detect-secrets] not installed (pip install detect-secrets==1.5.0) - skipped")

    results, sweeps = {}, {}
    for name_, (findings, dt) in runs.items():
        findings = [x for x in findings if in_split(x[0])]
        if name_.startswith("ours"):
            best = best_per_line(findings)
            blk, wrn = own_th.get(name_, (args.block_at, args.warn_at))
            for label, thr in ((f"BLOCK>={blk}", blk), (f"WARN+BLOCK>={wrn}", wrn)):
                hits = {k for k, (s, _) in best.items() if s >= thr}
                results[f"{name_} [{label}]"] = dict(
                    metrics=scorer.evaluate(hits), seconds=dt,
                    categories=scorer.per_category(hits),
                    top_fp=top_fp_reasons(findings, scorer, thr))
                if label.startswith("WARN"):
                    results[f"{name_} [{label}]"]["is_warn"] = True
            sweeps[name_] = sweep(findings, scorer, [round(0.05 * i, 2) for i in range(1, 21)])
        else:
            hits = {(f, l) for f, l, *_ in findings}
            results[name_] = dict(metrics=scorer.evaluate(hits), seconds=dt,
                                  categories=scorer.per_category(hits),
                                  top_fp=top_fp_reasons(findings, scorer, 0.0))

    table = md_table(results)
    print("\n" + table)
    print("Top false-positive sources for 'ours [WARN+BLOCK]':",
          next((v["top_fp"] for k, v in results.items() if k.startswith("ours") and v.get("is_warn")), "n/a"))

    os.makedirs(args.out, exist_ok=True)
    tag = name if args.repo_split == "all" or name != "creddata" else f"{name}_{args.repo_split}"
    json.dump(dict(dataset=name, repo_split=args.repo_split, tolerance=args.tolerance, x_policy=args.x_policy,
                   strict_scope=scope, results=results, ours_threshold_sweep=sweeps),
              open(os.path.join(args.out, f"{tag}_results.json"), "w"), indent=1)
    with open(os.path.join(args.out, f"{tag}_results.md"), "w") as f:
        f.write(f"# {name}\n\ntolerance={args.tolerance}, x_policy={args.x_policy}, "
                f"strict_scope={scope}\n\n{table}\n")
        for n_, r in results.items():
            f.write(f"\n## {n_}\n\nSecret recall by category (category, support, recall): "
                    f"{r['categories']['secret_recall_by_category']}\n\n"
                    f"Non-secret flag rate by category (category, support, flagged): "
                    f"{r['categories']['nonsecret_flag_rate_by_category']}\n")
    make_chart(os.path.join(args.out, f"{tag}_chart.png"),
               f"{name}: scanner comparison (tolerance={args.tolerance})", results, sweeps)
    print(f"  wrote {args.out}/{tag}_results.json, {tag}_results.md, {tag}_chart.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["synthetic", "creddata", "both"], default="both")
    ap.add_argument("--tools", default=",".join(ALL_TOOLS),
                    help="comma list from: " + ",".join(ALL_TOOLS))
    ap.add_argument("--synthetic-dir", default=os.path.join(ROOT, "dataset", "generated"))
    ap.add_argument("--creddata-dir", default=os.path.join(ROOT, "datasets", "CredData"))
    ap.add_argument("--scanner-dir", default=os.path.join(ROOT, "scanner"))
    ap.add_argument("--bin-dir", default=os.path.join(ROOT, "tools", "bin"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--tolerance", type=int, default=0, help="+-K line tolerance for T rows (default 0)")
    ap.add_argument("--x-policy", choices=["negative", "ignore"], default="negative",
                    help="CredData 'X' (unknown) rows: count flags as FP (default, CredData's own "
                         "convention) or ignore them")
    ap.add_argument("--strict-scope", choices=["all-files", "labeled-files"], default=None,
                    help="files whose unlabeled lines count in the strict protocol "
                         "(default: all-files for synthetic, labeled-files for creddata)")
    ap.add_argument("--block-at", type=float, default=0.70)
    ap.add_argument("--warn-at", type=float, default=0.40)
    ap.add_argument("--drop-categories", default="",
                    help="comma list of CredData Category values whose TRUE rows are dropped from "
                         "the ground truth, e.g. UUID (an identifier is not an IaC secret)")
    ap.add_argument("--synthetic-split", choices=["all", "train", "test"], default="all",
                    help="synthetic data: score only the train or test families")
    ap.add_argument("--repo-split", choices=["all", "dev", "test"], default="all",
                    help="CredData only: score on a repo-level 60%% dev / 40%% test split")
    ap.add_argument("--extra-scanner", action="append", default=[], metavar="NAME=DIR",
                    help="extra scanner dir with scan.py (+rules.toml), e.g. v2=scanner_v2 (repeatable)")
    ap.add_argument("--no-fixed", action="store_true",
                    help="skip the 'ours+fix' variant (in-memory secret_group patch)")
    ap.add_argument("--reuse-raw", action="store_true", help="re-score cached tool output")
    ap.add_argument("--jobs", type=int, default=None, help="worker processes for our scanner")
    args = ap.parse_args()
    args.tools = [t.strip() for t in args.tools.split(",") if t.strip()]
    bad = [t for t in args.tools if t not in ALL_TOOLS]
    if bad:
        sys.exit(f"unknown tool(s): {bad}")
    for ds in (["synthetic", "creddata"] if args.dataset == "both" else [args.dataset]):
        try:
            evaluate_dataset(ds, args)
        except FileNotFoundError as e:
            print(f"\n=== {ds} === skipped: {e}")


if __name__ == "__main__":
    main()
