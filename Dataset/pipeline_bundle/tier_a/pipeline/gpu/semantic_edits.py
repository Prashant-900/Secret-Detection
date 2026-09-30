#!/usr/bin/env python3
"""semantic_edits.py - LLM performs realistic edits on already-labeled files; labels are RE-DERIVED by string search.

The model is told to keep every labeled literal verbatim. We then verify (never trust):
  1. every labeled literal still appears in the new file
  2. the file still parses and each literal is still reachable in the parsed structure
  3. the model did not invent new high-entropy literals (they would be unlabeled secrets)
  4. the file did not shrink drastically
Labels are re-derived from where the known literal now sits (new line number, new key name, new key_hint),
so the model's edit changes the CONTEXT of a secret, never its label. The split is inherited from the parent
file, so there is no leakage between train and test.

usage: python semantic_edits.py --data out/tierB --out out/tierB_edits --n-files 600 --model <model>
"""
import argparse
import csv
import hashlib
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import secrets_lib as S  # noqa: E402
from common import file_lines, load_dirs  # noqa: E402
from llm_backend import add_backend_args, make_backend  # noqa: E402
from mutate_real import contains_string, parse_line, parse_ok  # noqa: E402

INSTR = {
    "env_block": "Move the credentials into an environment-variable style block or section that is natural for this file format.",
    "rename_keys": "Rename the keys that hold credentials to less obvious, application-specific names.",
    "comment_refactor": "Add short explanatory comments for each section and reorder the keys in a more logical order.",
    "extract_vars": "Extract repeated configuration into variables / anchors / locals where this format allows it.",
    "duplicate_env": "Add a second environment (for example staging) that mirrors the existing configuration and reuses the same literal values.",
    "inline_style": "Convert multi-line structures to a more compact inline style wherever that is valid.",
    "add_section": "Add one additional realistic resource / service / section that uses the existing configuration.",
}
SYSTEM = "You are a careful infrastructure engineer editing a configuration file. Return only the complete edited file."
LONG_TOKEN = re.compile(r"[A-Za-z0-9+/=_\-]{24,}")
FILE_OPEN, FILE_CLOSE = "<<<FILE", "FILE>>>"


def prompt_for(instr_text, fmt, text, needles):
    lst = "\n".join(f"- {n}" for n in needles)
    return (f"Edit the following {fmt.upper()} file.\n\nTask: {instr_text}\n\n"
            "Rules:\n- Return the COMPLETE edited file, no markdown fences, no commentary.\n"
            "- These exact literal strings must appear in your output, character for character (do not split, re-encode, "
            f"shorten, or remove them):\n{lst}\n"
            "- Do NOT add any new passwords, tokens, API keys or other high-entropy literal values.\n\n"
            f"{FILE_OPEN}\n{text}\n{FILE_CLOSE}")


def mock_responder(system, user):
    body = user.split(FILE_OPEN + "\n", 1)[1].rsplit("\n" + FILE_CLOSE, 1)[0]
    return body if body.lstrip().startswith("{") else "# reviewed\n" + body


def main():
    ap = argparse.ArgumentParser()
    add_backend_args(ap)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-files", type=int, default=600)
    ap.add_argument("--instr-per-file", type=int, default=1)
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    idx, labs = load_dirs([a.data])
    by_file = defaultdict(list)
    for l in labs:
        by_file[l["file_id"]].append(l)

    jobs = []
    for fid, ls in by_file.items():
        f = idx[fid]
        if f["fmt"] not in ("yaml", "hcl", "json", "env") or fid.endswith(("__secret", "__reference", "__placeholder")):
            continue
        lines = file_lines(f)
        text = "\n".join(lines)
        per_span = Counter(l["span_id"] for l in ls)
        needles, ok = [], True
        for l in ls:
            ln = lines[int(l["line_no"]) - 1]
            n = ln[int(l["span_start"]):int(l["span_end"])]
            if per_span[l["span_id"]] != 1 or len(n) < 8 or text.count(n) != 1:
                ok = False
                break
            needles.append((n, l))
        if ok and needles:
            jobs.append((fid, f, text, needles))
    rng.shuffle(jobs)
    jobs = jobs[:a.n_files]
    print(f"{len(jobs)} eligible parent files")

    reqs = []
    for fid, f, text, needles in jobs:
        for iid in rng.sample(sorted(INSTR), min(a.instr_per_file, len(INSTR))):
            reqs.append((fid, f, text, needles, iid, prompt_for(INSTR[iid], f["fmt"], text, [n for n, _ in needles])))
    be = make_backend(a, mock_responder)
    outs = be.chat([(SYSTEM, r[-1]) for r in reqs], max_tokens=4096, temperature=0.7, top_p=0.95, seed=a.seed)

    os.makedirs(os.path.join(a.out, "files"), exist_ok=True)
    label_rows, index_rows, fail = [], [], Counter()
    for (fid, f, text, needles, iid, _), raw in zip(reqs, outs):
        new = re.sub(r"^```[\w+-]*\n", "", raw.strip("\n"))
        new = re.sub(r"\n```\s*$", "", new)
        if len(new.split("\n")) < 0.5 * len(text.split("\n")):
            fail["shrunk"] += 1
            continue
        if any(n not in new for n, _ in needles):
            fail["literal_changed"] += 1
            continue
        ok, parsed = parse_ok(f["fmt"], new)
        if not ok:
            fail["parse_fail"] += 1
            continue
        if parsed is not None and not all(contains_string(parsed, n) for n, l in needles if not int(l["commented"])):
            fail["value_not_reachable"] += 1
            continue
        old_tokens = set(LONG_TOKEN.findall(text))
        invented = [t for t in set(LONG_TOKEN.findall(new)) - old_tokens
                    if re.search(r"\d", t) and re.search(r"[A-Za-z]", t) and not any(t in n or n in t for n, _ in needles)]
        if invented:
            fail["invented_literal"] += 1
            continue
        nid = f"{fid}__e{hashlib.sha1((iid + new).encode()).hexdigest()[:6]}"
        outp = os.path.join(a.out, "files", nid, f["relpath"])
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        open(outp, "w", encoding="utf-8").write(new)
        meta = dict(file_id=nid, repo=f["repo"], family=f["family"], split=f["split"], relpath=f["relpath"], fmt=f["fmt"])
        new_lines = new.split("\n")
        n_lab = 0
        for n, l in needles:
            for i, ln in enumerate(new_lines, 1):
                s = ln.find(n)
                if s < 0:
                    continue
                p = parse_line(f["fmt"], ln)
                key = p["key"] if p else ""
                label_rows.append(dict(meta, line_no=i, label=l["label"], category=l["category"], secret_type=l["secret_type"],
                                       operator=f"llm_edit:{iid}", key_name=key,
                                       key_hint="sensitive" if S.is_sensitive_key(key) else "neutral", encoding=l["encoding"],
                                       commented=int(ln.lstrip().startswith(("#", "//"))), span_id=len(label_rows) + 1,
                                       span_start=s, span_end=s + len(n), pair_id=fid, variant=l["operator"]))
                n_lab += 1
        index_rows.append(dict(meta, n_lines=len(new_lines), n_labeled=n_lab, validated=1))

    cols = ["file_id", "repo", "family", "split", "relpath", "fmt", "line_no", "label", "category", "secret_type", "operator",
            "key_name", "key_hint", "encoding", "commented", "span_id", "span_start", "span_end", "pair_id", "variant"]
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(label_rows)
    with open(os.path.join(a.out, "files_index.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_id", "repo", "family", "split", "relpath", "fmt", "n_lines", "n_labeled", "validated"])
        w.writeheader()
        w.writerows(index_rows)
    print(f"kept {len(index_rows)} edited files / {len(label_rows)} labels; rejected: {dict(fail)}")


if __name__ == "__main__":
    main()
