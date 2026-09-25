#!/usr/bin/env python3
"""train_codebert.py - fine-tune a small code encoder as a line-level secret classifier.

Input per candidate line:  "<filename> </s> <context before> </s> >> <target line> </s> <context after>"
Train on split=train only; pick the decision threshold on split=val; report/predict val, test, test_ood.
--mask-key replaces the key name on the target line by KEY (context-reliance ablation: how much does the
model lean on names like `password`?). --dry-run prints dataset sizes + samples and needs no torch.

usage: python train_codebert.py --data out/tierB,out/gen,out/tierB_edits --out runs/codebert --epochs 3
       python train_codebert.py --data ... --out runs/codebert_masked --mask-key
"""
import argparse
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import build_candidates, file_lines, load_dirs  # noqa: E402
from mutate_real import parse_line  # noqa: E402


def make_text(lines, c, relpath, ctx, mask_key):
    n = c["line_no"]
    tgt = lines[n - 1]
    if mask_key:
        p = parse_line(c["fmt"], tgt)
        if p:
            tgt = tgt.replace(p["key"], "KEY", 1)
    before = " ".join(x.strip() for x in lines[max(0, n - 1 - ctx):n - 1])
    after = " ".join(x.strip() for x in lines[n:n + ctx])
    return f"{os.path.basename(relpath)} </s> {before} </s> >> {tgt.strip()} </s> {after}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="microsoft/codebert-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--ctx", type=int, default=3)
    ap.add_argument("--neg-per-file", type=int, default=10)
    ap.add_argument("--neg-ratio", type=float, default=4.0, help="unlabeled negatives kept per positive (hard negatives always kept)")
    ap.add_argument("--mask-key", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    idx, labs = load_dirs(a.data.split(","))
    cands = build_candidates(idx, labs, a.neg_per_file, a.seed)
    cache = {}
    for c in cands:
        f = idx[c["file_id"]]
        if c["file_id"] not in cache:
            cache[c["file_id"]] = file_lines(f)
        c["text"] = make_text(cache[c["file_id"]], c, f["relpath"], a.ctx, a.mask_key)
    by_split = {s: [c for c in cands if c["split"] == s] for s in ("train", "val", "test", "test_ood")}
    train = by_split["train"]
    pos = [c for c in train if c["y"] == 1]
    hard = [c for c in train if c["y"] == 0 and c["category"] != "unlabeled"]
    easy = [c for c in train if c["y"] == 0 and c["category"] == "unlabeled"]
    rng.shuffle(easy)
    train = pos + hard + easy[:int(a.neg_ratio * max(1, len(pos)))]
    rng.shuffle(train)
    print({s: len(v) for s, v in by_split.items()}, "| train used:", len(train), f"(pos {len(pos)}, hard neg {len(hard)})")
    if a.dry_run:
        for c in train[:3]:
            print(c["y"], c["category"], "|", c["text"][:200])
        return

    import numpy as np
    import torch
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, num_labels=2).to(dev)

    class DS(Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        enc = tok([b["text"] for b in batch], truncation=True, max_length=a.max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([b["y"] for b in batch])
        return enc

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    dl = DataLoader(DS(train), batch_size=a.bs, shuffle=True, collate_fn=collate)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * len(dl) * a.epochs), len(dl) * a.epochs)
    use_bf16 = dev == "cuda" and torch.cuda.is_bf16_supported()

    def predict(rows):
        model.eval()
        out = []
        with torch.no_grad():
            for b in DataLoader(DS(rows), batch_size=a.bs * 2, collate_fn=collate):
                b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
                with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=use_bf16):
                    logits = model(**b).logits
                out += torch.softmax(logits.float(), -1)[:, 1].cpu().tolist()
        return out

    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for b in dl:
            b = {k: v.to(dev) for k, v in b.items()}
            with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            tot += loss.item()
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(dl):.4f}")

    val = by_split["val"]
    thr = 0.5
    if val:
        sv = np.array(predict(val))
        yv = np.array([c["y"] for c in val])
        best = max(np.linspace(0.05, 0.95, 91), key=lambda t: precision_recall_fscore_support(yv, sv >= t, average="binary", zero_division=0)[2])
        thr = float(best)
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for s in ("val", "test", "test_ood"):
        part = by_split[s]
        if not part:
            continue
        sc = predict(part)
        rows += [(c["file_id"], c["line_no"], round(x, 5)) for c, x in zip(part, sc)]
        y = np.array([c["y"] for c in part])
        p, r, f1, _ = precision_recall_fscore_support(y, np.array(sc) >= thr, average="binary", zero_division=0)
        print(f"{s:9s} line-level P={p:.3f} R={r:.3f} F1={f1:.3f} AP={average_precision_score(y, sc) if y.sum() else float('nan'):.3f}  (thr {thr:.2f})")
    with open(os.path.join(a.out, "predictions.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file_id", "line_no", "score"])
        w.writerows(rows)
    json.dump({"threshold": thr, "model": a.model, "mask_key": a.mask_key}, open(os.path.join(a.out, "run.json"), "w"))
    print("wrote", os.path.join(a.out, "predictions.csv"))


if __name__ == "__main__":
    main()
