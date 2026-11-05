#!/usr/bin/env python3
"""Build an offline, evidence-based report from the frozen experiment artifacts.

This program never fits a model, selects a new threshold, or runs inference.
Only the project's dependency-free span evaluator is imported for consistency
checks. Source text, literals, model input, and raw tool reports are not exported.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import html
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys

BUNDLE = Path(__file__).resolve().parent
REAL = {"creddata", "reviewed_real"}
SCORES = ("precision", "recall", "f1")
SPAN_FIELDS = ("documents", "positive_spans", "negative_spans", "true_positive",
               "false_positive", "false_negative", "precision", "recall", "f1",
               "explicit_negative_alerts", "duplicate_positive_alerts", "unknown_alerts",
               "unknown_alerts_per_kloc", "negative_flag_rate", "candidate_recall_ceiling",
               "candidate_spans", "alerts", "lines")
SAFE_PREDICTION = ("id", "document_id", "repo_id", "group_id", "split", "source",
                   "language", "start", "end", "label", "category", "score")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def slug(value):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_") or "item"


def fmt(value):
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def cell(value):
    return html.escape(fmt(value)).replace("|", "&#124;").replace("\n", " ")


def table(rows, columns):
    if not rows:
        return "No observations available.\n"
    return ("| " + " | ".join(cell(label) for _, label in columns) + " |\n| " +
            " | ".join("---" for _ in columns) + " |\n" +
            "\n".join("| " + " | ".join(cell(row.get(key)) for key, _ in columns) + " |" for row in rows) + "\n")


def csv_table(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # Avoid spreadsheet formula execution in labels from data artifacts.
            safe = {k: ("'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v)
                    for k, v in row.items()}
            writer.writerow(safe)


class Inputs:
    def __init__(self, root, partial):
        self.root, self.partial = root, partial
        self.inventory, self.missing = {}, []

    def path(self, name):
        p = Path(name)
        return p.resolve() if p.is_absolute() else (self.root / p).resolve()

    def label(self, p):
        try:
            return p.relative_to(self.root).as_posix()
        except ValueError:
            return str(p)

    def available(self, name, required=False):
        p = self.path(name)
        if p.is_file():
            return p
        entry = {"path": self.label(p), "required": required}
        if entry not in self.missing:
            self.missing.append(entry)
        if required and not self.partial:
            raise ValueError(f"Required artifact missing: {p}. Use --allow-partial only for an explicitly incomplete report.")
        return None

    def track(self, p):
        name = self.label(p)
        if name not in self.inventory:
            self.inventory[name] = {"path": name, "bytes": p.stat().st_size, "sha256": digest(p)}

    def json(self, name, required=False):
        p = self.available(name, required)
        if p is None:
            return None
        self.track(p)
        try:
            return json.loads(p.read_text(encoding="utf-8-sig"))
        except (ValueError, OSError) as exc:
            raise ValueError(f"Cannot read JSON artifact {p}: {type(exc).__name__}") from exc

    def rows(self, name, required=False):
        p = self.available(name, required)
        if p is None:
            return
        self.track(p)
        with p.open(encoding="utf-8-sig") as fh:
            for line_number, line in enumerate(fh, 1):
                if line.strip():
                    try:
                        row = json.loads(line)
                    except ValueError as exc:
                        raise ValueError(f"Invalid JSON at {p}:{line_number}") from exc
                    if not isinstance(row, dict):
                        raise ValueError(f"Expected an object at {p}:{line_number}")
                    yield row


def verify_metrics(stored, computed, label):
    """Reject stale metrics; compare every reported stratum, not only headline F1."""
    def compare(a, b, where):
        for key in SPAN_FIELDS:
            if key not in a:
                raise ValueError(f"Missing stored metric {label}/{where}/{key}")
            left, right = a[key], b[key]
            equal = left == right
            if isinstance(left, (int, float)) and isinstance(right, (int, float)):
                equal = math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-10)
            if not equal:
                raise ValueError(f"Saved metrics disagree with predictions/documents: {label}/{where}/{key}")
    compare(stored["overall"], computed["overall"], "overall")
    compare(stored["headline_real_test"], computed["headline_real_test"], "headline_real_test")
    for axis in ("source", "language", "split"):
        field = "by_" + axis
        if set(stored[field]) != set(computed[field]):
            raise ValueError(f"Saved group identities disagree: {label}/{field}")
        for group in computed[field]:
            compare(stored[field][group], computed[field][group], field + "/" + group)


def check_predictions(rows, documents, candidates, label, real_only=False):
    docs = {d["id"]: d for d in documents}
    expected = {cid: r for cid, r in candidates.items() if r["document_id"] in docs}
    if len({r.get("id") for r in rows}) != len(rows):
        raise ValueError(f"Duplicate prediction identifiers in {label}")
    if candidates and {r.get("id") for r in rows} != set(expected):
        raise ValueError(f"Prediction candidate population differs from frozen input in {label}")
    for row in rows:
        doc = docs.get(row.get("document_id"))
        if doc is None:
            raise ValueError(f"Prediction references a different document population in {label}")
        for key in ("split", "source", "repo_id", "group_id", "language"):
            if row.get(key) != doc.get(key):
                raise ValueError(f"Prediction/document {key} mismatch in {label}")
        if candidates:
            c = expected[row["id"]]
            if any(row.get(k) != c.get(k) for k in ("document_id", "start", "end", "label")):
                raise ValueError(f"Prediction/candidate identity or label mismatch in {label}")


def describe_corpus(documents, candidates):
    if not documents:
        return []
    groups = {}
    for doc in documents:
        key = (doc["split"], doc["source"], doc["language"])
        if key not in groups:
            groups[key] = {"split": key[0], "source": key[1], "language": key[2], "documents": 0,
                           "positive_spans": 0, "negative_spans": 0, "lines": 0,
                           "candidates_positive": 0, "candidates_negative": 0, "candidates_unknown": 0,
                           "repos": set(), "owners": set()}
        r = groups[key]
        r["documents"] += 1
        r["lines"] += max(1, len(doc["text"].splitlines()))
        r["positive_spans"] += sum(s["label"] == 1 for s in doc["spans"])
        r["negative_spans"] += sum(s["label"] == 0 for s in doc["spans"])
        r["repos"].add(doc["repo_id"])
        r["owners"].add(doc["group_id"])
    for row in candidates.values():
        key = (row["split"], row["source"], row["language"])
        if key not in groups:
            raise ValueError("Candidate has no matching corpus stratum")
        label = row.get("label")
        if label not in (0, 1, None):
            raise ValueError("Unexpected candidate label")
        groups[key]["candidates_" + ({0: "negative", 1: "positive", None: "unknown"}[label])] += 1
    return [{**{k: v for k, v in r.items() if k not in ("repos", "owners")},
             "repositories": len(r["repos"]), "owners": len(r["owners"])} for _, r in sorted(groups.items())]


def load_selected_predictions(inputs, selected, suite):
    members = selected.get("models", [])
    if not members:
        raise ValueError("Selected model has no members")
    sets = []
    for member in members:
        model_path = (suite / member["path"]).resolve()
        # Recorded model paths are relative to the suite, so copied runs remain portable.
        if not model_path.is_relative_to(suite.resolve()):
            raise ValueError("Selected model path escapes the experiment directory")
        prediction_path = model_path.parent / "real_validation_all_predictions.jsonl"
        if inputs.available(prediction_path, required=True) is None:
            return None
        meta = inputs.json(model_path.parent / "real_validation_all_predictions.meta.json")
        if meta and (meta.get("predictions_sha256") != digest(prediction_path) or
                     meta.get("documents_sha256") != selected.get("documents_sha256")):
            raise ValueError("Selected member validation prediction cache hash/provenance differs")
        sets.append([{k: r[k] for k in SAFE_PREDICTION if k in r} for r in inputs.rows(prediction_path)])
    weights = [float(m.get("weight", 1)) for m in members]
    if any(not math.isfinite(w) or w <= 0 for w in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("Selected model weights must be positive and sum to one")
    maps = [{r["id"]: r for r in rows} for rows in sets]
    if any(len(m) != len(rows) or set(m) != set(maps[0]) for m, rows in zip(maps, sets)):
        raise ValueError("Selected member predictions do not have identical unique candidates")
    result = []
    for row in sets[0]:
        if any(any(m[row["id"]].get(k) != row.get(k) for k in SAFE_PREDICTION if k != "score") for m in maps[1:]):
            raise ValueError("Selected member prediction metadata differs")
        score = (sum(m[row["id"]]["score"] for m in maps) / len(maps) if len(set(weights)) == 1 else
                 sum(w * m[row["id"]]["score"] for w, m in zip(weights, maps)))
        result.append({**row, "score": score})
    return result


def diagnostics(name, predictions, threshold, plots, out, candidate_statistics):
    stats = candidate_statistics(predictions, threshold)
    csv_table(out / "tables" / f"{name}_candidate_metrics.csv", [{"metric": k, "value": v} for k, v in stats["metrics"].items()])
    scope = "Explicitly labeled proposed candidates only; excludes unknowns and secrets missed by candidate extraction."
    m = stats["metrics"]
    plots.confusion(name + "_confusion", name.replace("_", " ") + " | candidate confusion matrix",
                    m["tn"], m["fp"], m["fn"], m["tp"], caption=scope)
    for key, xlabel, ylabel, diag in (("precision_recall", "Recall", "Precision", False), ("roc", "False positive rate", "True positive rate", True)):
        curve = stats["curves"][key]
        if curve["x"] and (key != "roc" or m["roc_auc"] is not None):
            plots.curve(name + "_" + key, name.replace("_", " ") + " | candidate " + key.replace("_", " "),
                        curve["x"], curve["y"], xlabel, ylabel, caption=scope, diagonal=diag)
            csv_table(out / "tables" / f"{name}_{key}.csv", [{"x": x, "y": y} for x, y in zip(curve["x"], curve["y"])])
    curve = stats["curves"]["thresholds"]
    if curve["x"]:
        plots.lines(name + "_thresholds", name.replace("_", " ") + " | candidate threshold sensitivity", curve["x"],
                    {"Precision": curve["y_precision"], "Recall": curve["y_recall"], "F1": curve["y_f1"]},
                    xlabel="Score threshold (descriptive only)", ylabel="Score", bounded=True,
                    caption=scope + f" The operating threshold remains {threshold:.6g}; this plot does not retune it.")
        csv_table(out / "tables" / f"{name}_threshold_sensitivity.csv", [dict(threshold=x, precision=p, recall=r, f1=f) for x, p, r, f in
                  zip(curve["x"], curve["y_precision"], curve["y_recall"], curve["y_f1"])])
    plots.reliability(name + "_reliability", name.replace("_", " ") + " | candidate reliability", stats["reliability"], caption=scope + " Bins measure score reliability; no probability recalibration is fitted.")
    csv_table(out / "tables" / f"{name}_reliability.csv", stats["reliability"])
    plots.histogram(name + "_scores", name.replace("_", " ") + " | candidate score distribution",
                    [p["score"] for p in predictions if p.get("label") == 0],
                    [p["score"] for p in predictions if p.get("label") == 1], caption=scope)
    return stats


def span_tables(name, evaluation, plots, out):
    rows = []
    for axis in ("source", "language"):
        for key, values in evaluation["by_" + axis].items():
            rows.append({"axis": axis, "group": key, **{k: values.get(k) for k in SPAN_FIELDS}})
        groups = evaluation["by_" + axis]
        # Retain all strata in CSV; keep a readable plot of at most 25 by support.
        keys = sorted(groups, key=lambda k: groups[k]["positive_spans"], reverse=True)[:25]
        plots.horizontal(name + "_by_" + axis, name.replace("_", " ") + " | end-to-end metrics by " + axis,
                         [f"{k} (positive spans={groups[k]['positive_spans']})" for k in keys],
                         {s.title(): [groups[k][s] if groups[k]["positive_spans"] else None for k in keys] for s in SCORES},
                         bounded=True, caption="Fixed validation threshold. Annotation-conditional precision; unknown alerts are separate. Up to 25 groups; full table in CSV.")
    csv_table(out / "tables" / f"{name}_span_strata.csv", rows)
    cat = evaluation["overall"]["by_category"]
    csv_table(out / "tables" / f"{name}_categories.csv", [{"category": k, **v} for k, v in cat.items()])
    keys = sorted(cat, key=lambda k: cat[k]["positive_spans"], reverse=True)[:25]
    plots.horizontal(name + "_categories", name.replace("_", " ") + " | detection by credential category",
                     [f"{k} (n={cat[k]['positive_spans']})" for k in keys],
                     {"Detected recall": [cat[k]["recall"] for k in keys],
                      "Candidate ceiling": [cat[k]["candidate_recall_ceiling"] for k in keys]}, bounded=True,
                     caption="One-to-one span matching. All categories, including benign categories, are in the CSV.")
    overall = evaluation["overall"]
    missed = overall["missed_spans"]
    csv_table(out / "tables" / f"{name}_missed_spans.csv", missed,
              fields=["document_id", "span_id", "category", "candidate_available"])
    proposal_misses = sum(not m["candidate_available"] for m in missed)
    plots.grouped(name + "_misses", name.replace("_", " ") + " | positive-span outcomes",
                  ["Detected", "No candidate", "Candidate below threshold / unmatched"],
                  {"Spans": [overall["true_positive"], proposal_misses, len(missed) - proposal_misses]}, ylabel="Positive spans",
                  caption="Candidate-available misses can include competition in one-to-one matching; they are not all pure classifier errors.")
    plots.grouped(name + "_alerts", name.replace("_", " ") + " | alert accounting",
                  ["Matched positives", "Explicit negatives", "Duplicate positives", "Unknown regions"],
                  {"Alerts": [overall["true_positive"], overall["explicit_negative_alerts"], overall["duplicate_positive_alerts"], overall["unknown_alerts"]]},
                  ylabel="Alerts", caption="Unknown-region alerts require review and are not silently counted as false positives.")


def make_html(markdown_text, out):
    try:
        import markdown
    except ImportError as exc:
        raise RuntimeError("Install reporting dependencies: .venv-train/bin/python -m pip install -r reporting_bundle/requirements.txt") from exc
    body = markdown.markdown(markdown_text, extensions=["tables", "fenced_code", "toc"])
    def embed(match):
        filename = match.group(1)
        path = (out / filename).resolve()
        if not path.is_relative_to(out.resolve()) or not path.is_file():
            return match.group(0)
        return 'src="data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode("ascii") + '"'
    body = re.sub(r'src="(plots/[^"<>]+\.png)"', embed, body)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DS404 — model evaluation report</title><style>
:root{color-scheme:light}body{font:16px/1.6 system-ui,sans-serif;color:#1c2c40;background:#f3f5f8;margin:0}main{max-width:1120px;margin:auto;background:white;padding:48px 55px}h1{font-size:2.4rem;line-height:1.15;color:#12395c}h2{margin-top:2.5em;border-top:2px solid #e4eaf0;padding-top:.7em;color:#12395c}h3{color:#235674}a{color:#12618c}table{display:block;overflow-x:auto;border-collapse:collapse;font-size:.88rem;margin:20px 0;max-width:100%}th,td{border:1px solid #dbe3eb;padding:8px 11px;vertical-align:top}th{background:#eaf0f6;text-align:left}tr:nth-child(even){background:#f8fafc}img{max-width:100%;height:auto;display:block;margin:18px auto}pre{background:#122536;color:#eaf2f8;overflow:auto;padding:18px;border-radius:7px}code{font-size:.88em}blockquote{border-left:5px solid #ba761d;background:#fff6e6;padding:12px 20px;margin:20px 0}hr{border:0;border-top:1px solid #dbe3eb}.toc{background:#f1f6fa;padding:15px 25px}@media(max-width:700px){main{padding:22px 18px}}@media print{body{background:white}main{padding:0;max-width:none}h2,h3{break-after:avoid}img,tr{break-inside:avoid}a{color:inherit}.toc{display:none}table{display:table;font-size:8pt}pre{white-space:pre-wrap}}
</style></head><body><main>""" + body + "</main></body></html>"


def historical_markdown(historical):
    """Render the saved evidence directly; background Markdown is not an input."""
    sections = [cell(historical.get("description")), cell(historical.get("metric_convention")),
                "The full historical record, including source hashes and nested counts, is in "
                "[data/historical_results.json](data/historical_results.json). Historical source paths "
                "identify the recorded snapshot and may not exist in the current checkout. "
                "These protocols must be interpreted separately from the current span evaluation."]
    metrics = [("name", "Tool / model"), ("operating_point", "Operating point"),
               ("threshold", "Threshold"), ("precision", "Precision"), ("recall", "Recall"),
               ("f1", "F1"), ("precision_strict", "Strict precision"),
               ("f1_strict", "Strict F1"), ("seconds", "Seconds")]
    for experiment in historical.get("experiments", []):
        sections += ["### " + cell(experiment["title"]),
                     table([{"field": k, "value": experiment.get(k)} for k in
                            ("dataset", "split", "scoring", "provenance")], [("field", "Field"), ("value", "Recorded value")]),
                     table(experiment["models"], metrics)]
        sections += ["- " + cell(note) for note in experiment.get("notes", [])]
        if experiment.get("count_semantics"):
            sections += [cell(experiment["count_semantics"])]
        for key in ("settings", "provenance_details", "development_auc", "development_candidate_recall", "model_summary"):
            if key in experiment:
                value = experiment[key]
                sections += ["**" + cell(key.replace("_", " ")) + ":** " +
                             cell(json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)]
        for model in experiment["models"]:
            details = []
            if model.get("raw_metrics"):
                details += ["Saved hit and label-row counts:",
                            table([{"field": k, "value": v} for k, v in model["raw_metrics"].items()],
                                  [("field", "Field"), ("value", "Recorded value")])]
            for category, rows in model.get("categories", {}).items():
                details += [cell(category.replace("_", " ")) + ":",
                            table([{"category": r[0], "support": r[1], "rate": r[2]} for r in rows],
                                  [("category", "Category"), ("support", "Label support"), ("rate", "Rate")])]
            if model.get("top_false_positive_sources"):
                details += ["Saved false-positive sources:",
                            table([{"source": r[0], "hits": r[1]} for r in model["top_false_positive_sources"]],
                                  [("source", "Source"), ("hits", "Hits")])]
            if details:
                sections += ["#### " + cell(model["name"]), *details]
        for name, rows in experiment.get("threshold_sweeps", {}).items():
            sections += ["#### Recorded threshold sweep: " + cell(name),
                         "Descriptive historical sweep; do not use it to retune a final-test threshold.",
                         table(rows, [(key, key) for key in dict.fromkeys(k for row in rows for k in row)])]
    return "\n\n".join(sections)


def project_overview(readme):
    """Reuse the root guide's workflow and interpretation without copying guides."""
    text = Path(readme).read_text(encoding="utf-8")
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    selected = [part for part in sections if part.startswith(("## Workflow and", "## Methodology and"))]
    if not selected:
        selected = [sections[0]]
    overview = "\n\n".join(selected)
    # The report includes the explanatory content; source paths remain labels.
    overview = re.sub(r"\[([^\]]+)\]\((?!https?://)[^)]+\)", r"\1", overview)
    return re.sub(r"^(#{1,5}) ", r"#\1 ", overview, flags=re.MULTILINE)


def build(args):
    sys.path.insert(0, str(BUNDLE))
    from report_stats import candidate_statistics, summarize_groups
    from plotting import PlotWriter
    root = Path(args.project_root).resolve()
    print("Reading frozen experiment and dataset artifacts...", flush=True)
    inputs = Inputs(root, args.allow_partial)
    suite = inputs.path(args.experiments)
    dataset = inputs.path(args.dataset)
    selected = inputs.json(suite / "selected.json", required=True)
    leaderboard_obj = inputs.json(suite / "leaderboard.json", required=True)
    suite_manifest = inputs.json(suite / "suite_manifest.json", required=True)
    # Validate dependencies and required inputs before creating any report output.
    import markdown  # noqa: F401
    if not (root / "secrets_pipeline" / "evaluate.py").is_file():
        raise ValueError("--project-root must be the extracted ISP directory containing secrets_pipeline/evaluate.py")
    sys.path.insert(0, str(root))
    from secrets_pipeline.evaluate import evaluate_documents
    documents_path, candidates_path = dataset / "documents.jsonl", dataset / "candidates.jsonl"
    documents = list(inputs.rows(documents_path, required=True))
    candidates = {}
    for row in inputs.rows(candidates_path, required=True):
        cid = row["id"]
        if cid in candidates:
            raise ValueError("Duplicate frozen candidate identifier")
        candidates[cid] = {k: row[k] for k in SAFE_PREDICTION if k in row}
    if selected:
        for p, key in ((documents_path, "documents_sha256"), (candidates_path, "input_sha256")):
            if p.is_file() and digest(p) != selected.get(key):
                raise ValueError(f"Frozen input hash does not match selected.json: {p}")
        if selected.get("validation_only") is not True or selected.get("test_used_for_selection") is not False:
            raise ValueError("selected.json does not affirm validation-only model selection")
        if suite_manifest and (suite_manifest.get("fingerprint") != selected.get("suite_fingerprint") or
                               suite_manifest.get("corpus_sha256") != selected.get("corpus_sha256")):
            raise ValueError("Suite manifest and selected bundle do not agree")
    validation_predictions = load_selected_predictions(inputs, selected, suite) if selected else None
    val_docs = [d for d in documents if d["split"] == "val" and d["source"] in REAL]
    sections, assessments = [], {}
    for split, directory in (("test", args.test), ("test_ood", args.test_ood)):
        print(f"Checking saved {split} assessment...", flush=True)
        result = inputs.json(inputs.path(directory) / "metrics.json", required=True)
        pred_path = inputs.available(inputs.path(directory) / "predictions.jsonl", required=True)
        rows = [{k: r[k] for k in SAFE_PREDICTION if k in r} for r in inputs.rows(pred_path)] if pred_path else None
        if result is None:
            continue
        if result.get("split") != split or result.get("thresholds_tuned") is not False:
            raise ValueError(f"Unexpected assessment split/tuning flag in {directory}")
        if selected:
            if result.get("corpus_sha256") != selected.get("corpus_sha256") or result.get("external_corpus"):
                raise ValueError(f"Assessment corpus differs from frozen experiment in {directory}")
            for op in ("warn", "block"):
                if result.get("thresholds", {}).get(op) != selected.get("thresholds", {}).get(op):
                    raise ValueError(f"Assessment {op} threshold differs from selected model in {directory}")
        ds = [d for d in documents if d["split"] == split]
        if documents and rows is not None:
            check_predictions(rows, ds, candidates, split)
            for op in ("warn", "block"):
                threshold = result["thresholds"].get(op)
                if threshold is None:
                    if result.get(op) is not None:
                        raise ValueError(f"Disabled {op} unexpectedly has saved metrics")
                    continue
                actual = evaluate_documents(ds, rows, threshold)
                verify_metrics(result[op], actual, split + "/" + op)
                # Use fresh computed strata and missed-spans, after proving stored summaries match.
                result[op] = actual
        assessments[split] = {"result": result, "predictions": rows, "documents": ds}
    if selected and documents and validation_predictions is not None:
        check_predictions(validation_predictions, val_docs, candidates, "real validation")
        verified_val = evaluate_documents(val_docs, validation_predictions, selected["thresholds"]["warn"])
        verify_metrics(selected["validation_metrics"], verified_val, "selected validation")
        val_ap = candidate_statistics(validation_predictions, selected["thresholds"]["warn"])["metrics"]["average_precision"]
        if selected.get("candidate_average_precision") is not None and not math.isclose(val_ap, selected["candidate_average_precision"], rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("Selected validation candidate AP differs from saved scores")
    else:
        verified_val = selected.get("validation_metrics") if selected else None

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = inputs.path(args.out) if args.out else BUNDLE / "output" / ("report-" + stamp)
    # An output must not replace/contain any input or the source bundle.
    for source in [BUNDLE / "build_report.py", *[inputs.path(p) for p in inputs.inventory]]:
        if source.is_relative_to(out.resolve()):
            raise ValueError("Output directory would contain existing input/source artifacts")
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output is nonempty: {out}. Choose a new directory to avoid stale plots.")
    out.mkdir(parents=True, exist_ok=True)
    print("Generating tables and figures from saved scores (no inference)...", flush=True)
    plots = PlotWriter(out / "plots")
    report = {"schema_version": 1, "created_utc": stamp, "mode": "saved_artifacts_only",
              "inference_performed": False, "thresholds_retuned": False, "selected_name": selected.get("name") if selected else None,
              "limitations": ["The selected model won on real validation; no test-driven model reselection is performed.",
                 "Span precision is annotation-conditional. Unknown-region alerts are review burden, not certified false positives.",
                 "Candidate accuracy/AP/ROC/calibration exclude unknown labels and missed proposals; they are not end-to-end scanner scores.",
                 "Existing assess artifacts do not record a model/bundle hash. Corpus, thresholds, candidate identities and recomputed metrics are checked, but model identity cannot be cryptographically proven.",
                 "No inference latency, throughput, peak GPU memory or train-set accuracy is invented when it was not recorded."],
              "evaluations": {}, "candidate_diagnostics": {}, "training_runs": [], "baseline_results": []}
    intro = ["# DS404 — selected model and scanner report", "[TOC]",
             f"Generated in UTC: `{stamp}`. Precision, recall, F1, AP and AUC are fractions in [0, 1]; MCC is in [−1, 1], and log loss is unbounded. N/A means undefined or unavailable.",
             "This report reads the completed experiment and assessment artifacts. It does not train, download models, run GPU inference, or change the selected model or its thresholds."]
    if selected:
        report["selection"] = {k: selected.get(k) for k in ("name", "models", "selection_rule", "thresholds", "suite_fingerprint", "input_sha256", "documents_sha256", "corpus_sha256")}
        sections += ["## Selected model", f"**Validation winner:** {cell(selected['name'])}",
                     f"Selection rule: `{cell(selected.get('selection_rule'))}`.",
                     table(selected["models"], [("run", "Training run"), ("name", "Encoder family"), ("weight", "Ensemble weight"), ("path", "Model path")]),
                     table([{"name": k, "value": v} for k, v in selected["thresholds"].items() if not isinstance(v, (dict, list))], [("name", "Calibration field"), ("value", "Value")]),
                     "A null BLOCK threshold means automatic blocking is disabled because validation did not meet the precision/support requirement. It is not replaced by an arbitrary threshold."]
    span_summary, candidates_summary = [], []
    if verified_val:
        report["evaluations"]["validation_real"] = {"warn": verified_val}
        span_summary.append({"scope": "Real validation (selection)", "operating_point": "warn", **{k: verified_val["overall"].get(k) for k in SPAN_FIELDS}})
    for split, entry in assessments.items():
        result, predictions, ds = entry["result"], entry["predictions"], entry["documents"]
        report["evaluations"][split] = result
        for op in ("warn", "block"):
            if result.get(op) is None:
                continue
            ev = result[op]
            if split == "test":
                span_summary.append({"scope": "Real test (primary)", "operating_point": op, **{k: ev["headline_real_test"].get(k) for k in SPAN_FIELDS}})
                span_summary.append({"scope": "All test sources (context only)", "operating_point": op, **{k: ev["overall"].get(k) for k in SPAN_FIELDS}})
            else:
                span_summary.append({"scope": "OOD test (separate)", "operating_point": op, **{k: ev["overall"].get(k) for k in SPAN_FIELDS}})
        span_tables(split, result["warn"], plots, out)
        if split == "test" and predictions is not None:
            real_pred = [p for p in predictions if p.get("source") in REAL]
            if ds:
                real_docs = [d for d in ds if d["source"] in REAL]
                real_eval = evaluate_documents(real_docs, real_pred, result["thresholds"]["warn"])
                span_tables("test_real", real_eval, plots, out)
        else:
            real_pred = predictions
        if real_pred is not None:
            name = "test_real" if split == "test" else "test_ood"
            stats = diagnostics(name, real_pred, result["thresholds"]["warn"], plots, out, candidate_statistics)
            report["candidate_diagnostics"][name] = stats
            if result["thresholds"].get("block") is not None:
                block_stats = candidate_statistics(real_pred, result["thresholds"]["block"])
                report["candidate_diagnostics"][name + "_block"] = block_stats
                m = block_stats["metrics"]
                plots.confusion(name + "_block_confusion", name.replace("_", " ") + " | BLOCK candidate confusion matrix",
                                m["tn"], m["fp"], m["fn"], m["tp"], caption="Explicitly labeled proposed candidates at the frozen BLOCK threshold.")
        # Separate source diagnostics avoid hiding synthetic/real distribution shifts.
        if predictions is not None:
            per_source = []
            for source in sorted({p["source"] for p in predictions}):
                s = candidate_statistics([p for p in predictions if p["source"] == source], result["thresholds"]["warn"])
                per_source.append({"source": source, "labeled_count": s["labeled_count"], "unknown_count": s["unknown_count"], **s["metrics"]})
            csv_table(out / "tables" / f"{split}_candidate_by_source.csv", per_source)
            report["evaluations"][split]["candidate_by_source"] = per_source
    if selected and validation_predictions is not None:
        report["candidate_diagnostics"]["validation_real"] = diagnostics("validation_real", validation_predictions,
               selected["thresholds"]["warn"], plots, out, candidate_statistics)
    for name, stats in report["candidate_diagnostics"].items():
        candidates_summary.append({"scope": name, "labeled_count": stats["labeled_count"], "unknown_count": stats["unknown_count"], **stats["metrics"]})
    report["span_summary"] = span_summary
    csv_table(out / "tables" / "end_to_end_metrics.csv", span_summary)
    csv_table(out / "tables" / "candidate_metrics.csv", candidates_summary)
    sections += ["## End-to-end scanner performance", "**The primary result is real test at WARN.** Validation was used for selection. OOD is a separate transfer test; the pooled test row is context only.",
                 table(span_summary, [("scope", "Scope"), ("operating_point", "Gate"), ("documents", "Docs"), ("positive_spans", "Positive spans"), ("true_positive", "TP"), ("false_positive", "FP"), ("false_negative", "FN"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("candidate_recall_ceiling", "Candidate ceiling"), ("unknown_alerts_per_kloc", "Unknown/KLOC")]),
                 "FP comprises alerts over explicit negative annotations plus duplicate alerts over already matched positives. There is no meaningful total count of true-negative spans in source text, so end-to-end accuracy and a conventional span confusion matrix are not reported. Empty strata do not provide performance evidence, even when the upstream evaluator stores zeros."]
    main_rows = [r for r in span_summary if r["operating_point"] == "warn" and r["scope"] != "All test sources (context only)"]
    plots.grouped("span_overview", "End-to-end span performance | distinct evaluation populations", [r["scope"] for r in main_rows],
                  {m.title(): [r[m] if r["positive_spans"] else None for r in main_rows] for m in SCORES}, bounded=True,
                  caption="Annotation-conditional precision. These populations remain distinct; validation is selection data.")
    ci_rows = []
    for split, entry in assessments.items():
        boot = entry["result"].get("real_repository_bootstrap")
        if boot:
            for metric, interval in boot["intervals"].items():
                ci_rows.append({"split": split, "population": "Real sources only / WARN", "metric": metric, **interval,
                                "repeats": boot["repeats"], "repositories": boot["repositories"], "confidence": boot["confidence"]})
    csv_table(out / "tables" / "bootstrap_intervals.csv", ci_rows)
    sections += ["## Saved confidence intervals", table(ci_rows, [("split", "Split"), ("population", "Population"), ("metric", "Metric"), ("lower", "Lower"), ("upper", "Upper"), ("repeats", "Resamples"), ("repositories", "Repositories")]),
                 "These are the saved 95% repository-cluster percentile intervals at WARN. They omit training-seed, selection, threshold-search and annotation uncertainty. A Tier B-only OOD split has no real-source bootstrap interval, even when assess was called with 500 samples. No unavailable interval is fabricated."]
    if ci_rows:
        plots.grouped("confidence_intervals", "Saved 95% bootstrap interval bounds | real sources, WARN",
                      [r["split"] + " / " + r["metric"] for r in ci_rows if r["metric"] in SCORES],
                      {"Lower": [r["lower"] for r in ci_rows if r["metric"] in SCORES], "Upper": [r["upper"] for r in ci_rows if r["metric"] in SCORES]}, bounded=True,
                      caption="Bounds are shown as paired bars; see CSV for repository support and number of resamples.")
    sections += ["## Candidate classifier diagnostics", "These metrics use only explicitly labeled extracted candidates. Unknown candidates and secrets with no proposal are excluded. Consequently, candidate F1/accuracy/ROC-AUC must not be described as whole-scanner performance.",
                 table(candidates_summary, [("scope", "Scope"), ("labeled_count", "Labeled"), ("unknown_count", "Unknown"), ("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("mcc", "MCC"), ("average_precision", "AP"), ("roc_auc", "ROC AUC"), ("brier_score", "Brier"), ("log_loss", "Log loss"), ("ece", "ECE")]),
                 "AP groups tied scores and uses the stepwise precision-recall integral. ROC AUC uses trapezoidal integration; it is undefined if either class is absent. ECE uses ten equal-width probability bins. Full TN/FP/FN/TP, specificity, NPV, FPR and FNR are in CSV/JSON. Threshold curves are descriptive and never choose a new test threshold."]

    leaderboard = (leaderboard_obj or {}).get("results", [])
    if leaderboard_obj and leaderboard_obj.get("test_used_for_selection") is not False:
        raise ValueError("Leaderboard does not affirm validation-only comparison")
    flat_leaderboard = [{k: v for k, v in r.items() if k != "models"} for r in leaderboard]
    csv_table(out / "tables" / "validation_leaderboard.csv", flat_leaderboard)
    report["leaderboard"] = leaderboard
    sections += ["## Validation experiments and training", table(flat_leaderboard, [("name", "Experiment"), ("span_precision", "Span precision"), ("span_recall", "Span recall"), ("span_f1", "Span F1"), ("candidate_average_precision", "Candidate AP"), ("unknown_alerts_per_kloc", "Unknown/KLOC")])]
    if leaderboard:
        plots.horizontal("validation_leaderboard", "Experiment selection | real-validation span metrics", [r["name"] for r in leaderboard],
                         {s.title(): [r["span_" + s] for r in leaderboard] for s in SCORES}, bounded=True,
                         caption="All rows use the same real validation documents. Test scores are not used to rank runs.")
    runs, seed_rows, history_rows = [], [], []
    # A winning ensemble can appear before its members in the sorted leaderboard.
    # Seed summaries must use single-run rows independently of run-file deduplication.
    for result in leaderboard:
        members = result.get("models", [])
        if len(members) == 1:
            match = re.match(r"(.+)-(real_only|augmented)-(plain|masked)-seed(\d+)$", members[0]["run"])
            if match:
                seed_rows.append({"model": match[1], "dataset": match[2], "mask": match[3], "seed": int(match[4]), "span_f1": result["span_f1"]})
    seen = set()
    for result in leaderboard:
        for member in result.get("models", []):
            if member["run"] in seen:
                continue
            seen.add(member["run"])
            run_dir = suite / "runs" / member["run"]
            run = inputs.json(run_dir / "run.json", required=True)
            if run is None:
                continue
            history = inputs.json(run_dir / "history.json") or run.get("history", [])
            cfg = run.get("config", {})
            best_cfg = inputs.json(run_dir / "best" / "config.json") or {}
            config_keys = ("model_type", "architectures", "hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "vocab_size", "max_position_embeddings", "hidden_dropout_prob", "attention_probs_dropout_prob", "classifier_dropout", "num_labels", "id2label")
            safe_run = {"id": member["run"], "config": cfg, "history": history, "best_checkpoint": run.get("best_checkpoint"),
                        "versions": run.get("versions"), "model_revision": run.get("model_revision"), "world_size": run.get("world_size"),
                        "effective_batch_size": run.get("effective_batch_size"), "architecture": {k: best_cfg[k] for k in config_keys if k in best_cfg},
                        "accounted_epoch_seconds": sum(h.get("seconds", 0) for h in history)}
            audit_keys = ("training_examples", "validation_examples", "train_positive_count", "validation_positive_count",
                          "ignored_unknown_counts", "all_split_candidate_counts", "supervised_source_counts",
                          "validation_sources", "synthetic_validation_fallback", "data_sha256", "test_used_for_selection")
            safe_run["data_audit"] = {k: run["data_audit"][k] for k in audit_keys if k in run.get("data_audit", {})}
            runs.append(safe_run)
            for h in history:
                history_rows.append({"run": member["run"], **h})
            if history:
                epoch = [h["epoch"] for h in history]
                plots.lines("training_" + slug(member["run"]), member["run"] + " | training loss", epoch,
                            {"Rank-0 weighted loss": [h["training_loss_rank0"] for h in history]}, xlabel="Epoch", ylabel="Weighted cross entropy",
                            caption="Rank-0 training loss. In distributed runs this is not an all-worker average.")
                plots.lines("validation_" + slug(member["run"]), member["run"] + " | candidate validation", epoch,
                            {"Average precision": [h["validation_average_precision"] for h in history], "WARN F1": [h["validation_warn_f1"] for h in history]},
                            xlabel="Epoch", ylabel="Candidate metric", bounded=True,
                            caption="Epoch diagnostics use explicit real-validation candidates; these are not full-document span metrics.")
    report["training_runs"] = runs
    csv_table(out / "tables" / "epoch_history.csv", history_rows)
    group_stats = summarize_groups(seed_rows, ["model", "dataset", "mask"], "span_f1")
    report["validation_seed_summary"] = group_stats
    csv_table(out / "tables" / "validation_seed_summary.csv", group_stats)
    sections += ["### Variation across training seeds", table(group_stats, [("model", "Encoder"), ("dataset", "Training data"), ("mask", "Key masking"), ("n", "Seeds"), ("mean", "Mean span F1"), ("std", "Sample SD"), ("min", "Min"), ("max", "Max")]),
                 "These are descriptive variation across validation runs, not test confidence intervals. Sample SD uses n−1; it is unavailable for a single seed."]
    paired = []
    pairs = defaultdict(dict)
    for row in seed_rows:
        pairs[(row["model"], row["mask"], row["seed"])][row["dataset"]] = row["span_f1"]
    for (model, mask, seed), pair in sorted(pairs.items()):
        if {"real_only", "augmented"} <= set(pair):
            paired.append({"model": model, "mask": mask, "seed": seed, "real_only_f1": pair["real_only"], "augmented_f1": pair["augmented"], "delta_f1": pair["augmented"] - pair["real_only"]})
    report["paired_validation_augmentation"] = paired
    csv_table(out / "tables" / "paired_augmentation.csv", paired)
    if paired:
        plots.grouped("augmentation_delta", "Augmentation effect | paired real-validation runs", [f"{r['model']} / {r['mask']} / {r['seed']}" for r in paired],
                      {"Augmented − real-only F1": [r["delta_f1"] for r in paired]}, ylabel="Change in span F1",
                      caption="Matched encoder, seed and masking. These are validation effects and do not prove a test gain.")
    for run in runs:
        if selected and run["id"] in {m["run"] for m in selected["models"]}:
            sections += ["### Selected member: " + cell(run["id"]),
                         table([{"field": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v} for k, v in run["architecture"].items()], [("field", "Saved architecture field"), ("value", "Value")]),
                         table([{"field": k, "value": v} for k, v in run["config"].items() if not isinstance(v, (list, dict))], [("field", "Training setting"), ("value", "Value")]),
                         table([{"field": k, "value": json.dumps(v, sort_keys=True)} for k, v in run["config"].items() if isinstance(v, (list, dict))], [("field", "Structured training setting"), ("value", "Value")]),
                         table([{"field": k, "value": json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v} for k, v in run["data_audit"].items()], [("field", "Recorded training population"), ("value", "Value")]),
                         f"Best checkpoint: `{cell(run['best_checkpoint'])}`. Effective batch size: {cell(run['effective_batch_size'])}. Sum of saved epoch timers: {run['accounted_epoch_seconds']:.1f} seconds; excludes checkpoint saving/final export and is not full job wall time."]

    corpus_rows = describe_corpus(documents, candidates)
    report["dataset_composition"] = corpus_rows
    csv_table(out / "tables" / "dataset_composition.csv", corpus_rows)
    sections += ["## Dataset composition and generation", table(corpus_rows, [("split", "Split"), ("source", "Source"), ("language", "Language"), ("documents", "Docs"), ("repositories", "Repos"), ("positive_spans", "Positive spans"), ("negative_spans", "Negative spans"), ("candidates_positive", "Positive candidates"), ("candidates_negative", "Negative candidates"), ("candidates_unknown", "Unknown candidates")]),
                 "Repository/owner counts in this table are per stratum and must not be summed as unique corpus totals."]
    if corpus_rows:
        splits = [s for s in ("train", "val", "test", "test_ood") if any(r["split"] == s for r in corpus_rows)]
        sources = sorted({r["source"] for r in corpus_rows})
        plots.grouped("dataset_sources", "Dataset composition | documents per source and split", splits,
                      {source: [sum(r["documents"] for r in corpus_rows if r["split"] == split and r["source"] == source) for split in splits] for source in sources}, ylabel="Documents",
                      caption="Original real data, modified Tier B and generated data remain separately labeled.")
    generation = inputs.json(dataset / "generation" / "report.json")
    if generation:
        safe_keys = ("status", "backend", "mock_data", "planned_groups", "accepted_groups", "rejected_groups", "documents", "positive_spans", "negative_spans", "unvalidated_documents", "accepted_by_family", "rejected_by_family", "rejection_reasons", "reject_rate", "documents_by_split", "coverage_cells")
        report["generation"] = {k: generation[k] for k in safe_keys if k in generation}
        sections += ["### Generation acceptance", table([{"field": k, "value": generation.get(k)} for k in safe_keys if not isinstance(generation.get(k), (list, dict))], [("field", "Field"), ("value", "Value")])]
        families = sorted(set(generation.get("accepted_by_family", {})) | set(generation.get("rejected_by_family", {})))
        plots.horizontal("generation_acceptance", "Generation quality gates | accepted and rejected groups", families,
                         {"Accepted": [generation.get("accepted_by_family", {}).get(k, 0) for k in families], "Rejected": [generation.get("rejected_by_family", {}).get(k, 0) for k in families]}, xlabel="Groups")
        csv_table(out / "tables" / "generation_rejections.csv", [{"reason": k, "reason_events": v} for k, v in generation.get("rejection_reasons", {}).items()])
        sections += ["Rejection-reason counts are retry/admission events, including retries of groups that later succeeded; they need not sum to rejected groups."]
    before, after = inputs.json(dataset / "coverage.before.json"), inputs.json(dataset / "coverage.after.json")
    if before and after:
        report["coverage_before_after"] = {"before": {k: before.get(k) for k in ("sources", "families", "span_labels", "annotation_precision", "train_cell_coverage")},
                                           "after": {k: after.get(k) for k in ("sources", "families", "span_labels", "annotation_precision", "train_cell_coverage")}}
        counts = []
        for phase, audit in (("before", before), ("after", after)):
            counts.extend({"phase": phase, **r} for r in audit.get("train_cell_coverage", []))
        csv_table(out / "tables" / "training_coverage_before_after.csv", counts)
        families = sorted(set(before.get("families", {})) | set(after.get("families", {})))
        plots.horizontal("coverage_before_after", "Corpus format coverage | before and after generation", families,
                         {"Before": [before.get("families", {}).get(k, 0) for k in families], "After": [after.get("families", {}).get(k, 0) for k in families]}, xlabel="Documents",
                         caption="Descriptive corpus counts, not evidence of improved generalization. Generation quotas use training data only.")

    sections += ["## Baselines under the current span protocol"]
    baseline_rows = []
    for folder in args.baselines:
        directory = inputs.path(folder)
        if not directory.is_dir():
            inputs.available(directory / "summary.json")
            continue
        for path in sorted(directory.glob("*.evaluation.json")):
            base = inputs.json(path)
            if selected and base.get("corpus_sha256") != selected.get("corpus_sha256"):
                raise ValueError(f"Baseline corpus differs: {path}")
            split_names = list(base.get("by_split", {}))
            if len(split_names) != 1:
                raise ValueError(f"Baseline evaluation must describe one split: {path}")
            split = split_names[0]
            metric = base["headline_real_test"] if split == "test" else base["overall"]
            row = {"tool": base.get("tool"), "split": split, "operating_point": base.get("operating_point"),
                   "scope": {"test": "Real test", "test_ood": "All selected OOD sources", "val": "All selected validation sources"}.get(split, split), **{k: metric.get(k) for k in SPAN_FIELDS}}
            baseline_rows.append(row)
            report["baseline_results"].append({"file": inputs.label(path), "summary": row, "legacy_training_overlap_caveat": base.get("legacy_training_overlap_caveat"), "localization_note": base.get("localization_note")})
    csv_table(out / "tables" / "current_baselines.csv", baseline_rows)
    sections += [table(baseline_rows, [("tool", "Tool"), ("split", "Split"), ("operating_point", "Gate"), ("scope", "Scope"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("unknown_alerts_per_kloc", "Unknown/KLOC")]),
                 "If no current baseline files exist, the assessment commands did not create them. See the root README for optional baseline commands. The shipped legacy model may overlap its original training repositories even under the new scoring protocol; detector localization also differs."]
    if baseline_rows:
        for split in sorted({r["split"] for r in baseline_rows}):
            rows = [r for r in baseline_rows if r["split"] == split]
            plots.grouped("baselines_" + split, "Existing scanner baselines | " + split, [str(r["tool"]) + " / " + str(r["operating_point"]) for r in rows],
                          {s.title(): [r[s] for r in rows] for s in SCORES}, bounded=True,
                          caption="Legacy training overlap and detector localization prevent an unqualified fair-ranking claim.")

    historical = inputs.json(BUNDLE / "data" / "historical_results.json", required=True)
    report["historical"] = historical
    historical_rows = []
    if historical:
        write_json(out / "data" / "historical_results.json", historical)
        for experiment in historical["experiments"]:
            models = experiment["models"]
            historical_rows.extend({"experiment": experiment["id"], **{k: v for k, v in r.items() if not isinstance(v, (dict, list))}} for r in models)
            plots.horizontal("historical_" + slug(experiment["id"]), "Historical | " + experiment["title"],
                             [str(r["name"]) + " / " + str(r.get("operating_point", "")) for r in models],
                             {s.title(): [r.get(s) for r in models] for s in SCORES}, bounded=True,
                             caption="Historical protocol only. " + " ".join(experiment.get("notes", [])[:2]))
    csv_table(out / "tables" / "historical_results.csv", historical_rows)
    sections += ["## Previous scanner results and benchmark history",
                 historical_markdown(historical) if historical else
                 "Historical comparison data is unavailable for this incomplete report."]
    readme = root / "README.md"
    if readme.is_file():
        inputs.track(readme)
        sections += ["## Model and scanner design", project_overview(readme)]
    sections += ["## Plot gallery", "Every available plot is saved as PNG and SVG. CSV files retain full tables even where charts show the 25 largest strata."]
    for item in plots.items:
        sections += ["### " + cell(item["title"]), f"![{cell(item['title'])}](plots/{item['filename']})", cell(item.get("caption", ""))]
    report["missing_artifacts"] = inputs.missing
    report["complete_required_inputs"] = not any(m["required"] for m in inputs.missing)
    report["plots"] = plots.items
    if not report["complete_required_inputs"]:
        intro.insert(2, "> **INCOMPLETE REPORT:** required artifacts were missing and --allow-partial was explicitly used. Missing results are not zeros and cannot support a model-performance claim.")
    sections += ["## Evidence, limitations and unavailable statistics", "\n".join("- " + x for x in report["limitations"]),
                 table(inputs.missing, [("path", "Unavailable artifact"), ("required", "Required for complete report")]),
                 "Full numeric outputs are in `report.json` and `tables/`. The input inventory records SHA-256 hashes of files consumed. The HTML embeds plot PNGs and opens offline; linked Markdown/CSV files accompany the output folder."]
    content = "\n\n".join(intro + sections) + "\n"
    (out / "README.md").write_text(content, encoding="utf-8")
    (out / "report.html").write_text(make_html(content, out), encoding="utf-8")
    write_json(out / "report.json", report)
    for p in (BUNDLE / "build_report.py", BUNDLE / "plotting.py", BUNDLE / "report_stats.py", root / "secrets_pipeline" / "evaluate.py"):
        inputs.track(p)
    versions = {"python": sys.version, "matplotlib": importlib.metadata.version("matplotlib"), "Markdown": importlib.metadata.version("Markdown")}
    write_json(out / "input_manifest.json", {"created_utc": stamp, "project_root": str(root), "files": list(inputs.inventory.values()), "versions": versions,
               "validation": "frozen file hashes + thresholds + candidate identity/labels + recomputed span metrics where available",
               "assessment_model_hash_available": False, "missing": inputs.missing})
    print(f"Report: {out / 'report.html'}")
    print(f"Markdown: {out / 'README.md'}")
    print(f"Plots: {len(plots.items)} (PNG + SVG). Required inputs complete: {report['complete_required_inputs']}")
    return out


def parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root", default=str(BUNDLE.parent))
    ap.add_argument("--experiments", default="artifacts/experiments")
    ap.add_argument("--dataset", default="artifacts/full")
    ap.add_argument("--test", default="artifacts/test")
    ap.add_argument("--test-ood", default="artifacts/test-ood")
    ap.add_argument("--baselines", nargs="*", default=["artifacts/baselines-test", "artifacts/baselines-test-ood"])
    ap.add_argument("--out", help="New/empty output directory; default: timestamped reporting_bundle/output/report-* (UTC)")
    ap.add_argument("--allow-partial", action="store_true", help="Explicitly permit an INCOMPLETE report with missing required artifacts")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        build(args)
    except (OSError, ValueError, KeyError, ImportError, RuntimeError) as exc:
        print(f"Report failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
