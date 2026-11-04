"""Offline reporting checks; all fixtures are fictional and require no models/GPU."""
from __future__ import annotations

import contextlib
import copy
import io
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BUNDLE_ROOT.parent
for location in (PROJECT_ROOT, BUNDLE_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from report_stats import candidate_statistics, summarize_groups


def scored(labels, scores):
    return [{"label": label, "score": score} for label, score in zip(labels, scores)]


class CandidateStatisticsTests(unittest.TestCase):
    def test_tied_ranking_scores_use_grouped_average_precision(self):
        rows = scored([1, 0, 1, 0], [.9, .8, .8, .1])
        original = copy.deepcopy(rows)
        report = candidate_statistics(rows, .8)
        metrics = report["metrics"]
        # AP = (1/2)*1 + (1/2)*(2/3); AUC = (2 + 1.5)/4.
        self.assertAlmostEqual(metrics["average_precision"], 5 / 6)
        self.assertAlmostEqual(metrics["roc_auc"], .875)
        self.assertEqual({key: metrics[key] for key in ("tp", "fp", "fn", "tn")},
                         {"tp": 2, "fp": 1, "fn": 0, "tn": 1})
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertEqual(metrics["recall"], 1)
        self.assertAlmostEqual(metrics["f1"], .8)
        self.assertAlmostEqual(metrics["mcc"], 1 / math.sqrt(3))
        self.assertEqual(report["curves"]["thresholds"]["x"], [.9, .8, .1])
        self.assertEqual(candidate_statistics(list(reversed(rows)), .8), report)
        self.assertEqual(rows, original)

    def test_fully_tied_classifier_is_not_artificially_perfect(self):
        report = candidate_statistics(scored([1, 0, 1, 0], [.5] * 4), .5)
        self.assertAlmostEqual(report["metrics"]["average_precision"], .5)
        self.assertAlmostEqual(report["metrics"]["roc_auc"], .5)
        self.assertEqual(report["curves"]["thresholds"]["x"], [.5])
        self.assertEqual(report["metrics"]["tp"], 2)
        self.assertEqual(report["metrics"]["fp"], 2)

    def test_unknown_labels_do_not_become_negatives(self):
        known = scored([1, 0], [.8, .2])
        base = candidate_statistics(known, .5)
        report = candidate_statistics(known + scored([None, None], [1, 0]), .5)
        self.assertEqual(report["count"], 4)
        self.assertEqual(report["labeled_count"], 2)
        self.assertEqual(report["unknown_count"], 2)
        self.assertEqual(report["metrics"], base["metrics"])
        self.assertEqual(report["curves"], base["curves"])
        self.assertEqual(report["reliability"], base["reliability"])

    def test_missing_positive_class_is_explicitly_undefined(self):
        report = candidate_statistics(scored([0, 0], [.9, .1]), .5)
        metrics = report["metrics"]
        for key in ("recall", "fnr", "balanced_accuracy", "mcc", "average_precision", "roc_auc"):
            self.assertIsNone(metrics[key], key)
        self.assertEqual(metrics["specificity"], .5)
        self.assertEqual(metrics["f1"], 0)
        self.assertEqual(report["curves"]["precision_recall"], {"x": [], "y": []})
        self.assertEqual(report["curves"]["roc"], {"x": [], "y": []})

    def test_missing_negative_class_does_not_get_roc_auc(self):
        report = candidate_statistics(scored([1, 1], [.9, .1]), .5)
        metrics = report["metrics"]
        for key in ("specificity", "fpr", "balanced_accuracy", "mcc", "roc_auc"):
            self.assertIsNone(metrics[key], key)
        self.assertEqual(metrics["recall"], .5)
        self.assertEqual(metrics["average_precision"], 1)
        self.assertEqual(report["curves"]["roc"], {"x": [], "y": []})

    def test_no_predicted_positives_uses_undefined_precision(self):
        metrics = candidate_statistics(scored([1, 0], [.8, .2]), 1)["metrics"]
        self.assertIsNone(metrics["precision"])
        self.assertIsNone(metrics["mcc"])
        self.assertEqual(metrics["recall"], 0)
        self.assertEqual(metrics["f1"], 0)
        self.assertEqual(metrics["accuracy"], .5)
        self.assertEqual(metrics["npv"], .5)

    def test_threshold_is_frozen_and_inclusive(self):
        report = candidate_statistics(scored([1, 0, 1, 0], [.5, .5, .499, .499]), .5)
        self.assertEqual(report["threshold"], .5)
        self.assertEqual({key: report["metrics"][key] for key in ("tp", "fp", "fn", "tn")},
                         {"tp": 1, "fp": 1, "fn": 1, "tn": 1})

    def test_empty_and_unknown_only_populations_have_no_quality_claim(self):
        for rows in ([], scored([None], [.9])):
            with self.subTest(rows=rows):
                report = candidate_statistics(rows, .5)
                self.assertEqual(report["labeled_count"], 0)
                for key, value in report["metrics"].items():
                    if key in ("tp", "fp", "fn", "tn"):
                        self.assertEqual(value, 0)
                    else:
                        self.assertIsNone(value, key)
                self.assertEqual(sum(bucket["count"] for bucket in report["reliability"]), 0)

    def test_endpoint_scores_have_finite_log_loss(self):
        perfect = candidate_statistics(scored([0, 1], [0, 1]), .5)["metrics"]
        worst = candidate_statistics(scored([0, 1], [1, 0]), .5)["metrics"]
        self.assertTrue(math.isfinite(perfect["log_loss"]))
        self.assertLess(perfect["log_loss"], 1e-12)
        self.assertEqual(perfect["brier_score"], 0)
        self.assertEqual(perfect["average_precision"], 1)
        self.assertEqual(perfect["roc_auc"], 1)
        self.assertTrue(math.isfinite(worst["log_loss"]))
        self.assertGreater(worst["log_loss"], 34)
        self.assertEqual(worst["brier_score"], 1)
        self.assertEqual(worst["roc_auc"], 0)
        self.assertEqual(worst["average_precision"], .5)

    def test_reliability_bins_use_left_closed_boundaries_and_include_one(self):
        report = candidate_statistics(scored([0, 1, 1], [0, .5, 1]), .5, bins=2)
        self.assertEqual([row["count"] for row in report["reliability"]], [1, 2])
        self.assertEqual(report["reliability"][1]["mean_score"], .75)
        self.assertEqual(report["reliability"][1]["positive_rate"], 1)
        self.assertAlmostEqual(report["metrics"]["ece"], 1 / 6)
        self.assertAlmostEqual(report["metrics"]["brier_score"], 1 / 12)

    def test_invalid_scores_and_thresholds_are_rejected_even_on_unknowns(self):
        invalid = [None, "0.5", True, False, float("nan"), float("inf"),
                   -float("inf"), -.01, 1.01, 10 ** 400]
        for value in invalid:
            with self.subTest(score=value):
                with self.assertRaises(ValueError):
                    candidate_statistics([{"label": None, "score": value}], .5)
            with self.subTest(threshold=value):
                with self.assertRaises(ValueError):
                    candidate_statistics([], value)

    def test_invalid_labels_rows_and_bin_counts_are_rejected(self):
        for label in (True, False, -1, 2, "1", 1.0, [], {}):
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    candidate_statistics([{"label": label, "score": .5}], .5)
        for row in (None, [], {}, {"label": 1}, {"score": .5}):
            with self.subTest(row=row):
                with self.assertRaises(ValueError):
                    candidate_statistics([row], .5)
        for bins in (0, -1, .5, True, "10"):
            with self.subTest(bins=bins):
                with self.assertRaises(ValueError):
                    candidate_statistics([], .5, bins=bins)


class SeedSummaryTests(unittest.TestCase):
    def test_uses_sample_standard_deviation_and_keeps_single_seed_undefined(self):
        rows = [
            {"encoder": "a", "dataset": "real", "f1": .1},
            {"encoder": "a", "dataset": "real", "f1": .2},
            {"encoder": "a", "dataset": "real", "f1": .3},
            {"encoder": "a", "dataset": "real", "f1": None},
            {"encoder": "b", "dataset": "augmented", "f1": .9},
        ]
        original = copy.deepcopy(rows)
        groups = summarize_groups(rows, ["encoder", "dataset"], "f1")
        self.assertEqual(groups[0]["n"], 3)
        self.assertAlmostEqual(groups[0]["mean"], .2)
        self.assertAlmostEqual(groups[0]["std"], .1)
        self.assertEqual((groups[0]["min"], groups[0]["max"]), (.1, .3))
        self.assertEqual(groups[1]["n"], 1)
        self.assertIsNone(groups[1]["std"])
        self.assertEqual(rows, original)

    def test_skips_unavailable_values_but_rejects_nonfinite_values(self):
        self.assertEqual(summarize_groups([{"model": "a", "f1": None}], "model", "f1"), [])
        for value in (True, "0.1", float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    summarize_groups([{"model": "a", "f1": value}], "model", "f1")


PRIVATE_MARKER = "REPORT_FIXTURE_PRIVATE_LITERAL"


class TableOnlyPlotWriter:
    """Skip redundant image rendering in tests of report aggregation branches."""
    def __init__(self, out_dir):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        self.items = []

    def discard(self, *args, **kwargs):
        return None

    grouped = horizontal = lines = confusion = curve = reliability = histogram = discard


def build_fixture(root):
    """Write the actual suite/assess schemas, with no exported model weights."""
    from secrets_pipeline.candidates import prepare_records
    from secrets_pipeline.evaluate import evaluate_documents
    from secrets_pipeline.experiments import file_sha256
    from secrets_pipeline.schema import validate_corpus, write_json, write_jsonl

    root = Path(root)
    shutil.copytree(PROJECT_ROOT / "secrets_pipeline", root / "secrets_pipeline",
                    ignore=shutil.ignore_patterns("__pycache__"))
    specs = [
        ("val", "creddata", "python", "validation.py",
         'def validation_transport():\n    password = "{positive}"\n    region = "{negative}"\n    return password\n'),
        ("test", "creddata", "yaml", "production.yaml",
         'services:\n  api:\n    bearer: "{positive}"\n    resource: "{negative}"\n    replicas: 3\n'),
        ("test", "tier_b", "toml", "synthetic.toml",
         '[deployment.credentials]\nprivate_token = "{positive}"\npublic_id = "{negative}"\nmode = "sandbox"\n'),
        ("test_ood", "tier_b", "shell", "unseen.sh",
         '#!/bin/sh\nexport DB_PASSWORD="{positive}"\nexport TENANT="{negative}"\nprintf done\n'),
    ]
    docs = []
    for number, (split, source, language, path, template) in enumerate(specs):
        positive, negative = PRIVATE_MARKER + str(number), "PUBLIC_FIXTURE_RESOURCE" + str(number)
        text = template.format(positive=positive, negative=negative)
        spans = []
        for label, value, category in ((1, positive, "password"), (0, negative, "identifier")):
            start = text.index(value)
            spans.append({"id": f"span-{number}-{label}", "start": start, "end": start + len(value),
                          "label": label, "category": category})
        docs.append({"id": f"document-{number}", "repo_id": f"repo-{number}",
                     "group_id": f"owner-{number}", "split": split, "source": source,
                     "path": path, "language": language, "text": text, "spans": spans, "metadata": {}})
    manifest = validate_corpus(docs)
    records = prepare_records(docs, legacy_scanner="none")
    predictions = [{**record, "score": .85 if record["label"] == 1 else (.1 if record["label"] == 0 else .25)}
                   for record in records]
    dataset = root / "artifacts" / "full"
    suite = root / "artifacts" / "experiments"
    write_jsonl(dataset / "documents.jsonl", docs)
    write_jsonl(dataset / "candidates.jsonl", records)
    run_name = "codebert-real_only-plain-seed13"
    run_dir = suite / "runs" / run_name
    thresholds = {"warn": .5, "block": .9, "test_used_for_selection": False,
                  "validation_sources": ["creddata"], "calibration_precision": "fp32"}
    val_docs = [doc for doc in docs if doc["split"] == "val"]
    val_predictions = [p for p in predictions if p["split"] == "val"]
    val_metrics = evaluate_documents(val_docs, val_predictions, thresholds["warn"])
    member = {"name": "codebert", "run": run_name, "path": f"runs/{run_name}/best", "weight": 1,
              "sha256": "fictional-unloaded-model"}
    selected = {
        "name": run_name, "models": [member], "thresholds": thresholds,
        "validation_metrics": val_metrics, "candidate_average_precision": 1,
        "selection_rule": "real_validation_span_f1_then_candidate_ap_then_lower_unknown_burden_then_fewer_models",
        "test_used_for_selection": False, "validation_only": True,
        "suite_fingerprint": "fictional-suite-fingerprint", "corpus_sha256": manifest["corpus_sha256"],
        "documents_sha256": file_sha256(dataset / "documents.jsonl"),
        "input_sha256": file_sha256(dataset / "candidates.jsonl"),
    }
    write_json(suite / "selected.json", selected)
    write_json(suite / "suite_manifest.json", {"fingerprint": selected["suite_fingerprint"],
               "corpus_sha256": manifest["corpus_sha256"], "input_sha256": selected["input_sha256"],
               "documents_sha256": selected["documents_sha256"]})
    overall = val_metrics["overall"]
    write_json(suite / "leaderboard.json", {"test_used_for_selection": False, "results": [{
        "name": run_name, "models": [member], "span_precision": overall["precision"],
        "span_recall": overall["recall"], "span_f1": overall["f1"],
        "unknown_alerts_per_kloc": overall["unknown_alerts_per_kloc"], "candidate_average_precision": 1}]})
    write_jsonl(run_dir / "real_validation_all_predictions.jsonl", val_predictions)
    history = [{"epoch": 1, "training_loss_rank0": .4, "validation_average_precision": .8,
                "validation_warn_f1": .7, "seconds": 2},
               {"epoch": 2, "training_loss_rank0": .2, "validation_average_precision": 1,
                "validation_warn_f1": 1, "seconds": 3}]
    write_json(run_dir / "run.json", {"config": {"model_name": "fictional-codebert", "seed": 13},
               "history": history, "best_checkpoint": "epoch-2", "world_size": 1,
               "effective_batch_size": 8, "versions": {"fixture": "1"}, "model_revision": "fictional"})
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "best" / "config.json", {"hidden_size": 768, "num_hidden_layers": 12,
               "num_attention_heads": 12, "num_labels": 2, "architectures": ["RobertaForSequenceClassification"]})
    for split, directory in (("test", "test"), ("test_ood", "test-ood")):
        subset = [doc for doc in docs if doc["split"] == split]
        scores = [p for p in predictions if p["split"] == split]
        assessment = {"split": split, "corpus_sha256": manifest["corpus_sha256"],
                      "thresholds_tuned": False, "external_corpus": False, "thresholds": thresholds,
                      "warn": evaluate_documents(subset, scores, thresholds["warn"]),
                      "block": evaluate_documents(subset, scores, thresholds["block"])}
        write_json(root / "artifacts" / directory / "metrics.json", assessment)
        write_jsonl(root / "artifacts" / directory / "predictions.jsonl", scores)
    return {"documents": docs, "candidates": records, "selected": selected}


class ReportIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="report-bundle-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.fixture = build_fixture(self.root)
        self.out = self.root / "report-output"

    def invoke(self, *extra):
        from build_report import main
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--project-root", str(self.root), "--out", str(self.out), *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def test_complete_report_reads_saved_artifacts_and_keeps_real_test_separate(self):
        code, stdout, stderr = self.invoke()
        self.assertEqual(code, 0, stderr)
        for filename in ("report.html", "README.md", "report.json", "input_manifest.json",
                         "PROJECT_EXPLAINED.md", "HISTORICAL_RESULTS.md", "data/historical_results.json"):
            self.assertTrue((self.out / filename).is_file(), filename)
        report = json.loads((self.out / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(report["complete_required_inputs"])
        self.assertFalse(report["inference_performed"])
        self.assertFalse(report["thresholds_retuned"])
        primary = next(row for row in report["span_summary"]
                       if row["scope"] == "Real test (primary)" and row["operating_point"] == "warn")
        pooled = next(row for row in report["span_summary"]
                      if row["scope"] == "All test sources (context only)" and row["operating_point"] == "warn")
        self.assertEqual(primary["documents"], 1)
        self.assertEqual(pooled["documents"], 2)
        self.assertEqual(primary["f1"], 1)
        self.assertEqual(report["candidate_diagnostics"]["test_real"]["positive_count"], 1)
        block = report["candidate_diagnostics"]["test_real_block"]
        self.assertEqual(block["threshold"], .9)
        self.assertEqual(block["metrics"]["tp"], 0)
        self.assertEqual(block["metrics"]["fn"], 1)
        self.assertIsNone(block["metrics"]["precision"])
        self.assertTrue((self.out / "plots" / "test_real_block_confusion.png").is_file())
        self.assertEqual(report["candidate_diagnostics"]["test_ood_block"]["threshold"], .9)
        self.assertEqual(report["training_runs"][0]["accounted_epoch_seconds"], 5)
        self.assertEqual(report["validation_seed_summary"][0]["n"], 1)
        self.assertIsNone(report["validation_seed_summary"][0]["std"])
        self.assertTrue(report["historical"]["experiments"])
        self.assertTrue(list((self.out / "plots").glob("*.png")))
        self.assertTrue(list((self.out / "plots").glob("*.svg")))
        self.assertTrue((self.out / "tables" / "candidate_metrics.csv").is_file())
        html_text = (self.out / "report.html").read_text(encoding="utf-8")
        self.assertIn('src="data:image/png;base64,', html_text)
        history = (self.out / "HISTORICAL_RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("](data/historical_results.json)", history)
        saved_history = json.loads((self.out / "data" / "historical_results.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_history, report["historical"])
        for target in re.findall(r"\]\(([^)\n]+)\)", history):
            if "://" not in target and not target.startswith("#"):
                self.assertTrue((self.out / target.split("#", 1)[0]).resolve().exists(), target)
        for path in self.out.rglob("*"):
            if path.suffix in (".html", ".json", ".md", ".csv", ".svg"):
                self.assertNotIn(PRIVATE_MARKER, path.read_text(encoding="utf-8"), str(path))

    def test_required_missing_predictions_fail_before_output_creation(self):
        (self.root / "artifacts" / "test-ood" / "predictions.jsonl").unlink()
        code, stdout, stderr = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("Required artifact missing", stderr)
        self.assertIn("predictions.jsonl", stderr)
        self.assertFalse(self.out.exists())

    def test_partial_mode_prominently_lists_missing_required_predictions(self):
        (self.root / "artifacts" / "test-ood" / "predictions.jsonl").unlink()
        code, stdout, stderr = self.invoke("--allow-partial")
        self.assertEqual(code, 0, stderr)
        report = json.loads((self.out / "report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["complete_required_inputs"])
        self.assertIn({"path": "artifacts/test-ood/predictions.jsonl", "required": True}, report["missing_artifacts"])
        self.assertIn("INCOMPLETE REPORT", (self.out / "README.md").read_text(encoding="utf-8"))
        self.assertNotIn("test_ood", report["candidate_diagnostics"])

    def test_inconsistent_saved_metrics_fail_even_in_partial_mode(self):
        path = self.root / "artifacts" / "test" / "metrics.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["warn"]["overall"]["f1"] = .123
        path.write_text(json.dumps(saved), encoding="utf-8")
        code, stdout, stderr = self.invoke("--allow-partial")
        self.assertEqual(code, 2)
        self.assertIn("disagree", stderr)
        self.assertFalse(self.out.exists())

    def test_modified_frozen_corpus_is_rejected(self):
        path = self.root / "artifacts" / "full" / "documents.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        code, stdout, stderr = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("hash", stderr.lower())
        self.assertFalse(self.out.exists())

    def test_saved_threshold_mismatch_is_rejected(self):
        path = self.root / "artifacts" / "test" / "metrics.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        saved["thresholds"]["warn"] = .2
        path.write_text(json.dumps(saved), encoding="utf-8")
        code, stdout, stderr = self.invoke()
        self.assertEqual(code, 2)
        self.assertIn("threshold differs", stderr)
        self.assertFalse(self.out.exists())


    def test_ensemble_first_keeps_member_seed_summaries_and_deduplicates_run_files(self):
        suite = self.root / "artifacts" / "experiments"
        selected = json.loads((suite / "selected.json").read_text(encoding="utf-8"))
        first_member = selected["models"][0]
        second_name = "unixcoder-augmented-plain-seed37"
        shutil.copytree(suite / "runs" / first_member["run"], suite / "runs" / second_name)
        second_member = {**first_member, "name": "unixcoder", "run": second_name,
                         "path": f"runs/{second_name}/best"}
        selected["name"] = "mean-of-family-winners"
        selected["models"] = [{**first_member, "weight": .5}, {**second_member, "weight": .5}]
        (suite / "selected.json").write_text(json.dumps(selected), encoding="utf-8")
        board = json.loads((suite / "leaderboard.json").read_text(encoding="utf-8"))
        first = board["results"][0]
        second = {**first, "name": second_name, "models": [second_member]}
        ensemble = {**first, "name": selected["name"], "models": selected["models"]}
        board["results"] = [ensemble, first, second]
        (suite / "leaderboard.json").write_text(json.dumps(board), encoding="utf-8")
        with patch("plotting.PlotWriter", TableOnlyPlotWriter):
            code, stdout, stderr = self.invoke()
        self.assertEqual(code, 0, stderr)
        report = json.loads((self.out / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["selected_name"], "mean-of-family-winners")
        self.assertEqual(len(report["training_runs"]), 2)
        self.assertEqual({row["model"] for row in report["validation_seed_summary"]}, {"codebert", "unixcoder"})
        self.assertEqual([row["n"] for row in report["validation_seed_summary"]], [1, 1])
        self.assertEqual({row["dataset"] for row in report["validation_seed_summary"]}, {"real_only", "augmented"})
        self.assertEqual(sum(row["accounted_epoch_seconds"] for row in report["training_runs"]), 10)

    def test_partial_missing_documents_does_not_invent_corpus_or_pool_real_diagnostics(self):
        (self.root / "artifacts" / "full" / "documents.jsonl").unlink()
        self.assertTrue((self.root / "artifacts" / "full" / "candidates.jsonl").exists())
        with patch("plotting.PlotWriter", TableOnlyPlotWriter):
            code, stdout, stderr = self.invoke("--allow-partial")
        self.assertEqual(code, 0, stderr)
        report = json.loads((self.out / "report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["complete_required_inputs"])
        self.assertEqual(report["dataset_composition"], [])
        self.assertIn({"path": "artifacts/full/documents.jsonl", "required": True}, report["missing_artifacts"])
        real = report["candidate_diagnostics"].get("test_real")
        if real is not None:
            self.assertEqual(real["positive_count"], 1, "Tier-B candidate must not enter real-test diagnostics")
        self.assertIn("INCOMPLETE REPORT", (self.out / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
