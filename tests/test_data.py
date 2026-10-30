import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from secrets_pipeline.audit import audit_documents, type_of
from secrets_pipeline.candidates import build_input, candidate_coverage, discover_candidates, prepare_records, preprocessing_config, supervise_candidates
from secrets_pipeline.evaluate import bootstrap_by_repo, evaluate_documents
from secrets_pipeline.ingest import SplitRegistry, load_creddata, load_tier_b, merge_corpora, main as ingest_main
from secrets_pipeline.schema import language_for, safe_child, validate_corpus, validate_document, write_jsonl


def document(text='password = "orange123"\n', *, split="train", did="a", spans=None):
    if spans is None:
        start = text.index("orange123")
        spans = [{"id": "s", "start": start, "end": start + 9, "label": 1, "category": "password"}]
    return {"id": did, "repo_id": "owner/repo", "group_id": "owner:owner", "source": "reviewed_real",
            "path": "config.py", "language": "python", "split": split, "text": text, "spans": spans, "metadata": {}}


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class SchemaTests(unittest.TestCase):
    def test_common_credential_filename_types_are_included(self):
        for path, expected in ((".env.local", "dotenv"), ("prod.env", "dotenv"), ("private.key", "pem"),
                               ("id_rsa", "pem"), (".npmrc", "config"), ("settings.cfg", "config"),
                               ("main.tf.json", "json"), ("script.ts", "typescript")):
            self.assertEqual(language_for(path), expected)

    def test_offsets_and_conflicting_labels_fail(self):
        doc = document()
        validate_document(doc)
        doc["spans"].append({**doc["spans"][0], "id": "negative", "label": 0})
        with self.assertRaisesRegex(ValueError, "conflicting"):
            validate_document(doc)

    def test_group_split_leakage_and_template_leakage(self):
        a, b = document(), document(did="b", split="test")
        with self.assertRaisesRegex(ValueError, "Cross-split"):
            validate_corpus([a, b])
        b["repo_id"], b["group_id"] = "other/repo", "owner:other"
        b["text"] = b["text"].replace("orange123", "violet456")
        with self.assertRaisesRegex(ValueError, "masked template"):
            validate_corpus([a, b])

    def test_safe_path_cannot_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                safe_child(tmp, "../outside")

    def test_registry_freezes_imported_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "splits.json"
            registry = SplitRegistry(path)
            self.assertEqual(registry.assign("owner:alice", "val"), "val")
            registry.save()
            same = SplitRegistry(path)
            self.assertEqual(same.assign("owner:alice"), "val")
            with self.assertRaisesRegex(ValueError, "Frozen split"):
                same.assign("owner:alice", "train")

    def test_quarantine_prefers_heldout_over_training_duplicate(self):
        a, b = document(), document(did="b", split="test")
        b["repo_id"], b["group_id"] = "other/repo", "owner:other"
        errors = []
        accepted = merge_corpora([a, b], errors, skip_invalid=True)
        self.assertEqual([d["id"] for d in accepted], ["b"])
        self.assertEqual(len(errors), 1)

    def test_cross_split_near_clones_are_rejected_or_quarantined(self):
        text = "\n".join(f"setting_{i} = {i}" for i in range(35)) + '\npassword = "orange123"\n'
        a = document(text)
        b = document(text.replace("setting_34 = 34", "setting_34 = 999"), did="b", split="test")
        b["repo_id"], b["group_id"] = "different/repo", "owner:different"
        with self.assertRaisesRegex(ValueError, "near-duplicate"):
            validate_corpus([a, b])
        errors = []
        self.assertEqual([d["id"] for d in merge_corpora([a, b], errors, True)], ["b"])
        self.assertIn("near-duplicate", errors[0]["error"])


class CandidateTests(unittest.TestCase):
    def test_coarse_positive_does_not_label_multiple_literals_positive(self):
        text = 'connect("username", "password")\n'
        d = document(text, spans=[{"id": "line", "start": 0, "end": len(text) - 1, "label": 1, "category": "Password",
                                  "metadata": {"annotation_precision": "line_range"}}])
        rows = prepare_records([d], legacy_scanner="none")
        self.assertGreater(len(rows), 1)
        self.assertTrue(all(row["label"] is None for row in rows))
        self.assertTrue(all(row["metadata"]["ambiguous_coarse_positive"] for row in rows))
        self.assertEqual(candidate_coverage([d], rows)["positive_spans"], 1)
        self.assertEqual(evaluate_documents([d], [dict(document_id="a", start=rows[0]["start"], end=rows[0]["end"], score=.9)])["overall"]["true_positive"], 1)

    def test_precise_positive_remains_supervised_with_multiple_proposals(self):
        d = document('connect("username", "orange123")\n')
        rows = prepare_records([d], legacy_scanner="none")
        self.assertEqual(sum(row["label"] == 1 for row in rows), 1)

    def test_discovery_does_not_depend_on_ground_truth(self):
        doc = document()
        before = discover_candidates(doc, "none")
        doc["spans"] = []
        self.assertEqual(before, discover_candidates(doc, "none"))
        rows = prepare_records([doc], legacy_scanner="none")
        self.assertTrue(rows)
        self.assertTrue(all(row["label"] is None for row in rows))

    def test_low_entropy_literal_and_target_first(self):
        text = "# context\n" * 600 + 'password = "abc"\n'
        doc = document(text, spans=[])
        found = next(p for p in discover_candidates(doc, "none") if text[p["start"]:p["end"]] == "abc")
        rendered = build_input(doc, found, max_chars=256)
        self.assertTrue(rendered.startswith("<candidate>\nabc"))
        self.assertNotIn("train", rendered)

    def test_pem_is_one_candidate_and_one_positive(self):
        key = "-----BEGIN PRIVATE KEY-----\nABCD1234\n-----END PRIVATE KEY-----"
        doc = document(key + "\n", spans=[{"id": "key", "start": 0, "end": len(key), "label": 1, "category": "pem"}])
        proposals = discover_candidates(doc, "none")
        self.assertEqual(len(proposals), 1)
        records = prepare_records([doc], legacy_scanner="none")
        self.assertEqual(candidate_coverage([doc], records)["positive_spans"], 1)

    def test_no_oracle_injection_of_missed_positives(self):
        doc = document("xy", spans=[{"id": "s", "start": 0, "end": 2, "label": 1, "category": "short"}])
        rows = prepare_records([doc], legacy_scanner="none")
        self.assertEqual(rows, [])
        self.assertEqual(candidate_coverage([doc], rows)["candidate_recall_ceiling"], 0)

    def test_legacy_provider_rules_are_present_by_default(self):
        doc = document('access_key = "AKIAABCDEFGHIJKLMNOP"\n', spans=[])
        proposals = discover_candidates(doc)
        self.assertTrue(any("legacy" in p["metadata"]["detector_sources"] for p in proposals))

    def test_training_records_bind_preprocessing_and_document_content(self):
        rows = prepare_records([document()], context_lines=2, max_chars=1024, legacy_scanner="none")
        config = rows[0]["metadata"]["preprocessing"]
        self.assertEqual(config, preprocessing_config(2, 1024, None))
        self.assertEqual(config["legacy_scanner"], "none")
        self.assertEqual(len(config["code_sha256"]["candidates"]), 64)
        self.assertIsNone(config["code_sha256"]["rules"])
        self.assertEqual(len(rows[0]["metadata"]["document_text_sha256"]), 64)


class EvaluationTests(unittest.TestCase):
    def test_bootstrap_resamples_repository_clusters_reproducibly(self):
        a, b = document(), document('password = "violet456"\n', did="b", spans=[])
        # Two documents belong to one repository and always travel together.
        span = a["spans"][0]
        pred = {"document_id": "a", "start": span["start"], "end": span["end"], "score": .9}
        report = bootstrap_by_repo([a, b], [pred], .5, repeats=30, seed=2)
        self.assertEqual(report["repositories"], 1)
        self.assertEqual(report["intervals"]["recall"], {"lower": 1.0, "upper": 1.0})
        self.assertEqual(report, bootstrap_by_repo([a, b], [pred], .5, repeats=30, seed=2))
        with self.assertRaises(ValueError):
            bootstrap_by_repo([], [], .5)

    def test_one_prediction_cannot_detect_two_secrets(self):
        doc = document("aaaa bbbb", spans=[{"id": "a", "start": 0, "end": 4, "label": 1, "category": "token"},
                                          {"id": "b", "start": 5, "end": 9, "label": 1, "category": "token"}])
        report = evaluate_documents([doc], [{"document_id": "a", "start": 0, "end": 9, "score": .9}])["overall"]
        self.assertEqual(report["true_positive"], 1)
        self.assertEqual(report["false_negative"], 1)

    def test_unknown_alerts_separate_and_missing_proposals_count(self):
        doc = document('password = "orange123"\nunknown = "other123"\n')
        start = doc["text"].index("other123")
        report = evaluate_documents([doc], [{"document_id": "a", "start": start, "end": start + 8, "score": .9}])["overall"]
        self.assertEqual(report["unknown_alerts"], 1)
        self.assertEqual(report["false_positive"], 0)
        self.assertEqual(report["false_negative"], 1)
        self.assertEqual(report["candidate_recall_ceiling"], 0)

    def test_duplicate_predictions_rejected(self):
        doc = document()
        span = doc["spans"][0]
        p = {"document_id": "a", "start": span["start"], "end": span["end"], "score": .9}
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            evaluate_documents([doc], [p, p])


class ImportTests(unittest.TestCase):
    def test_manifest_qualifies_unresolved_creddata_owner_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = document()
            d.update(source="creddata", repo_id="creddata-opaque:abc", group_id="repo:creddata-opaque:abc")
            d["metadata"]["owner_resolved"] = False
            write_jsonl(root / "input.jsonl", [d])
            with patch("builtins.print"):
                ingest_main(["--documents", str(root / "input.jsonl"), "--out", str(root / "output.jsonl")])
            report = json.loads((root / "output.manifest.json").read_text(encoding="utf-8"))
            resolution = report["creddata_owner_resolution"]
            self.assertEqual(resolution["unresolved_documents"], 1)
            self.assertEqual(resolution["unresolved_repositories"], 1)
            self.assertFalse(resolution["owner_heldout_guaranteed"])

    def test_creddata_preserves_uuid_and_x_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "meta").mkdir()
            (root / "data" / "abc").mkdir(parents=True)
            (root / "data" / "abc" / "config.py").write_text('id="abcd"\nkey="efgh"\n', encoding="utf-8")
            write_csv(root / "meta" / "abc.csv",
                      ["Id", "FilePath", "LineStart", "LineEnd", "ValueStart", "ValueEnd", "GroundTruth", "Category"],
                      [{"Id": 1, "FilePath": "data/abc/config.py", "LineStart": 1, "LineEnd": 1, "ValueStart": 4, "ValueEnd": 8, "GroundTruth": "T", "Category": "UUID"},
                       {"Id": 2, "FilePath": "data/abc/config.py", "LineStart": 2, "LineEnd": 2, "ValueStart": 5, "ValueEnd": 9, "GroundTruth": "X", "Category": "Key"}])
            errors = []
            docs = load_creddata(root, SplitRegistry(), errors, {"abc": "alice/project"})
            self.assertFalse(errors)
            self.assertEqual(docs[0]["spans"][0]["label"], 1)
            self.assertEqual(docs[0]["spans"][0]["category"], "UUID")
            self.assertEqual(len(docs[0]["spans"]), 1)
            self.assertEqual(len(docs[0]["metadata"]["unknown_spans"]), 1)
            self.assertEqual(docs[0]["group_id"], "owner:alice")

    def test_tier_b_missing_file_reported_and_multiline_joined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "files" / "present").mkdir(parents=True)
            (root / "files" / "present" / "file.txt").write_text("FIRST\nSECOND\n", encoding="utf-8")
            fields = ["file_id", "repo", "family", "split", "relpath", "fmt", "validated"]
            base = {"repo": "alice/repo", "family": "yaml", "split": "train", "relpath": "file.txt", "fmt": "text", "validated": "0"}
            write_csv(root / "files_index.csv", fields, [{**base, "file_id": "present"}, {**base, "file_id": "missing"}])
            lfields = ["file_id", "repo", "split", "line_no", "label", "category", "span_id", "span_start", "span_end"]
            lb = {"file_id": "present", "repo": "alice/repo", "split": "train", "label": "secret", "category": "pem", "span_id": "1", "span_start": 0}
            write_csv(root / "labels.csv", lfields, [{**lb, "line_no": 1, "span_end": 5}, {**lb, "line_no": 2, "span_end": 6}])
            errors = []
            docs = load_tier_b(root, SplitRegistry(), errors)
            self.assertEqual(len(errors), 1)
            self.assertEqual(len(docs), 1)
            self.assertEqual(len(docs[0]["spans"]), 1)
            self.assertEqual(docs[0]["text"][docs[0]["spans"][0]["start"]:docs[0]["spans"][0]["end"]], "FIRST\nSECOND")

    def test_audit_allocations_ignore_heldout_counts(self):
        train = document()
        heldout = document('token = "different"\n', did="b", split="test", spans=[])
        heldout["repo_id"], heldout["group_id"] = "bob/project", "owner:bob"
        before = audit_documents([train])["recommended_generation_cells"]
        after = audit_documents([train, heldout])["recommended_generation_cells"]
        self.assertEqual(before, after)

    def test_audit_uses_shared_provider_taxonomy_and_seed_catalog(self):
        report = audit_documents([document()], max_cells=16)
        types = {row["secret_type"] for row in report["recommended_generation_cells"]}
        self.assertGreaterEqual(len(types), 12)
        self.assertNotIn("jenkins", {row["family"] for row in report["recommended_generation_cells"]})
        self.assertEqual(type_of({"category": "Auth:Basic Authorization"}), "basic_auth")
        self.assertEqual(type_of({"secret_type": "huggingface_token"}), "huggingface_token")
        self.assertEqual(type_of({"category": "Password"}), "password_unspecified")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seeds.json"
            path.write_text(json.dumps({"formats": {".env": [{"full_name": "alice/project"}],
                                                       ".xml": [{"full_name": "alice/project"}]}}), encoding="utf-8")
            catalog = audit_documents([document()], seed_catalog=path)["seed_catalog"]
            self.assertEqual(catalog["unique_repositories"], 1)
            self.assertFalse(catalog["licenses_verified"])


if __name__ == "__main__":
    unittest.main()
