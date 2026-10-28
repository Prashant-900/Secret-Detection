import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from secrets_pipeline.assess import main as assess_main
from secrets_pipeline.candidates import prepare_records
from secrets_pipeline.mine import mine_training_errors
from secrets_pipeline.scan import findings_from_predictions, read_target, validate_thresholds
from secrets_pipeline.smoke import fixture_documents, run_smoke
from secrets_pipeline.workflow import build_dataset


class WorkflowTests(unittest.TestCase):
    def test_complete_offline_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_smoke(Path(tmp) / "smoke")
            self.assertEqual(result["status"], "passed")
            self.assertGreater(result["documents"], 4)
            self.assertFalse(result["model_trained"])
            self.assertGreater(result["training_dry_run"]["validation_examples"], 0)

    def test_failed_rerun_cannot_keep_prepared_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "dataset_status.json").write_text('{"status":"prepared"}')
            with patch("secrets_pipeline.workflow.ingest.main", side_effect=ValueError("bad source")):
                with self.assertRaises(ValueError):
                    build_dataset(out=out, generation_config={})
            status = json.loads((out / "dataset_status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertFalse(status["usable_for_training"])

    def test_disabled_block_redacts_even_short_values(self):
        docs = fixture_documents()
        span = docs[0]["spans"][0]
        pred = {"document_id": docs[0]["id"], "path": docs[0]["path"],
                "start": span["start"], "end": span["end"], "score": .999}
        found = findings_from_predictions(docs, [pred], {"warn": .4, "block": None})
        self.assertEqual(found[0]["verdict"], "WARN")
        self.assertEqual(found[0]["match"], "[REDACTED]")
        self.assertNotIn("fictional", json.dumps(found))

    def test_scanner_skips_envs_and_reports_incomplete_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".venv-train").mkdir()
            (root / ".venv-train" / "ignored.py").write_text('password="ignore"')
            (root / ".env.production").write_text("PASSWORD=fictional\n")
            (root / "large.json").write_text("a" * 100)
            docs, skipped, errors = read_target(root, max_file_bytes=50)
            self.assertEqual([d["path"] for d in docs], [".env.production"])
            self.assertEqual(errors[0]["reason"], "file_size_limit")

    def test_mining_refuses_heldout_errors(self):
        records = prepare_records(fixture_documents())
        test = next(r for r in records if r["split"] == "test")
        with self.assertRaises(ValueError):
            mine_training_errors(records, [{**test, "score": 1.0}])

    def test_assessment_refuses_changed_frozen_inputs_before_model_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "selected.json").write_text(json.dumps({"input_sha256": "wrong", "documents_sha256": "wrong"}))
            (root / "docs.jsonl").write_text("")
            (root / "data.jsonl").write_text("")
            with self.assertRaisesRegex(ValueError, "differ"):
                assess_main(["--bundle", str(root / "selected.json"), "--documents", str(root / "docs.jsonl"),
                             "--data", str(root / "data.jsonl"), "--split", "test", "--out", str(root / "out")])


if __name__ == "__main__":
    unittest.main()
