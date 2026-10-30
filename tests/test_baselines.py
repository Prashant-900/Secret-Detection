import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from secrets_pipeline.baselines import (BaselineError, locate_lines, materialize_snapshot,
                                        parse_gitleaks, parse_trufflehog, run_baselines, run_external)
from secrets_pipeline.schema import write_jsonl


def doc(text="password = ABC\nnext = DEF\n"):
    start = text.index("ABC") if "ABC" in text else 0
    return {"id": "d", "repo_id": "alice/repo", "group_id": "owner:alice", "source": "reviewed_real",
            "split": "test", "path": "../../config.env", "language": "dotenv", "text": text,
            "spans": [{"id": "s", "start": start, "end": start + 3, "label": 1, "category": "password"}], "metadata": {}}


class BaselineParsingTests(unittest.TestCase):
    def test_columns_are_one_based_inclusive(self):
        d = doc()
        self.assertEqual(locate_lines(d, 1, 1, 12, 14), (11, 14, "reported_columns"))

    def test_columns_use_utf8_byte_offsets(self):
        d = doc("é = ABC\n")
        # UTF-8: é occupies two bytes, so A starts at byte-column 6.
        start, end, precision = locate_lines(d, 1, 1, 6, 8)
        self.assertEqual(d["text"][start:end], "ABC")
        with self.assertRaisesRegex(BaselineError, "bisects"):
            locate_lines(d, 1, 1, 2, 2)

    def test_multiline_columns(self):
        d = doc()
        start, end, _ = locate_lines(d, 1, 2, 12, 10)
        self.assertEqual(d["text"][start:end], "ABC\nnext = DEF")

    def test_bad_lines_fail_not_silent_no_findings(self):
        with self.assertRaisesRegex(BaselineError, "outside"):
            locate_lines(doc(), 99)

    def test_gitleaks_normalization_and_dedup_strip_raw_values(self):
        d = doc()
        mapping = {"hash/config.env": d}
        raw = {"File": "/old/location/hash/config.env", "StartLine": 1, "EndLine": 1,
               "StartColumn": 12, "EndColumn": 14, "Secret": "DO_NOT_COPY", "Match": "DO_NOT_COPY", "RuleID": "rule1"}
        result = parse_gitleaks([raw, {**raw, "RuleID": "rule2"}], mapping)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["metadata"]["detectors"], ["rule1", "rule2"])
        self.assertNotIn("DO_NOT_COPY", json.dumps(result))
        self.assertEqual(parse_gitleaks(None, mapping), [])

    def test_trufflehog_uses_line_metadata_not_decoded_secret(self):
        d = doc()
        raw = {"SourceMetadata": {"Data": {"Filesystem": {"file": "hash/config.env", "line": 1}}},
               "Raw": "decoded-value-not-in-source", "DetectorName": "Custom", "DecoderName": "BASE64"}
        result = parse_trufflehog([json.dumps(raw)], {"hash/config.env": d})
        self.assertEqual(result[0]["start"], 0)
        self.assertEqual(result[0]["end"], 14)
        self.assertEqual(result[0]["metadata"]["localization"], "reported_line_range")
        self.assertNotIn("decoded-value", json.dumps(result))

    def test_unmapped_tool_file_rejected(self):
        with self.assertRaisesRegex(BaselineError, "unmapped"):
            parse_gitleaks([{"File": "elsewhere", "StartLine": 1}], {"hash/config.env": doc()})

    def test_snapshot_does_not_follow_logical_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "snapshot"
            mapping = materialize_snapshot([doc()], root)
            relative = next(iter(mapping))
            self.assertEqual(Path(relative).name, "config.env")
            self.assertTrue((root / relative).is_file())
            self.assertFalse((Path(tmp) / "config.env").exists())


class BaselineExecutionTests(unittest.TestCase):
    def test_nonzero_status_fails_and_shell_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("secrets_pipeline.baselines.subprocess.run", return_value=SimpleNamespace(returncode=2)) as runner:
                with self.assertRaisesRegex(BaselineError, "exit 2"):
                    run_external(["dummy"], root / "out.log", root / "err.log", root, 1)
                self.assertFalse(runner.call_args.kwargs["shell"])

    def test_timeout_is_explicit_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch("secrets_pipeline.baselines.subprocess.run", side_effect=subprocess.TimeoutExpired("dummy", 1)):
                with self.assertRaisesRegex(BaselineError, "exceeded"):
                    run_external(["dummy"], root / "out.log", root / "err.log", root, 1)

    def test_end_to_end_external_adapters_and_cache_without_binaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus, output = root / "documents.jsonl", root / "baselines"
            write_jsonl(corpus, [doc()])
            commands = []

            def fake(command, stdout_path, stderr_path, cwd, timeout, allowed_exit_codes=(0,)):
                commands.append(command)
                snapshot_file = next(cwd.rglob("*.env"))
                if "dir" in command:
                    report_path = Path(command[command.index("--report-path") + 1])
                    report_path.write_text(json.dumps([{"File": str(snapshot_file), "StartLine": 1, "EndLine": 1,
                                                       "StartColumn": 12, "EndColumn": 14, "RuleID": "test"}]), encoding="utf-8")
                else:
                    stdout_path.write_text(json.dumps({"SourceMetadata": {"Data": {"Filesystem": {"file": str(snapshot_file), "line": 1}}},
                                                      "DetectorName": "test"}) + "\n", encoding="utf-8")
                return {"seconds": 0, "returncode": 0}

            with patch("secrets_pipeline.baselines.run_external", side_effect=fake):
                result = run_baselines(corpus, output, tools=("gitleaks", "trufflehog"), gitleaks="fake-gitleaks", trufflehog="fake-trufflehog")
            self.assertEqual(result["tools"]["gitleaks"]["operating_points"]["binary"]["recall"], 1)
            self.assertIn("--no-verification", commands[1])
            self.assertIn("--no-update", commands[1])
            with patch("secrets_pipeline.baselines.run_external", side_effect=AssertionError("cache must not run binaries")):
                again = run_baselines(corpus, output, tools=("gitleaks", "trufflehog"), reuse_raw=True)
            self.assertEqual(again["tools"]["trufflehog"]["predictions"], 1)
            changed = doc("password = ABCXYZ\n")
            write_jsonl(corpus, [changed])
            with self.assertRaisesRegex(BaselineError, "different corpus"):
                run_baselines(corpus, output, tools=("gitleaks",), reuse_raw=True)


if __name__ == "__main__":
    unittest.main()
