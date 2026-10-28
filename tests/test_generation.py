"""Contract tests for generation, labels, parser failures, and resumability."""
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from secrets_pipeline.generate import (build_plan, fill_variants, generate_dataset,
                                       LocalBackend, mock_scaffold, normalize_config, scaffold_gate,
                                       topic_split)
from secrets_pipeline.validation import DuplicateIndex, fingerprint, masked_text, validate_spans, validate_syntax


def config(**overrides):
    raw = {"seed": 29, "families": ["json_config", "python", "toml"],
           "samples_per_family": 3, "workers": 2, "retries": 2,
           "split_ratios": {"train": 1.0, "val": 0.0, "test": 0.0}}
    raw.update(overrides)
    return normalize_config(raw, mock=True)


class GenerationTests(unittest.TestCase):
    def test_exact_spans_counterfactuals_and_no_label_path(self):
        cfg = config()
        for item in build_plan(cfg):
            docs = fill_variants(mock_scaffold(item), item, cfg)
            self.assertEqual([sum(s["label"] for s in d["spans"]) for d in docs], [1, 0, 0])
            self.assertEqual(len({d["group_id"] for d in docs}), 1)
            self.assertEqual(len({d["path"] for d in docs}), 1)
            self.assertEqual(len({d["repo_id"] for d in docs}), 1)
            for doc in docs:
                self.assertTrue(validate_spans(doc))
                self.assertTrue(doc["metadata"]["validation"]["parsed"])
                self.assertNotIn("@@", doc["text"])
                self.assertEqual(len(doc["spans"]), 6)
                for span in doc["spans"]:
                    self.assertEqual(span["surface_sha256"], hashlib.sha256(doc["text"][span["start"]:span["end"]].encode()).hexdigest())

    def test_reproducibility_and_resume_integrity(self):
        cfg = config(families=["json_config"], samples_per_family=3)
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "run"
            report = generate_dataset(cfg, out, mock=True)
            first = (out / "documents.jsonl").read_bytes()
            self.assertEqual(report["status"], "complete")
            self.assertGreater(report["documents"], 0)
            resumed = generate_dataset(cfg, out, mock=True)
            self.assertEqual(first, (out / "documents.jsonl").read_bytes())
            self.assertEqual(report, resumed)
            with self.assertRaises(ValueError):
                generate_dataset({**cfg, "seed": 30}, out, mock=True)
            shard_path = next((out / "shards").glob("*.json"))
            shard = json.loads(shard_path.read_text())
            shard["documents"][0]["text"] += "tamper"
            shard_path.write_text(json.dumps(shard))
            with self.assertRaisesRegex(ValueError, "integrity"):
                generate_dataset(cfg, out, mock=True)

    def test_same_topic_different_plan_items_do_not_bypass_dedup(self):
        cfg = config(families=["json_config"], retries=1,
                     cells=[{"family": "json_config", "count": 3, "topic": "same topic"}])
        first = build_plan(cfg)[0]
        repeated = mock_scaffold(first)
        with tempfile.TemporaryDirectory() as temp, patch("secrets_pipeline.generate.mock_scaffold", return_value=repeated):
            report = generate_dataset(cfg, Path(temp) / "run", mock=True)
        self.assertEqual(report["accepted_groups"], 1)
        self.assertEqual(report["rejected_groups"], 2)

    def test_group_split_freeze_and_ood_reservation(self):
        cfg = config(families=["json_config", "jenkins", "pulumi"],
                     include_splits=["train", "val", "test", "test_ood"],
                     split_ratios={"train": .7, "val": .15, "test": .15})
        plan = build_plan(cfg)
        groups = {}
        for item in plan:
            groups.setdefault(item["group_id"], set()).add(item["split"])
            if item["family"] in {"jenkins", "pulumi"}:
                self.assertEqual(item["split"], "test_ood")
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        cfg["include_splits"] = ["train"]
        self.assertTrue(all(item["family"] not in {"jenkins", "pulumi"} for item in build_plan(cfg)))
        self.assertEqual(topic_split("jenkins", "any topic", cfg), "test_ood")

    def test_parser_fail_closed_and_unvalidated_explicit(self):
        self.assertFalse(validate_syntax('{"x":', "json").valid)
        self.assertFalse(validate_syntax('{"x":1,"x":2}', "json").valid)
        self.assertFalse(validate_syntax('<!DOCTYPE foo><a/>', "xml").valid)
        with patch.dict("sys.modules", {"hcl2": None}):
            result = validate_syntax('locals { value = "x" }', "hcl")
            self.assertFalse(result.valid)
            self.assertFalse(result.parsed)
            relaxed = validate_syntax('locals { value = "x" }', "hcl", False)
            self.assertTrue(relaxed.valid)
            self.assertFalse(relaxed.parsed)

    def test_masked_dedup_catches_new_values_across_splits(self):
        cfg = config(families=["json_config"], samples_per_family=1)
        item = build_plan(cfg)[0]
        doc = fill_variants(mock_scaffold(item), item, cfg)[0]
        other_item = {**item, "id": "other", "seed": item["seed"] + 500}
        other = fill_variants(mock_scaffold(item), other_item, cfg)[0]
        self.assertNotEqual(doc["text"], other["text"])
        bank = DuplicateIndex(.85)
        bank.add(masked_text(doc), "one", "test")
        self.assertTrue(bank.match(masked_text(other), "two", "train")["cross_split"])
        self.assertIsNone(bank.match(masked_text(doc), "one", "test"))

    def test_prefix_dedup_matches_brute_force_across_frequency_freeze(self):
        rng = random.Random(804)
        for threshold in (.3, .6, .85, 1.0):
            bank = DuplicateIndex(threshold)
            for index in range(145):
                words = [f"token{rng.randrange(50)}" for _ in range(45)]
                text = " ".join(words)
                bank.add(text, str(index), "test")
                changed = words[:]
                changed[rng.randrange(len(changed))] = "replacement"
                query = " ".join(changed)
                _, shingles = fingerprint(query)
                brute = any(len(shingles & entry["shingles"]) / len(shingles | entry["shingles"]) >= threshold
                            for entry in bank.entries)
                self.assertEqual(bank.match(query, "new", "train") is not None, brute)
            self.assertIsNotNone(bank.frozen_frequency)

    def test_cross_split_only_does_not_stop_at_same_split_clone(self):
        text = "\n".join(f'config_{i} = "setting_{i}"' for i in range(30))
        cross = text.replace("config_0", "renamed_0")
        bank = DuplicateIndex(.85)
        bank.add(text, "old_train", "train")
        self.assertIsNone(bank.match(text, "new_train", "train", cross_split_only=True))
        bank.add(cross, "test_repo", "test")
        self.assertFalse(bank.match(text, "new_train", "train")["cross_split"])
        match = bank.match(text, "new_train", "train", cross_split_only=True)
        self.assertTrue(match["cross_split"])
        self.assertEqual(match["group_id"], "test_repo")
        # Cover split-partitioned exact lookup as well as the near-match path.
        bank.add(text, "test_exact", "test")
        self.assertEqual(bank.match(text, "new_train", "train", cross_split_only=True)["group_id"], "test_exact")

    def test_unknown_markers_and_unmarked_credentials_rejected(self):
        cfg = config(families=["python"], samples_per_family=1)
        item = build_plan(cfg)[0]
        scaffold = mock_scaffold(item)
        with self.assertRaisesRegex(ValueError, "unmarked_sensitive_assignment"):
            scaffold_gate(scaffold + '\npassword = "another-password"\n')
        with self.assertRaises(ValueError):
            scaffold_gate(scaffold.replace("@@ID:identifier@@", "@@OTHER:oops@@"))

    def test_multiline_secret_has_one_span_and_correct_surface(self):
        cfg = config(families=["python"], cells=[{"family": "python", "secret_type": "private_key_pem", "count": 1}])
        item = build_plan(cfg)[0]
        pem = "-----BEGIN PRIVATE KEY-----\nQUJDREVGR0g=\n-----END PRIVATE KEY-----"
        with patch("secrets_pipeline.generate.secret_value", return_value=pem):
            doc = fill_variants(mock_scaffold(item), item, cfg)[0]
        positives = [span for span in doc["spans"] if span["label"]]
        self.assertEqual(len(positives), 1)
        span = positives[0]
        self.assertEqual(doc["text"][span["start"]:span["end"]], pem)
        self.assertTrue(validate_syntax(doc["text"], "python").valid)

    def test_local_backend_request_contract_and_truncation(self):
        cfg = config()
        item = build_plan(cfg)[0]
        payload = {"choices": [{"finish_reason": "stop", "message": {"content": "source"}}]}
        with patch("urllib.request.OpenerDirector.open") as open_mock:
            open_mock.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
            self.assertEqual(LocalBackend(cfg["backend"]).complete(item, 0), "source")
            request = open_mock.call_args.args[0]
            self.assertEqual(request.full_url, "http://127.0.0.1:8000/v1/chat/completions")
            body = json.loads(request.data)
            self.assertIn("@@SECRET:primary@@", body["messages"][1]["content"])
            self.assertEqual(body["model"], cfg["backend"]["model"])
            payload["choices"][0]["finish_reason"] = "length"
            open_mock.return_value.__enter__.return_value.read.return_value = json.dumps(payload).encode()
            with self.assertRaisesRegex(ValueError, "truncated"):
                LocalBackend(cfg["backend"]).complete(item, 0)

    def test_exclusion_hash_and_cross_corpus_gate(self):
        cfg = config(families=["json_config"], samples_per_family=1, retries=1)
        item = build_plan(cfg)[0]
        known = fill_variants(mock_scaffold(item), item, cfg)[0]
        known["id"], known["split"] = "held-out-real", "test"
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            exclusions = temp / "heldout.jsonl"
            exclusions.write_text(json.dumps(known) + "\n", encoding="utf-8")
            cfg["exclude_documents"] = [str(exclusions)]
            report = generate_dataset(cfg, temp / "run", mock=True)
            self.assertEqual(report["documents"], 0)
            self.assertEqual(report["status"], "needs_review")
            self.assertEqual(report["rejection_reasons"]["cross_split_duplicate"], 1)


if __name__ == "__main__":
    unittest.main()
