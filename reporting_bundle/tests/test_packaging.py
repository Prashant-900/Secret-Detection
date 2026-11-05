"""Source releases remain portable without including local corpora or notes."""
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.package_release import package_project


class PackagingTests(unittest.TestCase):
    def test_archive_without_git_keeps_runtime_inputs_but_excludes_local_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            included = {
                "README.md", ".gitignore", ".gitattributes",
                "secrets_pipeline/scan.py", "scripts/run_dataset.sh",
                "Scanner/scanner/rules.toml", "Scanner/scanner/model_v2.json",
                "reporting_bundle/build_report.py", "reporting_bundle/data/historical_results.json",
                "Dataset/pipeline_bundle/tier_a/manifest_v2.csv",
                "Dataset/pipeline_bundle/tier_a/splits.csv",
            }
            excluded = {
                "local_docs/README.md", "local_docs/example.py", ".git/config",
                "artifacts/history-reconstruction/reconstruct.py", "artifacts/run/model.json",
                "Scanner/datasets/example.json", "Scanner/tools/bin/scanner.exe",
                "Scanner/results/raw/hits.json", "Scanner/.venv/lib/private.py",
                "Dataset/pipeline_bundle/tier_a/out/records.json",
                "Dataset/pipeline_bundle/tier_a/preexisting_hits.csv",
                "Dataset/pipeline_bundle/tier_a/manifest.csv",
                "Dataset/pipeline_bundle/tier_a/requirements.txt",
                "Dataset/pipeline_bundle/tier_a/pipeline/gpu/train_codebert.py",
                "Dataset/pipeline_bundle/tier_a/repos/downloaded/source.py",
                "Scanner/evaluation/evaluate_all.py", "creddata_seed_repos.json",
                "reporting_bundle/output/report.json", "reporting_bundle/.matplotlib/config.json",
                "secrets_pipeline/__pycache__/cached.py", "secrets_pipeline/.venv/private.py",
                "reporting_bundle/README.md",
            }
            for name in included | excluded:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(name, encoding="utf-8")
            destination = root / "release.zip"
            self.assertEqual(package_project(root, destination), len(included))
            with zipfile.ZipFile(destination) as archive:
                self.assertEqual(set(archive.namelist()), {"ISP/" + name for name in included})
            before = destination.read_bytes()
            with self.assertRaisesRegex(SystemExit, "already exists"):
                package_project(root, destination)
            self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
