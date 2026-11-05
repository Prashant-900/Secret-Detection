"""Package portable project files without machine-specific environments/runs."""
import argparse
from pathlib import Path
import zipfile


# Deliberately enumerate source locations: a copied checkout can contain large
# downloaded corpora and model outputs even when Git is unavailable.
SOURCE_PATTERNS = (
    "README.md", ".gitignore", ".gitattributes",
    "secrets_pipeline/**/*.py", "configs/*.json", "requirements/*.txt",
    "scripts/*.py", "scripts/*.sh", "tests/**/*.py",
    "Dataset/pipeline_bundle/tier_a/manifest_v2.csv",
    "Dataset/pipeline_bundle/tier_a/splits.csv",
    "Dataset/pipeline_bundle/tier_a/fetch_repos.sh",
    "Dataset/pipeline_bundle/tier_a/pipeline/*.py",
    "Scanner/scanner/*.py", "Scanner/scanner/*.toml", "Scanner/scanner/*.json",
    "Scanner/training/*.py", "Scanner/evaluation/setup_tools.sh",
    "reporting_bundle/*.py", "reporting_bundle/*.sh", "reporting_bundle/requirements.txt",
    "reporting_bundle/data/historical_results.json", "reporting_bundle/tests/**/*.py",
)
EXCLUDED_PARTS = {".git", ".runtime", "__pycache__", ".pytest_cache", ".hf-cache",
                  "node_modules", "artifacts", "local_docs", ".matplotlib", "output"}


def source_files(root):
    """Return portable sources without walking downloaded data or run folders."""
    root = Path(root).resolve()
    selected = set()
    for pattern in SOURCE_PATTERNS:
        for path in root.glob(pattern):
            rel = path.relative_to(root)
            if any(part in EXCLUDED_PARTS or part.startswith(".venv") for part in rel.parts):
                continue
            if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
                continue
            if path.is_file() and path.resolve().is_relative_to(root):
                selected.add(path)
    return sorted(selected)


def package_project(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out.exists():
        raise SystemExit("Output already exists; choose a new archive name")
    out.parent.mkdir(parents=True, exist_ok=True)
    files = [path for path in source_files(root) if path != out]
    with zipfile.ZipFile(out, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, "ISP/" + path.relative_to(root).as_posix())
    return len(files)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/ISP-pipeline.zip")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    count = package_project(root, out)
    print(f"Packaged {count} files into {out}")


if __name__ == "__main__":
    main()
