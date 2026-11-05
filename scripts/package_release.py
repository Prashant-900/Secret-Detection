"""Package portable project files without machine-specific environments/runs."""
import argparse
from pathlib import Path
import zipfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/ISP-pipeline.zip")
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = Path(args.out).resolve()
    if out.exists():
        raise SystemExit("Output already exists; choose a new archive name")
    out.parent.mkdir(parents=True, exist_ok=True)
    excluded = {".git", ".runtime", "__pycache__", ".pytest_cache", ".hf-cache", "node_modules", "artifacts"}
    count = 0
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if path.is_symlink() or not path.is_file() or path.resolve() == out:
                continue
            if any(part in excluded or part.startswith(".venv") for part in rel.parts):
                continue
            if path.suffix in (".pyc", ".zip"):
                continue
            archive.write(path, "ISP/" + rel.as_posix())
            count += 1
    print(f"Packaged {count} files into {out}")


if __name__ == "__main__":
    main()
