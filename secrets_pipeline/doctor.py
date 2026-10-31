"""Report interpreter, parsers, ML dependencies, and GPU availability."""
import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path


def inspect_environment():
    packages = {}
    for name in ("torch", "transformers", "vllm", "PyYAML", "python-hcl2", "cryptography", "tree-sitter-language-pack"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
              "reference_python": "3.12", "packages": packages, "gpus": []}
    if packages["torch"]:
        import torch
        result["torch_cuda"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            result["gpus"].append({"index": i, "name": p.name, "memory_gib": round(p.total_memory / 2**30, 1)})
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    result = inspect_environment()
    if args.out:
        from .schema import write_json
        write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
