#!/usr/bin/env bash
# Run from any directory: bash /path/to/ISP/scripts/setup_remote.sh [data|train|generator|all]
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
MODE="${1:-all}"
case "$MODE" in data|train|generator|all) ;; *) echo 'Expected data, train, generator, or all' >&2; exit 2;; esac
mkdir -p artifacts/setup

# Use existing Python 3.12 or obtain a private interpreter with installed Conda/uv.
# The server's Python 3.14 and base Conda environment are never modified.
if command -v python3.12 >/dev/null 2>&1; then
  PY312="$(command -v python3.12)"
elif [[ -x "$ROOT/.runtime/python312/bin/python" ]]; then
  PY312="$ROOT/.runtime/python312/bin/python"
elif command -v conda >/dev/null 2>&1; then
  conda create --yes --prefix "$ROOT/.runtime/python312" python=3.12 pip
  PY312="$ROOT/.runtime/python312/bin/python"
elif command -v uv >/dev/null 2>&1; then
  uv python install 3.12
  PY312="$(uv python find 3.12)"
else
  echo 'Install Python 3.12, Conda, or uv, then rerun this script. The server shown has Conda in base.' >&2
  exit 2
fi
"$PY312" -c 'import sys; assert sys.version_info[:2] == (3,12), sys.version'

make_env() {
  local envdir="$1"
  if [[ ! -x "$envdir/bin/python" ]]; then "$PY312" -m venv "$envdir"; fi
  "$envdir/bin/python" -c 'import sys; assert sys.version_info[:2] == (3,12), "Use a new Python 3.12 venv"'
  "$envdir/bin/python" -m pip install --upgrade 'pip==25.3'
}

if [[ "$MODE" == data || "$MODE" == train || "$MODE" == all ]]; then
  make_env "$ROOT/.venv-train"
  "$ROOT/.venv-train/bin/python" -m pip install -r requirements/data.txt
  if [[ "$MODE" != data ]]; then
    "$ROOT/.venv-train/bin/python" -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
    "$ROOT/.venv-train/bin/python" -m pip install -r requirements/train.txt
  fi
  "$ROOT/.venv-train/bin/python" -m pip check
  "$ROOT/.venv-train/bin/python" -m pip freeze > artifacts/setup/train.lock.txt
  "$ROOT/.venv-train/bin/python" -m secrets_pipeline doctor --out artifacts/setup/train-environment.json
fi
if [[ "$MODE" == generator || "$MODE" == all ]]; then
  make_env "$ROOT/.venv-generator"
  "$ROOT/.venv-generator/bin/python" -m pip install -r requirements/generator.txt
  "$ROOT/.venv-generator/bin/python" -m pip check
  "$ROOT/.venv-generator/bin/python" -m pip freeze > artifacts/setup/generator.lock.txt
fi
echo 'Setup complete. Use .venv-train/bin/python for the pipeline; .venv-generator/bin/vllm for generation serving.'
