#!/usr/bin/env bash
# Download the original benchmark in an isolated collector environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DEST="${1:-Scanner/datasets/CredData}"
if [[ ! -x .venv-train/bin/python ]]; then
  echo 'First run: bash scripts/setup_remote.sh data' >&2
  exit 2
fi
if [[ ! -x .venv-collector/bin/python ]]; then .venv-train/bin/python -m venv .venv-collector; fi
if [[ ! -d "$DEST/.git" ]]; then
  git clone --depth 1 https://github.com/Samsung/CredData.git "$DEST"
fi
.venv-collector/bin/python -m pip install -r "$DEST/requirements.txt"
mkdir -p artifacts/setup
git -C "$DEST" rev-parse HEAD > artifacts/setup/creddata-revision.txt
COLLECTOR="$ROOT/.venv-collector/bin/python"
cd "$DEST"
if [[ -d data ]]; then
  echo 'Existing data directory retained. The importer will report missing/inconsistent files.'
  echo 'If a previous download was interrupted, rerun download_data.py manually in this collector environment.'
else
  "$COLLECTOR" download_data.py --data_dir data --jobs "${JOBS:-8}"
fi
