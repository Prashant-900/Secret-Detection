#!/usr/bin/env bash
set -euo pipefail
BUNDLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname -- "$BUNDLE_DIR")"
REPORT_PYTHON="${REPORT_PYTHON:-$PROJECT_DIR/.venv-train/bin/python}"
if [[ ! -x "$REPORT_PYTHON" ]]; then
  printf '%s\n' "Python not found: $REPORT_PYTHON" "Set REPORT_PYTHON=/path/to/python or use .venv-train/bin/python reporting_bundle/build_report.py" >&2
  exit 2
fi
exec "$REPORT_PYTHON" "$BUNDLE_DIR/build_report.py" --project-root "$PROJECT_DIR" "$@"
