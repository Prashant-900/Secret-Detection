#!/usr/bin/env bash
# One command to run everything.
#   ./evaluation/run_all.sh quick     # tools + synthetic + 15-repo CredData subset  (~minutes)
#   ./evaluation/run_all.sh full      # tools + synthetic + full CredData             (~long)
# Extra arguments are passed to evaluate_all.py, e.g.  ./evaluation/run_all.sh quick --tolerance 2
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-quick}; shift || true
bash evaluation/setup_tools.sh
bash evaluation/get_creddata.sh "$mode"
case "$mode" in quick) cd_dir=datasets/CredData_quick;; full) cd_dir=datasets/CredData;; esac

# CHANGED: Uses the active environment's python instead of .venv
python evaluation/evaluate_all.py --dataset both --creddata-dir "$cd_dir" "$@"
echo; echo "Results: results/*.md  results/*.json  results/*_chart.png"
