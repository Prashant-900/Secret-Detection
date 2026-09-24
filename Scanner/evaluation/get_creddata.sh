#!/usr/bin/env bash
# Gets the CredData benchmark (https://github.com/Samsung/CredData).
#   ./evaluation/get_creddata.sh quick [N]   -> N random repos (default 15)  -> datasets/CredData_quick
#   ./evaluation/get_creddata.sh full        -> all 337 repos (big, slow)     -> datasets/CredData
# CredData has no data in git: its download_data.py clones the original repos at pinned
# commits and obfuscates the credential values. Needs git + internet. Python 3.10-3.12 (not 3.13).
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-quick}; n=${2:-15}
case "$mode" in
  quick) dir=datasets/CredData_quick;;
  full)  dir=datasets/CredData;;
  *) echo "usage: $0 quick [N] | full"; exit 1;;
esac

# REMOVED: Virtual environment check line

mkdir -p datasets
if [ ! -d "$dir/.git" ]; then
  git clone --depth 1 https://github.com/Samsung/CredData.git "$dir"
fi

# CHANGED: Uses active environment's pip
pip install -q -r "$dir/requirements.txt"

if [ "$mode" = quick ] && [ ! -f "$dir/.subset_done" ]; then
  # CHANGED: Uses active environment's python
  python evaluation/subset_creddata.py "$dir" "$n" && touch "$dir/.subset_done"
fi
if [ -d "$dir/data" ]; then
  echo "$dir/data already exists - nothing to do (delete it to re-download)"; exit 0
fi
cd "$dir"
JOBS=${JOBS:-8}

# CHANGED: Uses active environment's python
python download_data.py --data_dir data --jobs "$JOBS"
echo "CredData ready in $dir"
