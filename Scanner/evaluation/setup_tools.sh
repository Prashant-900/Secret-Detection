#!/usr/bin/env bash
# Installs everything the evaluation needs (Go tools locally, Python packages in active environment):
#   tools/bin/gitleaks      (pinned release binary)
#   tools/bin/trufflehog    (pinned release binary)
#   Active Conda Env        (python packages with detect-secrets, matplotlib, tomli)
# Works on Linux and macOS (x86_64 / arm64). On Windows use WSL.
set -euo pipefail
cd "$(dirname "$0")/.."

GITLEAKS_V=8.24.3        # Pinned baseline detector version
TRUFFLEHOG_V=3.88.0
DETECT_SECRETS_V=1.5.0

os=$(uname -s | tr '[:upper:]' '[:lower:]')
case "$os" in linux|darwin) ;; *) echo "Unsupported OS '$os' - use WSL on Windows."; exit 1;; esac
case "$(uname -m)" in
  x86_64|amd64)  gl_arch=x64;   th_arch=amd64;;
  aarch64|arm64) gl_arch=arm64; th_arch=arm64;;
  *) echo "Unsupported CPU $(uname -m)"; exit 1;;
esac

mkdir -p tools/bin tools/tmp
if [ ! -x tools/bin/gitleaks ]; then
  echo ">> gitleaks v$GITLEAKS_V"
  curl -fsSL --retry 3 -o tools/tmp/gl.tgz \
    "https://github.com/gitleaks/gitleaks/releases/download/v${GITLEAKS_V}/gitleaks_${GITLEAKS_V}_${os}_${gl_arch}.tar.gz"
  tar -xzf tools/tmp/gl.tgz -C tools/bin gitleaks
fi
if [ ! -x tools/bin/trufflehog ]; then
  echo ">> trufflehog v$TRUFFLEHOG_V"
  curl -fsSL --retry 3 -o tools/tmp/th.tgz \
    "https://github.com/trufflesecurity/trufflehog/releases/download/v${TRUFFLEHOG_V}/trufflehog_${TRUFFLEHOG_V}_${os}_${th_arch}.tar.gz"
  tar -xzf tools/tmp/th.tgz -C tools/bin trufflehog
fi
rm -rf tools/tmp

# Python check: Make sure active Conda environment meets version criteria
ok_py() { "$1" -c 'import sys; v=sys.version_info[:2]; sys.exit(0 if v in ((3,10),(3,11),(3,12)) or v>=(3,14) else 1)' 2>/dev/null; }

# Force it to evaluate the current active environment's Python
if ! ok_py python; then
  echo "Error: The active Python environment version ($(python -V)) is incompatible."
  echo "Need Python 3.10, 3.11, 3.12 or 3.14+ (3.13 breaks CredData's download_data.py)."
  exit 1
fi

echo ">> using $(python -V) (Conda Environment)"

# CHANGED: Installs packages straight into the active environment's pip
pip install -q --upgrade pip
pip install -q "detect-secrets==${DETECT_SECRETS_V}" matplotlib tomli credsweeper scikit-learn numpy

echo
echo "Installed:"
tools/bin/gitleaks version
tools/bin/trufflehog --version 2>&1 | head -1
python -m detect_secrets --version
