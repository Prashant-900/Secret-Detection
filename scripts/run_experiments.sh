#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${TRAIN_GPUS:?Set TRAIN_GPUS to one allocated GPU id for the sequential experiment suite}"
if [[ "$TRAIN_GPUS" == *,* ]]; then
  echo 'The experiment suite trains sequentially on one GPU. For multi-GPU single runs use scripts/train_model.sh.' >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
exec "$ROOT/.venv-train/bin/python" -m secrets_pipeline experiments "$@"
