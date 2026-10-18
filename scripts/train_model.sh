#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${TRAIN_GPUS:?Set TRAIN_GPUS to allocated GPU ids, e.g. TRAIN_GPUS=6}"
export CUDA_VISIBLE_DEVICES="$TRAIN_GPUS"
IFS=',' read -r -a DEVICES <<< "$TRAIN_GPUS"
if [[ "${#DEVICES[@]}" -gt 1 ]]; then
  exec "$ROOT/.venv-train/bin/torchrun" --standalone --nproc_per_node="${#DEVICES[@]}" -m secrets_pipeline.train "$@"
fi
exec "$ROOT/.venv-train/bin/python" -m secrets_pipeline train "$@"
