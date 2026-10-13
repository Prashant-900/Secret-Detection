#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
: "${GENERATOR_GPUS:?Set GENERATOR_GPUS to allocated GPU ids, e.g. GENERATOR_GPUS=4,5}"
export CUDA_VISIBLE_DEVICES="$GENERATOR_GPUS"
IFS=',' read -r -a DEVICES <<< "$GENERATOR_GPUS"
export VLLM_NO_USAGE_STATS=1
MODEL="${GENERATOR_MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct}"
ARGS=(serve "$MODEL" --host 127.0.0.1 --port "${GENERATOR_PORT:-8000}"
      --tensor-parallel-size "${#DEVICES[@]}" --dtype bfloat16
      --max-model-len "${GENERATOR_CONTEXT:-8192}" --max-num-seqs "${GENERATOR_MAX_SEQS:-16}"
      --gpu-memory-utilization "${GENERATOR_MEMORY_FRACTION:-0.85}")
if [[ -n "${GENERATOR_REVISION:-}" ]]; then ARGS+=(--revision "$GENERATOR_REVISION"); fi
exec "$ROOT/.venv-generator/bin/vllm" "${ARGS[@]}"
