#!/usr/bin/env bash
# Serve Inkling-2-Mini for agentic coding with ~512K tokens of context, the profile picked by the GPU:
#   RTX 3090 / 4090 (24 GB): INT8 dense weights, k8v4 KV cache (8-bit keys, 4-bit values), embedding table in host
#                            RAM, no image tower, prefill chunk 2048.
#   RTX 5090 (32 GB):        FP8 dense weights and FP8 KV cache (its hardware path), images on, prefill chunk 3072;
#                            512,000 tokens fit on paper (~8 GiB of KV beside ~20.4 GiB) but are untested.
# All three: the DSpark drafter, and conversations parked in host RAM when another one runs.
#   ./start.sh <Inkling-2-Mini folder> [more ingine-serve flags; use --dense / --kv-cache (not --dense-fp8 /
#   --kv-fp8) to change the profile's formats, and MAX_CTX for the context]
# Environment: PORT (8003), MAX_CTX (512000: the 24 GB cards hold 524,288 but keep ~150 MiB of headroom), BIN.
set -euo pipefail
MODEL=${1:?usage: ./start.sh <Inkling-2-Mini folder> [more ingine-serve flags]}
shift
ROOT=$(cd "$(dirname "$0")" && pwd)
cc=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES:-0}" --query-gpu=compute_cap --format=csv,noheader | head -1)
case "$cc" in
  8.6 | 8.9)
    BUILD=build-ampere
    PROFILE=(--dense int8 --kv-cache k8v4 --embed-host --no-vision --prefill-chunk 2048) ;;
  12.*)
    BUILD=build-install
    PROFILE=(--dense fp8 --kv-cache fp8 --prefill-chunk 3072) ;;
  *)
    echo "start.sh: no profile for this GPU (compute capability $cc)" >&2
    exit 1 ;;
esac
BIN=${BIN:-$ROOT/native/$BUILD/apps/serve/ingine-serve}
exec "$BIN" --engine --model-dir "$MODEL" --port "${PORT:-8003}" --max-ctx "${MAX_CTX:-512000}" \
  --spec --spec-threshold 0.8 "${PROFILE[@]}" "$@"
