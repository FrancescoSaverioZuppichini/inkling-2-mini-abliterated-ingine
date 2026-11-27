#!/usr/bin/env bash
# Serve Inkling-2-Mini with ~512K tokens of context on one 24 GB RTX 3090 or RTX 4090: INT8 dense weights, a k8v4 KV
# cache (8-bit keys, 4-bit values), the DSpark drafter, and conversations parked in host RAM when another one runs.
#   ./start.sh <Inkling-2-Mini folder> [more ingine-serve flags]
# Environment: PORT (8003), MAX_CTX (per GPU below), BIN (the ingine-serve to run).
set -euo pipefail
MODEL=${1:?usage: ./start.sh <Inkling-2-Mini folder> [more ingine-serve flags]}
shift
BIN=${BIN:-$(dirname "$0")/native/build-ampere/apps/serve/ingine-serve}
# The largest context that loads with 300 MiB to spare (round-5 memory map); the 4090's driver keeps 481 MiB more.
case "$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)" in
  8.6) CTX=524288 ;;  # RTX 3090
  8.9) CTX=507904 ;;  # RTX 4090
  *) CTX=491520 ;;
esac
exec "$BIN" --engine --model-dir "$MODEL" --port "${PORT:-8003}" --max-ctx "${MAX_CTX:-$CTX}" \
  --dense int8 --kv-cache k8v4 --spec --spec-threshold 0.8 --no-vision --embed-host --prefill-chunk 2048 "$@"
