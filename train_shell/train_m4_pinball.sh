#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# Select the environment with PYTHON; optional multi-GPU DDP via NPROC_PER_NODE.
PYTHON_BIN="${PYTHON:-python}"
if [[ "${NPROC_PER_NODE:-1}" -gt 1 ]]; then
  exec "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node "$NPROC_PER_NODE" \
    --master_addr "${MASTER_ADDR:-127.0.0.1}" --master_port "${MASTER_PORT:-29500}" \
    -m forecast_backbones.m4_pinball.train "$@"
else
  exec "$PYTHON_BIN" -m forecast_backbones.m4_pinball.train "$@"
fi
