#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
exec "${PYTHON:-python}" -m forecast_backbones.m4_pinball.evaluate "$@"
