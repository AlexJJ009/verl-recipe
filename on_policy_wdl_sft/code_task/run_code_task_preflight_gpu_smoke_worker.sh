#!/usr/bin/env bash
# Worker contract for bounded local GPU calibration. This is never release evidence.
set -euo pipefail

MANIFEST=${1:?manifest required}
OUTPUT_ROOT=${2:?output root required}
[ -f "$MANIFEST" ] || { echo "ERROR: missing manifest" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT"
printf '{"schema_version":1,"evidence_class":"infrastructure_preflight","status":"worker_contract_ready","manifest":"%s"}\n' \
  "$MANIFEST" > "$OUTPUT_ROOT/worker_status.json"

if [ "${ALLOW_CODE_PREFLIGHT_GPU_SMOKE_WORKER_EXECUTION:-0}" != 1 ]; then
  echo "ERROR: worker execution requires ALLOW_CODE_PREFLIGHT_GPU_SMOKE_WORKER_EXECUTION=1" >&2
  exit 1
fi

echo "ERROR: real GPU calibration implementation is introduced only after sandbox benchmark acceptance" >&2
exit 1
