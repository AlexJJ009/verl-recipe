#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Canonical M2 pilot: rho=1 by default; rho=0 exercises the exact no-op path.
export DYNPERM_RHO=${DYNPERM_RHO:-1.0}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
case "$TOTAL_TRAINING_STEPS" in
    20|30) ;;
    *) echo "ERROR: the DynPerm pilot is restricted to P20 or P30" >&2; exit 1 ;;
esac

exec bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_dynperm_common.sh" "$@"
