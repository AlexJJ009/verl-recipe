#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conditional endpoint only. A P60-specific authorization receipt is mandatory.
export DYNPERM_RHO=${DYNPERM_RHO:-1.0}
export TOTAL_TRAINING_STEPS=60
exec bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_dynperm_common.sh" "$@"
