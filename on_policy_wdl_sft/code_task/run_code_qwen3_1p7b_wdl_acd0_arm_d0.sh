#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_PREFIX=${RUN_PREFIX:-CODE-WDL-ACD0-P60-ARM-D0-QWEN3-1P7B}
export FUSION_LAMBDA=0.8
export FUSION_MODE=strong_scaled
exec bash "${SCRIPT_DIR}/run_code_qwen3_1p7b_wdl_acd0_joint_common.sh" "$@"
