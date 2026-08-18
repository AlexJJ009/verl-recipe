#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUN_PREFIX=${RUN_PREFIX:-MATH-WDL-CAUSAL-P60-ARM-D0-QWEN3-1P7B}
export FUSION_LAMBDA=0.8
export FUSION_MODE=strong_scaled
export MODEL_PATH=${MODEL_PATH:-/data-1/.cache/huggingface/math-wdl-causal-p60-arm-d0}
exec bash "${SCRIPT_DIR}/run_math_qwen3_1p7b_wdl_causal_p60_common.sh" "$@"
