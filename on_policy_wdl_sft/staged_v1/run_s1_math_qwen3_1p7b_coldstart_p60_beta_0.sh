#!/usr/bin/env bash
# Conservative math Stage1 wrapper from Qwen3-1.7B format cold-start SFT weights.
set -euo pipefail

if [ "${ALLOW_MATH_QWEN3_1P7B_COLDSTART_STAGE1:-0}" != "1" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    echo "[math-coldstart-s1] ERROR: guarded wrapper; set ALLOW_MATH_QWEN3_1P7B_COLDSTART_STAGE1=1 after validating 1.7B math Stage1 resource/config assumptions." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_PREFIX=${RUN_PREFIX:-ONPOLICY-SFT-Qwen3-1P7B-COLDSTART-MATH-S1-PLATEAU-P60-BETA0-V1}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-/data-1/model_weights/format_cold_start/qwen3-1p7b-math-format-sft}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "[math-coldstart-s1] DRY_RUN guarded wrapper would exec run_s1_base_sft.sh"
    echo "RUN_PREFIX=$RUN_PREFIX"
    echo "INIT_MODEL_PATH=$INIT_MODEL_PATH"
    echo "TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS"
    exit 0
fi

exec bash "${SCRIPT_DIR}/run_s1_base_sft.sh" "$@"
