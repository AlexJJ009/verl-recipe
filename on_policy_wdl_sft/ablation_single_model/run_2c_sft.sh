#!/usr/bin/env bash
# ==============================================================================
# Ablation 2C-SFT: single model, Qwen3-4B-Base-SFT-stage-1 init, wdl_sft_is,
# β=0, lr=1e-6. Compared with 1C and 2C-BASE.
# ==============================================================================
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-Qwen3-4B-MATH-2C-SFT-LABELFIX"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"/data-1/.cache/Qwen3-4B-Base-SFT-stage-1"}
export LOSS_MODE=${LOSS_MODE:-"wdl_sft_is"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export LR=${LR:-1e-6}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_ablation.sh" "$@"
