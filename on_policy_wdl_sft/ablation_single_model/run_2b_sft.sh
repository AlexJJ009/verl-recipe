#!/usr/bin/env bash
# ==============================================================================
# Ablation 2B-SFT: single model, Qwen3-4B-Base-SFT-stage-1 init, wdl_sft_is, β=0.1.
# Compared with 1B and 2B-BASE. Critical test: with SFT init AND reverse SFT on
# incorrects, does single-model stability hold the way v2's binary mask claims?
# ==============================================================================
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-Qwen3-4B-MATH-2B-SFT"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"/data-1/.cache/Qwen3-4B-Base-SFT-stage-1"}
export LOSS_MODE=${LOSS_MODE:-"wdl_sft_is"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}
export LR=${LR:-5e-7}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_ablation.sh" "$@"
