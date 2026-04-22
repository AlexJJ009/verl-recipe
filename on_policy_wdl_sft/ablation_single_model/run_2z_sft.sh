#!/usr/bin/env bash
# ==============================================================================
# Ablation 2Z-SFT: single model, Qwen3-4B-Base-SFT-stage-1 init, LOSS=minirl.
# Pure RL baseline starting from the same SFT init that joint uses for model2.
# ==============================================================================
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"MINIRL-Qwen3-4B-MATH-2Z-SFT"}
export INIT_MODEL_PATH=${INIT_MODEL_PATH:-"/data-1/.cache/Qwen3-4B-Base-SFT-stage-1"}
export LOSS_MODE=${LOSS_MODE:-"minirl"}
export LR=${LR:-5e-7}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_ablation.sh" "$@"
