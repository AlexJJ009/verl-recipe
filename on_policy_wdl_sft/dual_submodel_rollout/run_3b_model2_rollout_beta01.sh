#!/usr/bin/env bash
# Dual-submodel rollout 3B: model2-selected rollout, wdl_sft_is, beta=0.1.
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-Qwen3-4B-MATH-3B-DUAL-M2-BETA01"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}
export LR=${LR:-5e-7}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_dual_rollout.sh" "$@"
