#!/usr/bin/env bash
# Optional Stage 2 fast validation wrapper: Stage 1 beta=0.0 best -> Stage 2 beta=0.1.
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-STAGED-V1-S2-FROM-S1-BETA0-BETA01"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}
export STAGE1_CKPT_DIR=${STAGE1_CKPT_DIR:-"/data-1/checkpoints/ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA0-V1_1779962803"}
export STAGE1_STEP=${STAGE1_STEP:-85}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_run_stage2_model2_rollout_common.sh" "$@"
