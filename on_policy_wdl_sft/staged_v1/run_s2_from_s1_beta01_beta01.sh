#!/usr/bin/env bash
# Optional Stage 2 fast validation wrapper: Stage 1 beta=0.1 best -> Stage 2 beta=0.1.
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-STAGED-V1-S2-BOXED-FROM-S1-BETA01-BETA01"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.1}
export STAGE1_RUN_PREFIX=${STAGE1_RUN_PREFIX:-"ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA01-V1"}
export STAGE1_STEP=${STAGE1_STEP:-best}
export MERGED_MODEL2_DIR=${MERGED_MODEL2_DIR:-"/data-1/model_weights/staged_v1/boxed_matched/model2-from-s1-boxed-beta01-best"}
export REQUIRE_MERGED_MODEL2_PROVENANCE=${REQUIRE_MERGED_MODEL2_PROVENANCE:-True}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_run_stage2_model2_rollout_common.sh" "$@"
