#!/usr/bin/env bash
# Stage 2 fast validation: Stage 1 beta=0.0 best -> Stage 2 beta=0.0.
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-STAGED-V1-S2-BOXED-FROM-S1-BETA0-BETA0"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export STAGE1_RUN_PREFIX=${STAGE1_RUN_PREFIX:-"ONPOLICY-SFT-Qwen3-4B-MATH-S1-BOXED-BETA0-V1"}
export STAGE1_STEP=${STAGE1_STEP:-best}
export MERGED_MODEL2_DIR=${MERGED_MODEL2_DIR:-"/data-1/model_weights/staged_v1/boxed_matched/model2-from-s1-boxed-beta0-best"}
export REQUIRE_MERGED_MODEL2_PROVENANCE=${REQUIRE_MERGED_MODEL2_PROVENANCE:-True}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_run_stage2_model2_rollout_common.sh" "$@"
