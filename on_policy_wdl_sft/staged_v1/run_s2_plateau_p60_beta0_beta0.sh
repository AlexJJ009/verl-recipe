#!/usr/bin/env bash
# Plateau handoff Stage 2: Stage 1 beta=0.0 step 60 -> Stage 2 beta=0.0.
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-STAGED-V1-S2-PLATEAU-P60-BETA0-BETA0"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-40}
export STAGE1_RUN_PREFIX=${STAGE1_RUN_PREFIX:-"ONPOLICY-SFT-Qwen3-4B-MATH-S1-PLATEAU-P60-BETA0-V1"}
export STAGE1_STEP=${STAGE1_STEP:-60}
export MERGED_MODEL2_DIR=${MERGED_MODEL2_DIR:-"/data-1/model_weights/staged_v1/plateau_handoff_p60/model2-from-s1-p60-beta0-step60"}
export REQUIRE_MERGED_MODEL2_PROVENANCE=${REQUIRE_MERGED_MODEL2_PROVENANCE:-True}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_run_stage2_model2_rollout_common.sh" "$@"
