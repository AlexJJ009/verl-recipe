#!/usr/bin/env bash
# Deprecated mixed-beta wrapper. Matched beta is required for boxed reruns.
set -xeuo pipefail

if [ "${ALLOW_MIXED_STAGE2:-0}" != "1" ]; then
    echo "ERROR: mixed Stage2 run is disabled. Use run_s2_from_s1_beta01_beta01.sh for matched beta=0.1." >&2
    echo "Set ALLOW_MIXED_STAGE2=1 only for an explicit ablation." >&2
    exit 1
fi

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-STAGED-V1-S2-FROM-S1-BETA01-BETA0"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export STAGE1_RUN_PREFIX=${STAGE1_RUN_PREFIX:-"ONPOLICY-SFT-Qwen3-4B-MATH-S1-BETA01-V1"}
export STAGE1_STEP=${STAGE1_STEP:-best}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_run_stage2_model2_rollout_common.sh" "$@"
