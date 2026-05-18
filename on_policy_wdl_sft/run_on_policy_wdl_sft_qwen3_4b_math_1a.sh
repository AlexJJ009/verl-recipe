#!/usr/bin/env bash
# ==============================================================================
# Post-fix Experiment 1A: joint WDL-SFT-IS, beta=0, lr=5e-7.
#
# This is the spec-correct rerun after the 2026-04-27 reward-label handoff fix.
# It writes a LABELFIX-prefixed run and will not resume old pre-fix 1A outputs.
# ==============================================================================
set -xeuo pipefail

export RUN_PREFIX=${RUN_PREFIX:-"WDL-SFT-Qwen3-4B-MATH-1A-LABELFIX"}
export WDL_SFT_BETA=${WDL_SFT_BETA:-0.0}
export LR=${LR:-5e-7}
export LOSS_MODE=${LOSS_MODE:-wdl_sft_is}

WRAPPER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WRAPPER_SCRIPT_DIR
# shellcheck disable=SC1091
source "${WRAPPER_SCRIPT_DIR}/_common_wdl_sft_is_joint.sh" "$@"
