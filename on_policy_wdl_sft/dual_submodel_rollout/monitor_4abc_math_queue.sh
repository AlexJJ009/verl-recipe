#!/usr/bin/env bash
# Queue wrapper for 4A dual-model training followed by 4B/4C single-model runs.
#
# This file defines the 4ABC queue only. Generic queue mechanics live in the
# project-level script `scripts/training_queue_monitor.sh`.

set -euo pipefail

REPO_HOST=${REPO_HOST:-/data-1/verl07/verl-dual-rollout}
REPO_CONTAINER=${REPO_CONTAINER:-/workspace/verl}
CKPT_ROOT=${CKPT_ROOT:-/data-1/checkpoints}
MIN_FREE_GB=${MIN_FREE_GB:-160}
MAX_GPU_UTIL=${MAX_GPU_UTIL:-50}
FINAL_STEP=${FINAL_STEP:-115}
POLL_SEC=${POLL_SEC:-300}
LOG_FILE=${LOG_FILE:-"${REPO_HOST}/recipe/on_policy_wdl_sft/dual_submodel_rollout/monitor_4abc_math_queue.log"}

RUN_PREFIXES=(
    "WDL-SFT-Qwen3-4B-MATH-4A-DUAL-M2-GROUP-ADV-IS"
    "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4B-MATHDATA-BASE-E1"
    "WDL-GROUP-ADV-IS-Qwen3-4B-MATH-4C-MATHDATA-SFT-E1"
)
RUN_SCRIPTS=(
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/dual_submodel_rollout/run_4a_model2_group_adv_is.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/ablation_single_model/run_4b_math_base.sh"
    "${REPO_CONTAINER}/recipe/on_policy_wdl_sft/ablation_single_model/run_4c_math_sft.sh"
)
TMUX_NAMES=(
    "dual_model2_group_adv_is_4a"
    "single_model_group_adv_is_4b_base"
    "single_model_group_adv_is_4c_sft"
)

GENERIC_MONITOR=${GENERIC_MONITOR:-"${REPO_HOST}/scripts/training_queue_monitor.sh"}
[ -f "$GENERIC_MONITOR" ] || { echo "ERROR: generic queue monitor not found: $GENERIC_MONITOR" >&2; exit 1; }

# shellcheck disable=SC1090
source "$GENERIC_MONITOR"
training_queue_monitor_main "$@"
