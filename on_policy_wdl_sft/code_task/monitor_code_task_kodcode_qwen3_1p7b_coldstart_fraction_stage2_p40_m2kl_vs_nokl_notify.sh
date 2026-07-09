#!/usr/bin/env bash
# Monitor for Qwen3-1.7B KodCode cold-start fraction Stage2 P40 no-KL vs M2-only KL.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export QUEUE_MODE=kodcode_qwen3_1p7b_coldstart_fraction_stage2_p40_m2kl_vs_nokl
export MONITOR_NAME=${MONITOR_NAME:-code_task_kodcode_qwen3_1p7b_coldstart_fraction_stage2_p40_m2kl_vs_nokl}
export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_qwen3_1p7b_coldstart_fraction_stage2_p40_m2kl_vs_nokl_notify.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_qwen3_1p7b_coldstart_fraction_s2_p40_m2kl_vs_nokl_queue}
export STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}
export COLDSTART_FRACTION_STAGE2_FUSION_LAMBDAS=${COLDSTART_FRACTION_STAGE2_FUSION_LAMBDAS:-0.8}

exec bash "${SCRIPT_DIR}/monitor_code_task_queue_notify.sh" "$@"
