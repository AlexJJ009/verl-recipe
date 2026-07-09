#!/usr/bin/env bash
# Monitor for the KodCode Instruct-2507 CTX8K P40 Stage2 submodel-KL lambda sweep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export QUEUE_MODE=kodcode_instruct2507_ctx8k_stage2_p40_subkl_lambda_sweep
export MONITOR_NAME=${MONITOR_NAME:-code_task_kodcode_instruct2507_ctx8k_stage2_p40_subkl_lambda_sweep}
export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_instruct2507_ctx8k_stage2_p40_subkl_lambda_sweep_notify.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_i2507_ctx8k_s2_p40_subkl_lambda_queue}
export STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}

exec bash "${SCRIPT_DIR}/monitor_code_task_queue_notify.sh" "$@"
