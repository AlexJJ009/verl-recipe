#!/usr/bin/env bash
# Monitor for Qwen3-1.7B KodCode CTX8K P40 Stage2 no-KL vs model2-only KL lambda sweep.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export QUEUE_MODE=kodcode_qwen3_1p7b_instruct_ctx8k_stage2_p40_m2kl_vs_nokl_lambda_sweep
export MONITOR_NAME=${MONITOR_NAME:-code_task_kodcode_qwen3_1p7b_instruct_ctx8k_stage2_p40_m2kl_vs_nokl_lambda_sweep}
export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_stage2_p40_m2kl_vs_nokl_lambda_sweep_notify.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_qwen3_1p7b_s2_p40_m2kl_vs_nokl_queue}
export STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-60}

exec bash "${SCRIPT_DIR}/monitor_code_task_queue_notify.sh" "$@"
