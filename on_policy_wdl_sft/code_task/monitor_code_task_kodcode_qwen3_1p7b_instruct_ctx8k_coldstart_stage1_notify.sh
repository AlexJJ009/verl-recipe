#!/usr/bin/env bash
# Monitor for the KodCode Qwen3-1.7B CTX8K cold-start Stage1 queue.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export QUEUE_MODE=kodcode_qwen3_1p7b_coldstart_ctx8k_stage1
export MONITOR_NAME=${MONITOR_NAME:-code_task_kodcode_qwen3_1p7b_coldstart_ctx8k_stage1}
export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_code_task_kodcode_qwen3_1p7b_instruct_ctx8k_coldstart_stage1_notify.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_kodcode_qwen3_1p7b_coldstart_stage1_queue}

exec bash "${SCRIPT_DIR}/monitor_code_task_queue_notify.sh" "$@"
