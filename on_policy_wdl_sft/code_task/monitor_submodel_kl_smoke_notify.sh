#!/usr/bin/env bash
# Thin monitor for the submodel-KL smoke matrix queue.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export QUEUE_MODE=submodel_kl_smoke
export MONITOR_NAME=${MONITOR_NAME:-submodel_kl_smoke}
export QUEUE_TMUX=${QUEUE_TMUX:-submodel_kl_smoke_queue}
export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/monitor_submodel_kl_smoke_notify.log}
export STAGE2_TOTAL_TRAINING_STEPS=${STAGE2_TOTAL_TRAINING_STEPS:-${TOTAL_TRAINING_STEPS:-5}}
export WXPUSHER_NOTIFY=${WXPUSHER_NOTIFY:-0}

exec bash "${SCRIPT_DIR}/monitor_code_task_queue_notify.sh" "$@"
