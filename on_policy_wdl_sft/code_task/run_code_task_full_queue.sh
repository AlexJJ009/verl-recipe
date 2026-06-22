#!/usr/bin/env bash
# Host-side formal code-task Stage1 queue using KodCode-Light-RL-10K.
set -euo pipefail

export LOG_FILE=${LOG_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue.log}
export QUEUE_TMUX=${QUEUE_TMUX:-code_task_full_queue}
export QUEUE_MODE=full
export START_INDEX=${START_INDEX:-0}
export END_INDEX=${END_INDEX:-1}
export QUEUE_CONTINUE_ON_FAILURE=${QUEUE_CONTINUE_ON_FAILURE:-0}
export QUEUE_STATUS_FILE=${QUEUE_STATUS_FILE:-/data-1/verl07/verl/recipe/on_policy_wdl_sft/code_task/run_code_task_full_queue_status.tsv}

if [ "${DRY_RUN:-0}" != "1" ] && [ "${ALLOW_CODE_FULL_TRAINING:-0}" != "1" ]; then
    echo "[code-task full queue] ERROR: full queue requires explicit approval env ALLOW_CODE_FULL_TRAINING=1" >&2
    exit 1
fi
if [ "${ALLOW_CODE_FULL_TRAINING:-0}" = "1" ]; then
    export ALLOW_G2_TRAINING_SMOKE=${ALLOW_G2_TRAINING_SMOKE:-1}
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_code_task_smoke_queue.sh" "$@"
